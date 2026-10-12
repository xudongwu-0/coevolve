"""Paired real H->W continuation: external proposer versus trained local 2B."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

from alternating_common import (ROOT, check_proposals, evaluate_proposals, evidence,
    identity, read, requests_for, rows, run_editor, sha, source_snapshot, spec, write_json)
from alternating_cycles import explored_proposals
from alternating_pool import Pool
from external_harness_replay import proposer, score as score_external
from harness_bootstrap import START
from harness_code_contract import validate_source
from harness_focused_bootstrap import choose_focus
from meta_harness.benchmark import compute_pareto_frontier
from meta_harness.meta_harness_retool import pick_accepted_from_frontier
from pipeline import api


def utc(): return datetime.datetime.now(datetime.timezone.utc).isoformat()


def status(run, stage, **extra):
    value = {'stage': stage, 'utc': utc(), **extra}
    write_json(run / 'STATUS.json', value); print(json.dumps(value), flush=True)


def prepare(run):
    assert not run.exists()
    run.mkdir(parents=True)
    old = ROOT / 'runs/whale_alternating_ready_20260916'
    qualified = ROOT / 'runs/whale_math_mean8_native_20260915'
    assert read(qualified / 'verification.json')['passed_audit']
    assert read(qualified / 'verification.json')['qualified_for_gate']
    prior = read(old / 'config.json')
    model = qualified / 'round_01/checkpoint'
    editor = ROOT / 'runs/whale_dual2b_focused_20260916/editor_continuation/checkpoint'
    base = run / 'initial_harness.py'; base.write_text(START)
    data = {k: v for k, v in prior['data']['splits'].items()
        if k in ['reward_0', 'reward_1', 'selection', 'train', 'monitor']}
    data['test'] = read(qualified / 'config.json')['data']['splits']['test']
    predictions = [str(run / arm / f'cycle_{n:02d}/solver_update/round_01/checkpoint')
        for n in [1, 2] for arm in ['internal', 'external']]
    config = {'cycles': 2, 'arms': ['internal', 'external'], 'solver_initial_checkpoint': str(model),
        'initial_solver_identity': identity(model), 'editor_initial_checkpoint': str(editor),
        'initial_editor_identity': identity(editor), 'initial_harness': str(base),
        'initial_harness_sha256': sha(base), 'data': {'splits': data},
        'pool_gpus': ['8', '9'] + [f'node1:{i}' for i in range(10)],
        'checkpoint_allowlist': [str(model)] + predictions, 'editor_gpu': '6',
        'solver_training_gpus': {'internal': '6', 'external': '7'},
        'shared_prefix': {'run': str(qualified), 'verification_sha256': sha(qualified / 'verification.json'),
            'checkpoint_state_sha256': sha(model / 'training_state.json'),
            'charged_train_trajectories_per_arm': read(model / 'training_state.json')['rollouts'],
            'new_prefix_training_updates': 0,
            'editable_initial_harness': 'CandidateEnv inherits ReToolEnv, only explicitly sets the unchanged MAX_TURNS=2.'},
        'solver_rsft_lr': 3e-7, 'solver_draws': 1, 'solver_temperature': 1.0,
        'editor_grpo_epochs': 4, 'editor_grpo_lr': 1e-6, 'proposal_temperature': 1.0,
        'pre_proposals_per_group': 8, 'post_proposals_per_group': 4, 'groups': 2,
        'selection_candidates': 2, 'evaluation_seed': 42, 'evaluation_draws': 1,
        'proposer_port': 18542,
        'search_space': 'Both arms may change SYSTEM_PROMPT, USER_PROMPT_TEMPLATE, MAX_TURNS 1..3, no_tools_called and get_prompt_messages; identical executable-code contract, no prompt-only restriction.',
        'budget_rule': 'Internal distinct evaluated candidate/data pairs determine the external cap separately by group and pre/post stage. Duplicate/no-op/invalid slots cost no new target rollout. Same train128 per cycle and selection24 per shortlisted candidate.',
        'zero_variance_rule': 'No fabricated editor update, no extra exploration or teacher correction.',
        'evaluation_rule': 'All search and training finish before opening current monitoring or test scores. One seed, one draw. Same W/H states reuse one evaluation.',
        'limits': ['Small MATH number_theory Level2-4 subset, not paper DAPO/AIME training.',
            '2048 output /1024 per call /4096 context, thinking disabled, max3 turns; paper math uses 8192 output and allows16 turns.',
            'Shared audited weight warm start; editor synthetic SFT initialization costs are inherited and reported separately.',
            'Existing monitor32 and test128 have historical use; this is not an untouched independent test.',
            'Active conversation is the external proposer; external thinking/FLOPs/dollar cost is unmetered.']}
    write_json(run / 'config.json', config)
    write_json(run / 'PROTOCOL.json', {'registered_utc': utc(), 'config_sha256': sha(run / 'config.json'),
        'new_train_trajectories_per_arm': 256, 'new_weight_updates_under_own_selected_harness_per_arm': 2,
        'shared_prefix_charged_equally': True, 'proposal_slots_per_arm': 48,
        'held_fixed_within_each_phase': True, 'actual_execution': 'shared W1 -> H1 -> train W2 on H1 -> H2 -> train W3 on H2',
        'no_replay_of_other_arm_trained_weights': True, 'external_receives_no_internal_candidate_source_or_outcome': True,
        'sampler_budget_caps_shared_as_counts_only': True,
        'whale_paper': 'https://arxiv.org/html/2609.00196v1'})
    source_snapshot(run, 'frozen', ['paired_joint_whale.py', 'alternating_cycles.py',
        'alternating_solver_train.py', 'external_harness_replay.py', 'proposer_api.py'])
    for arm in config['arms']: (run / arm).mkdir()
    status(run, 'prepared')


def select(pool, cycle, config, model, incumbent, directory, scored, limit):
    unique = {}
    valid = [p for p in scored if p['outcome_kind'] == 'valid']
    for p in sorted(valid, key=lambda p: (-p['reward_delta'], p['id'])):
        unique.setdefault(hashlib.sha256(p['source'].encode()).hexdigest(), p)
    shortlist = list(unique.values())[:limit]
    choices = {'incumbent': Path(incumbent)}
    choices.update({p['id']: directory / (p['id'] + '.py') for p in shortlist})
    dest = cycle / 'selection'
    for name, path in choices.items():
        target = dest / 'harnesses' / name / 'harness.py'
        target.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(path, target)
    results = pool.evaluate_many([spec(model, path, config['data']['splits']['selection'],
        dest / 'results' / name, seed=42, temperature=1.0) for name, path in choices.items()])
    values = dict(zip(choices, results))
    frontier = {'_pareto': [{'harness': name, 'avg_success_rate': success, 'avg_mean_turn_count': turns}
        for name, success, turns in compute_pareto_frontier([(name, r['success_rate'], r['mean_turn_count'])
            for name, r in values.items()])]}
    _, accepted = pick_accepted_from_frontier(frontier, dest); assert accepted
    selected = dest / 'harnesses' / accepted / 'harness.py'
    write_json(dest / 'decision.json', {'accepted': accepted, 'shortlist_ids': [p['id'] for p in shortlist],
        'results': values, 'frontier': frontier, 'harness': str(selected), 'sha256': sha(selected),
        'changed': sha(selected) != sha(incumbent), 'selection_candidate_limit': limit,
        'test_feedback_used': False})
    return selected, len(shortlist)


def cases_and_requests(pool, cycle, config, model, harness, count, inherited=None):
    number = int(cycle.name[-2:])
    cases = {f'reward_{i}': {'data': config['data']['splits'][f'reward_{i}'], 'harness': str(harness),
        'baseline': str(cycle / 'baselines' / f'reward_{i}'), 'seed': 142 + number} for i in range(2)}
    if inherited:
        for group, c in cases.items():
            c['baseline'] = inherited[group]['baseline']
        write_json(cycle / 'SHARED_BASELINE.json', {'cases': cases, 'logical_trajectories': 16,
            'physical_new_trajectories': 0, 'reason': 'identical starting W/H/data/seed'})
    else:
        pool.evaluate_many([spec(model, harness, c['data'], c['baseline'], seed=c['seed'], temperature=1.0)
            for c in cases.values()])
    requests = []
    for i, (group, case) in enumerate(cases.items()):
        focus = choose_focus(rows(Path(case['baseline']) / 'trajectories.jsonl'), harness.read_text(), i)
        requests += requests_for(group, harness.read_text(), case['baseline'], focus, count, 94200 + number * 1000 + i * 100)
    write_json(cycle / 'CASES.json', cases)
    return cases, requests


def allocation(scored, seen):
    counts = [0, 0]
    for p in scored:
        path = p['reward_provenance']['candidate']
        if path and path not in seen:
            seen.add(path); counts[int(p['group'][-1])] += 1
    return counts


def external_proposals(run, cycle, url, stage, requests, incumbent, caps, previous=None):
    directory = cycle / (stage + '_proposals'); directory.mkdir()
    grouped = []
    for group in ['reward_0', 'reward_1']:
        packet = [r for r in requests if r['group'] == group]
        grouped.append({'group': group, 'ids': [r['id'] for r in packet], 'messages': packet[0]['messages']})
    payload = {'task_prompt': 'Act as the external WHALE proposer. Write complete CandidateEnv modules using the supplied API and edit focus. '
        'Executable hook/feedback/termination edits are allowed, not only prompt edits. Do not change verifier, tools, budget enforcement or solve individual tasks. '
        'Meet the actual new candidate/data evaluation caps. Reuse identical previously scored source when necessary; do not create trivial whitespace variants. '
        'Return candidates with id,group,source,hypothesis. Each group must contain exactly the listed slots.',
        'run_dir': str(directory), 'iteration': int(cycle.name[-2:]), 'accepted_harness_source': incumbent.read_text(),
        'gate_context': {'stage': stage, 'requests': grouped, 'new_evaluation_caps': caps,
            'prior_development_results': [{k: p.get(k) for k in ['id', 'group', 'source', 'reward_delta', 'outcome_kind']}
                for p in (previous or [])]}, 'frontier': {}}
    submitted = api(url, '/jobs', payload); write_json(directory / 'request_receipt.json', submitted)
    status(run, 'awaiting_active_agent', cycle=int(cycle.name[-2:]), phase=stage, job_id=submitted['job_id'])
    deadline = time.monotonic() + 3600
    while True:
        response = api(url, '/jobs/' + submitted['job_id'])
        if 'candidates' in response: break
        if time.monotonic() > deadline: raise TimeoutError('external proposer')
        time.sleep(1)
    expected = {(r['group'], r['id']) for r in requests}
    assert len(response['candidates']) == len(expected)
    assert {(p['group'], p['id']) for p in response['candidates']} == expected
    result = []
    for p in response['candidates']:
        try: source, valid, error = validate_source(p['source']), True, None
        except Exception as exc: source, valid, error = p.get('source', ''), False, repr(exc)
        row = dict(p, source=source, valid_source=valid, validation_error=error)
        (directory / (p['id'] + '.py')).write_text(source); result.append(row)
    (directory / 'generations.jsonl').write_text(''.join(json.dumps(p) + '\n' for p in result))
    write_json(directory / 'response_receipt.json', {'job_id': submitted['job_id'],
        'response_sha256': sha(run / 'proposer_jobs' / submitted['job_id'] / 'response.json')})
    return directory, result


def train_solver(cycle, config, model, harness, gpu):
    start = time.monotonic(); output = cycle / 'solver_update'; output.mkdir()
    (output / 'calls').mkdir()
    source = cycle / 'solver_rollouts/trajectories.jsonl'
    for r in rows(source):
        for c in r['policy_calls']: write_json(output / 'calls' / (c['call_id'] + '.json'), c)
    target = output / 'rollouts.jsonl'; shutil.copy2(source, target)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu, WHALE_RUN_ROOT=str(output), WHALE_MODEL_ROOT=str(model),
        WHALE_PARAMETER_DTYPE='float32', WHALE_LEARNING_RATE=str(config['solver_rsft_lr']), WHALE_SAFE_GRAD_NORM='1',
        HF_HUB_OFFLINE='1', TOKENIZERS_PARALLELISM='false', PYTHONPATH=':'.join(str(ROOT / p)
            for p in ['whale_local', 'WHALE/domains/math_reasoning', 'WHALE/domains/chess_puzzles']))
    with (output / 'train.log').open('a') as log:
        subprocess.run([str(ROOT / '.venv-whale-portable/bin/python'), '-u', str(ROOT / 'whale_local/alternating_solver_train.py'),
            '--rollouts', str(target), '--data', config['data']['splits']['train']['path'], '--harness', str(harness),
            '--expected-digest', identity(model)['weight_digest']], env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
    checkpoint = Path(read(output / 'complete.json')['identity']['weights'])
    write_json(output / 'TIMING.json', {'elapsed_seconds': time.monotonic() - start, 'gpu': gpu})
    return checkpoint


def execute(run):
    config = read(run / 'config.json'); assert sha(run / 'config.json') == read(run / 'PROTOCOL.json')['config_sha256']
    started = time.monotonic(); write_json(run / 'START.json', {'utc': utc()})
    models = {a: Path(config['solver_initial_checkpoint']) for a in config['arms']}
    harnesses = {a: Path(config['initial_harness']) for a in config['arms']}
    editor = Path(config['editor_initial_checkpoint'])
    points = {a: [{'label': 'initial', 'model': str(models[a]), 'harness': str(harnesses[a])}] for a in config['arms']}
    try:
        with proposer(run, config['proposer_port']) as url, Pool(run / 'pool', config) as pool:
            for number in range(1, 3):
                cycles = {a: run / a / f'cycle_{number:02d}' for a in config['arms']}
                for a, c in cycles.items():
                    c.mkdir(); write_json(c / 'START.json', {'solver': identity(models[a]), 'harness': str(harnesses[a]),
                        'harness_sha256': sha(harnesses[a]), 'editor': identity(editor) if a == 'internal' else None})
                cycle = cycles['internal']; seen = set()
                status(run, 'internal_baselines', cycle=number)
                internal_cases, requests = cases_and_requests(pool, cycle, config, models['internal'], harnesses['internal'], 8)
                status(run, 'internal_pre_generation', cycle=number)
                pre = cycle / 'pre_proposals'; explored_proposals(editor, pre, requests, config['proposal_temperature'])
                scored, summary = evaluate_proposals(pool, run / 'internal', pre, models['internal'], internal_cases)
                caps_pre = allocation(scored, seen)
                status(run, 'internal_editor_grpo', cycle=number)
                trained = run_editor(run, editor, cycle / 'editor_grpo', scored, mode='train', epochs=4)
                prior_editor = editor
                if trained['updated']: editor = Path(trained['identity']['weights'])
                post_requests = [r for r in requests if int(r['id'][-2:]) < 4]
                status(run, 'internal_post_generation', cycle=number, editor_updated=trained['updated'])
                post = cycle / 'post_proposals'; explored_proposals(editor, post, post_requests, config['proposal_temperature'])
                scored_post, post_summary = evaluate_proposals(pool, run / 'internal', post, models['internal'], internal_cases)
                caps_post = allocation(scored_post, seen)
                status(run, 'internal_selection', cycle=number)
                selected, selection_limit = select(pool, cycle, config, models['internal'], harnesses['internal'], post, scored_post, 2)
                quotas = {'pre': caps_pre, 'post': caps_post, 'selection_candidates': selection_limit}
                write_json(cycle / 'ALLOCATION.json', quotas)
                write_json(cycle / 'EDITOR_RESULT.json', {'prior': identity(prior_editor), 'after': identity(editor),
                    'updated': trained['updated'], 'pre': summary, 'post': post_summary})
                original_internal_harness = harnesses['internal']; harnesses['internal'] = selected
                cycle = cycles['external']; status(run, 'external_baselines', cycle=number)
                same = (identity(models['internal']) == identity(models['external']) and
                    sha(original_internal_harness) == sha(harnesses['external']))
                cases, requests = cases_and_requests(pool, cycle, config, models['external'], harnesses['external'], 8,
                    inherited=internal_cases if same else None)
                write_json(cycle / 'ALLOCATION.json', quotas)
                temporary_config = dict(config, new_evaluation_caps_per_group={str(number): {'pre': caps_pre, 'post': caps_post}})
                pre, generated = external_proposals(run, cycle, url, 'pre', requests, harnesses['external'], caps_pre)
                scored = score_external(pool, run / 'external', cycle, temporary_config, models['external'],
                    harnesses['external'], cases, 'pre', pre, generated)
                post_requests = [r for r in requests if int(r['id'][-2:]) < 4]
                post, generated = external_proposals(run, cycle, url, 'post', post_requests, harnesses['external'], caps_post, scored)
                scored_post = score_external(pool, run / 'external', cycle, temporary_config, models['external'],
                    harnesses['external'], cases, 'post', post, generated)
                status(run, 'external_selection', cycle=number)
                harnesses['external'], _ = select(pool, cycle, config, models['external'], harnesses['external'], post, scored_post, selection_limit)
                for arm in config['arms']:
                    points[arm].append({'label': f'cycle_{number:02d}_after_harness', 'model': str(models[arm]), 'harness': str(harnesses[arm])})
                    status(run, 'solver_sampling_under_selected_harness', cycle=number, arm=arm)
                    pool.evaluate_many([spec(models[arm], harnesses[arm], config['data']['splits']['train'],
                        cycles[arm] / 'solver_rollouts', seed=431 + number, temperature=config['solver_temperature'])])
                status(run, 'both_solver_training', cycle=number)
                with ThreadPoolExecutor(max_workers=2) as executor:
                    futures = {a: executor.submit(train_solver, cycles[a], config, models[a], harnesses[a], config['solver_training_gpus'][a])
                        for a in config['arms']}
                    for arm, future in futures.items(): models[arm] = future.result()
                for arm in config['arms']:
                    points[arm].append({'label': f'cycle_{number:02d}_after_weights', 'model': str(models[arm]), 'harness': str(harnesses[arm])})
                    write_json(cycles[arm] / 'complete.json', {'solver': identity(models[arm]), 'harness': str(harnesses[arm]),
                        'harness_sha256': sha(harnesses[arm]), 'trained_on_selected_harness': True})
                write_json(run / 'stage_identities.json', points)
            write_json(run / 'FROZEN.json', {'utc': utc(), 'points': points, 'editor_final': identity(editor),
                'search_training_elapsed_seconds': time.monotonic() - started, 'monitor_or_test_feedback_used': False})
            cache = {}; curve = {}
            for arm, arm_points in points.items():
                curve[arm] = []
                for p in arm_points:
                    key = (identity(p['model'])['weight_digest'], sha(p['harness']))
                    if key not in cache:
                        output = run / 'monitor' / arm / p['label']
                        status(run, 'single_seed_monitor', arm=arm, label=p['label'])
                        pool.evaluate_many([spec(p['model'], p['harness'], config['data']['splits']['monitor'], output,
                            seed=42, temperature=1.0)])
                        cache[key] = str(output)
                    curve[arm].append({**p, 'weight_digest': key[0], 'harness_sha256': key[1], 'output': cache[key]})
                write_json(run / 'monitor_curve_sources.json', curve)
            final = {'shared_start': points['internal'][0], **{a: points[a][-1] for a in config['arms']}}
            for name, p in final.items():
                status(run, 'single_seed_final_test', arm=name)
                pool.evaluate_many([spec(p['model'], p['harness'], config['data']['splits']['test'], run / 'test' / name,
                    seed=42, temperature=1.0)])
            write_json(run / 'FINAL_MODELS.json', {a: {'solver': identity(models[a]), 'harness': str(harnesses[a]),
                'harness_sha256': sha(harnesses[a])} for a in config['arms']})
        write_json(run / 'COMPLETE.json', {'utc': utc(), 'elapsed_seconds': time.monotonic() - started,
            'new_solver_checkpoints': 4, 'internal_editor_final': identity(editor)})
        status(run, 'complete_pending_audit')
    except BaseException as exc:
        status(run, 'failed', error=repr(exc)); raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('action', choices=['prepare', 'run'])
    parser.add_argument('--run', type=Path, required=True); args = parser.parse_args()
    if args.action == 'prepare': prepare(args.run.resolve())
    else: execute(args.run.resolve())

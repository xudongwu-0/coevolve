"""Budget-matched active-agent harness search with frozen W2/W3/W4 replay."""
import argparse
import ast
from contextlib import contextmanager
import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from alternating_common import evidence, identity, read, rows, source_snapshot, spec
from alternating_pool import Pool
from harness_code_contract import validate_source
from harness_focused_bootstrap import request
from pipeline import ROOT, api, sha, wait_ready, write_json
from meta_harness.benchmark import compute_pareto_frontier
from meta_harness.meta_harness_retool import pick_accepted_from_frontier

REFERENCE = ROOT / 'runs/whale_alternating_ready_20260916'
EXPANDED = ROOT / 'runs/whale_expanded_eval_20260916'


def utc():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def status(run, stage, **extra):
    value = {'stage': stage, 'utc': utc(), **extra}
    write_json(run / 'STATUS.json', value)
    print(json.dumps(value), flush=True)


def without_system(source):
    tree = ast.parse(source)
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef))
    cls.body = [n for n in cls.body if not (isinstance(n, ast.Assign) and
        isinstance(n.targets[0], ast.Name) and n.targets[0].id == 'SYSTEM_PROMPT')]
    return ast.dump(tree, include_attributes=False)


def prepare(run, gpus):
    assert not run.exists(), 'Use a new run directory'
    run.mkdir(parents=True)
    original, expanded = read(REFERENCE / 'config.json'), read(EXPANDED / 'config.json')
    assert read(REFERENCE / 'verification.json')['passed_audit']
    assert read(EXPANDED / 'verification.json')['passed_audit']
    config = {'reference': str(REFERENCE), 'expanded_reference': str(EXPANDED),
        'reference_audit_sha256': sha(REFERENCE / 'verification.json'),
        'expanded_audit_sha256': sha(EXPANDED / 'verification.json'),
        'models': expanded['models'], 'solver_initial_checkpoint': original['solver_initial_checkpoint'],
        'checkpoint_allowlist': expanded['checkpoint_allowlist'], 'initial_harness': original['initial_harness'],
        'initial_harness_sha256': sha(original['initial_harness']), 'pool_gpus': gpus,
        'data': {'splits': {k: v for k, v in original['data']['splits'].items()
            if k in ['reward_0', 'reward_1', 'selection']} | expanded['data']['splits']},
        'cohorts': expanded['cohort_order'], 'cycles': 2, 'seed': 42, 'temperature': 1.0,
        'draws': 1, 'training_updates': 0, 'proposer_port': 18542,
        'proposal_slots': {'pre': 8, 'post': 4},
        'new_evaluation_caps_per_group': {'1': {'pre': [6, 5], 'post': [0, 0]},
                                           '2': {'pre': [3, 5], 'post': [1, 1]}},
        'selection_candidates_per_cycle': 2,
        'edit_scope': 'Only SYSTEM_PROMPT; preserve all hooks, MAX_TURNS and other code AST.',
        'budget_meaning': 'Match proposal slots and cap actual distinct candidate/dataset evaluations at reference usage; cached evaluations cost zero. External reasoning/token/financial cost is not matched.',
        'limitations': ['Frozen weight replay, not retraining under new harnesses.',
            'All 173 evaluation questions have historical use; evaluation feedback is withheld until both selections finish.',
            'Active current conversation authors proposals; this is not an unattended Codex API.',
            'The active conversation has seen prior aggregate experiment results; it receives no new evaluation task text or answers.']}
    write_json(run / 'config.json', config)
    write_json(run / 'PROTOCOL.json', {'registered_utc': utc(), 'config_sha256': sha(run / 'config.json'),
        'proposal_slots_per_cycle': 24, 'candidate_reward_trajectory_caps': [88, 80],
        'baseline_trajectories_per_cycle': 16, 'selection_trajectories_per_cycle': 72,
        'selection_rule': 'Original WHALE Pareto frontier, highest success, then fewer turns.',
        'unevaluated_candidates_ineligible': True, 'test_feedback_used_for_search': False,
        'evaluation': 'Draw zero, seed 42, temperature 1.0, 2048 total / 1024 per-call tokens, same cohort indexing as prior evaluation.',
        'frozen_weight_path': ['W2_initial', 'W3_cycle1', 'W4_cycle2']})
    source_snapshot(run, 'frozen', ['external_harness_replay.py', 'proposer_api.py'])
    status(run, 'prepared')


@contextmanager
def proposer(run, port):
    with (run / 'proposer_api.log').open('a') as log:
        process = subprocess.Popen([sys.executable, '-u', str(ROOT / 'whale_local/proposer_api.py')],
            env=dict(os.environ, WHALE_PROPOSER_JOBS=str(run / 'proposer_jobs'), WHALE_PROPOSER_PORT=str(port)),
            stdout=log, stderr=subprocess.STDOUT)
        try:
            url = f'http://127.0.0.1:{port}'
            ready = wait_ready(process, url)
            assert ready['jobs'] == str(run / 'proposer_jobs')
            yield url
        finally:
            process.terminate()
            process.wait(timeout=15)
            write_json(run / 'proposer_stopped.json', {'returncode': process.returncode})


def proposals(run, cycle, config, url, stage, harness, cases, previous=None):
    number = int(cycle.name[-2:])
    directory = cycle / (stage + '_proposals')
    directory.mkdir()
    count = config['proposal_slots'][stage]
    requests = [{'group': g, 'messages': request(harness.read_text(), evidence(c['baseline']), 'strategy'),
        'ids': [f'{g}_{i:02d}' for i in range(count)]} for g, c in cases.items()]
    payload = {'task_prompt': 'Author complete generic harness modules for each slot. Only SYSTEM_PROMPT may change. '
        'Do not access evaluation tasks or reference answers. Submit candidates with id, group, source, hypothesis. '
        'Candidates are evaluated in submitted order within each group until the registered new-evaluation cap. '
        'At the post stage, cached pre candidates may be reused for free and remain eligible; unseen over-budget candidates are ineligible.',
        'run_dir': str(directory), 'iteration': number,
        'accepted_harness_source': harness.read_text(),
        'gate_context': {'stage': stage, 'requests': requests,
            'new_evaluation_caps': config['new_evaluation_caps_per_group'][str(number)][stage],
            'prior_development_results': [{k: p[k] for k in ['id', 'group', 'source', 'reward_delta', 'outcome_kind']}
                for p in (previous or [])]}, 'frontier': {}}
    submitted = api(url, '/jobs', payload)
    write_json(directory / 'request_receipt.json', submitted)
    status(run, 'awaiting_active_agent', cycle=number, phase=stage, job_id=submitted['job_id'])
    deadline = time.monotonic() + 3600
    while True:
        response = api(url, '/jobs/' + submitted['job_id'])
        if 'candidates' in response:
            break
        if time.monotonic() > deadline:
            raise TimeoutError('Active agent proposal response')
        time.sleep(1)
    expected = {(r['group'], name) for r in requests for name in r['ids']}
    assert len(response['candidates']) == len(expected)
    assert {(p['group'], p['id']) for p in response['candidates']} == expected
    output = []
    for p in response['candidates']:
        try:
            source = validate_source(p['source'])
            assert without_system(source) == without_system(harness.read_text()), 'Edit scope exceeded'
            valid, error = True, None
        except Exception as exc:
            source, valid, error = p['source'], False, repr(exc)
        item = dict(p, source=source, valid_source=valid, validation_error=error)
        (directory / (p['id'] + '.py')).write_text(source)
        output.append(item)
    (directory / 'generations.jsonl').write_text(''.join(json.dumps(x) + '\n' for x in output))
    write_json(directory / 'response_receipt.json', {'job_id': submitted['job_id'],
        'response_sha256': sha(run / 'proposer_jobs' / submitted['job_id'] / 'response.json')})
    return directory, output


def score(pool, run, cycle, config, model, harness, cases, stage, directory, proposals):
    from alternating_common import check_proposals
    checks = check_proposals(directory, cases['reward_0']['data'])
    caps = config['new_evaluation_caps_per_group'][str(int(cycle.name[-2:]))][stage]
    counts, jobs, mapped = [0, 0], {}, {}
    for p in proposals:
        group = int(p['group'][-1]); case = cases[p['group']]
        if not p['valid_source'] or not checks[p['id']]['passed']:
            mapped[p['id']] = ('invalid', None); continue
        if p['source'].strip() == harness.read_text().strip():
            mapped[p['id']] = ('no_op', None); continue
        key = hashlib.sha256(json.dumps({'source': p['source'], 'model': identity(model)['weight_digest'],
            'data': case['data']['sha256'], 'seed': case['seed']}, sort_keys=True).encode()).hexdigest()
        output = run / 'outcome_cache' / key
        if not (output / 'val.json').exists() and key not in jobs:
            if counts[group] >= caps[group]:
                mapped[p['id']] = ('budget_excluded', None); continue
            counts[group] += 1
            jobs[key] = spec(model, directory / (p['id'] + '.py'), case['data'], output,
                seed=case['seed'], temperature=1.0)
        mapped[p['id']] = ('valid', output)
    status(run, 'scoring_candidates', cycle=int(cycle.name[-2:]), phase=stage, new_evaluations=counts)
    pool.evaluate_many(list(jobs.values()))
    scored = []
    for p in proposals:
        kind, output = mapped[p['id']]; base = read(Path(cases[p['group']]['baseline']) / 'val.json')
        value = read(output / 'val.json') if output else None
        scored.append(dict(p, outcome_kind=kind,
            reward_delta=value['success_rate'] - base['success_rate'] if value else None,
            reward_provenance={'candidate': str(output) if output else None,
                'baseline': cases[p['group']]['baseline'], 'candidate_score': value['success_rate'] if value else None,
                'baseline_score': base['success_rate']}))
    (directory / 'scored.jsonl').write_text(''.join(json.dumps(x) + '\n' for x in scored))
    write_json(directory / 'budget.json', {'caps': caps, 'new_evaluations': counts,
        'new_trajectories': sum(counts) * 8, 'proposals': len(scored)})
    return scored


def select(pool, cycle, config, model, incumbent, directory, scored):
    unique = {}
    for p in sorted((x for x in scored if x['outcome_kind'] == 'valid'), key=lambda x: (-x['reward_delta'], x['id'])):
        unique.setdefault(hashlib.sha256(p['source'].encode()).hexdigest(), p)
    shortlist = list(unique.values())[:2]
    assert len(shortlist) == 2, 'Need two eligible distinct candidates'
    selection = cycle / 'selection'
    choices = {'incumbent': incumbent} | {p['id']: directory / (p['id'] + '.py') for p in shortlist}
    for name, source in choices.items():
        target = selection / 'harnesses' / name / 'harness.py'
        target.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(source, target)
    values = dict(zip(choices, pool.evaluate_many([spec(model, path, config['data']['splits']['selection'],
        selection / 'results' / name, seed=42, temperature=1.0) for name, path in choices.items()])))
    points = [(k, v['success_rate'], v['mean_turn_count']) for k, v in values.items()]
    frontier = {'_pareto': [{'harness': n, 'avg_success_rate': s, 'avg_mean_turn_count': t}
        for n, s, t in compute_pareto_frontier(points)]}
    _, accepted = pick_accepted_from_frontier(frontier, selection)
    assert accepted
    selected = selection / 'harnesses' / accepted / 'harness.py'
    write_json(selection / 'decision.json', {'accepted': accepted, 'results': values, 'frontier': frontier,
        'shortlist_ids': [p['id'] for p in shortlist], 'selected_sha256': sha(selected),
        'test_feedback_used': False, 'changed': sha(selected) != sha(incumbent)})
    return selected


def execute(run):
    config = read(run / 'config.json'); assert sha(run / 'config.json') == read(run / 'PROTOCOL.json')['config_sha256']
    started = time.monotonic(); write_json(run / 'START.json', {'utc': utc()})
    harness = Path(config['initial_harness']); chosen = []
    try:
        with proposer(run, config['proposer_port']) as url, Pool(run / 'pool', config) as pool:
            for number, model_name in enumerate(['W2_initial', 'W3_cycle1'], 1):
                cycle = run / f'cycle_{number:02d}'; cycle.mkdir()
                model = Path(config['models'][model_name]['model'])
                cases = {f'reward_{i}': {'data': config['data']['splits'][f'reward_{i}'],
                    'baseline': str(cycle / 'baselines' / f'reward_{i}'), 'seed': 142 + number} for i in range(2)}
                write_json(cycle / 'START.json', {'solver': identity(model), 'harness': str(harness), 'sha256': sha(harness)})
                status(run, 'reward_baselines', cycle=number)
                pool.evaluate_many([spec(model, harness, c['data'], c['baseline'], seed=c['seed'], temperature=1.0)
                    for c in cases.values()])
                pre_dir, pre = proposals(run, cycle, config, url, 'pre', harness, cases)
                scored = score(pool, run, cycle, config, model, harness, cases, 'pre', pre_dir, pre)
                post_dir, post = proposals(run, cycle, config, url, 'post', harness, cases, scored)
                scored_post = score(pool, run, cycle, config, model, harness, cases, 'post', post_dir, post)
                status(run, 'harness_selection', cycle=number)
                harness = select(pool, cycle, config, model, harness, post_dir, scored_post)
                chosen.append(str(harness))
                write_json(cycle / 'complete.json', {'model': str(model), 'harness': str(harness), 'sha256': sha(harness)})
                status(run, 'harness_selected', cycle=number, decision=read(cycle / 'selection/decision.json')['accepted'])
            write_json(run / 'SEARCH_COMPLETE.json', {'utc': utc(), 'harnesses': chosen,
                'elapsed_seconds': time.monotonic() - started, 'test_feedback_used': False})
            states = [('initial', 'W2_initial', config['initial_harness']),
                ('cycle_01_after_harness', 'W2_initial', chosen[0]),
                ('cycle_01_after_weights', 'W3_cycle1', chosen[0]),
                ('cycle_02_after_harness', 'W3_cycle1', chosen[1]),
                ('cycle_02_after_weights', 'W4_cycle2', chosen[1])]
            registry = {}; curve = []
            for label, model_name, h in states:
                key = (config['models'][model_name]['weight_digest'], sha(h))
                if key in registry:
                    entries = registry[key]
                elif sha(h) == config['initial_harness_sha256']:
                    entries = {c: {'path': str(EXPANDED / 'eval' / model_name / c), 'reused_draw': 0,
                        'trajectories_sha256': sha(EXPANDED / 'eval' / model_name / c / 'trajectories.jsonl')}
                        for c in config['cohorts']}
                else:
                    status(run, 'single_seed_evaluation', label=label, model=model_name)
                    pool.evaluate_many([spec(config['models'][model_name]['model'], h, config['data']['splits'][c],
                        run / 'eval' / label / c, seed=42, temperature=1.0) for c in config['cohorts']])
                    entries = {c: {'path': str(run / 'eval' / label / c), 'reused_draw': None,
                        'trajectories_sha256': sha(run / 'eval' / label / c / 'trajectories.jsonl')} for c in config['cohorts']}
                registry[key] = entries
                curve.append({'label': label, 'model': model_name, 'weight_digest': key[0],
                    'harness': h, 'harness_sha256': key[1], 'cohorts': entries})
                write_json(run / 'curve_sources.json', curve)
        write_json(run / 'COMPLETE.json', {'utc': utc(), 'elapsed_seconds': time.monotonic() - started,
            'training_updates': 0, 'harnesses': chosen})
        status(run, 'complete_pending_audit')
    except BaseException as exc:
        status(run, 'failed', error=repr(exc)); raise


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=['prepare', 'run']); p.add_argument('--run', type=Path, required=True)
    p.add_argument('--gpus', nargs='+', default=['6', '7', '8', '9'] + [f'node1:{i}' for i in range(10)])
    args = p.parse_args()
    if args.action == 'prepare': prepare(args.run.resolve(), args.gpus)
    else: execute(args.run.resolve())

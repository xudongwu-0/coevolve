"""Rapid H -> W -> H comparison: external, frozen editor, learned editor.

All arms start with the same W/H. The internal arms start with the same E.
Only development outcomes enter search/training. Test all frozen stages last.
"""
import argparse
from contextlib import nullcontext
import hashlib
import json
from pathlib import Path
import shutil
import time

from alternating_common import check_proposals, identity, run_editor, source_snapshot
from alternating_cycles import explored_proposals
from alternating_pool import Pool
from external_harness_replay import proposer
from harness_code_contract import validate_source
from math_self_replacement import read, rows, dump_rows, model_spec, summarize, status, utc
from paired_joint_whale import train_solver
from rapid_internal_alternating import open_requests
from meta_harness.benchmark import compute_pareto_frontier
from meta_harness.meta_harness_retool import pick_accepted_from_frontier
from pipeline import ROOT, api, sha, write_json

ARMS = ['external', 'fixed_editor', 'trained_editor']


def prepare(run):
    run.mkdir(parents=True, exist_ok=False)
    prior = ROOT / 'runs/whale_math_self_replacement_20260917'
    old = read(prior / 'config.json')
    shutil.copy2(old['initial_harness'], run / 'initial_harness.py')
    w = old['solver_initial_checkpoint']; e = old['editor_initial_checkpoint']
    cfg = dict(arms=ARMS, solver_initial_checkpoint=w, editor_initial_checkpoint=e,
        initial_solver=identity(w), initial_editor=identity(e), initial_harness=str(run / 'initial_harness.py'),
        data=old['data'], pool_gpus=['7', '8', '9'] + [f'node1:{i}' for i in range(1, 10)],
        checkpoint_allowlist=[w] + [str(run / a / 'weight_phase/solver_update/round_01/checkpoint') for a in ARMS],
        editor_gpu='6', solver_gpu='6', solver_rsft_lr=1e-7,
        train_rollouts=32, train_schedule=[[0, i] for i in range(32)],
        editor_grpo_updates=4, proposal_temperature=1.0, pre_candidates=8, post_candidates=4,
        harness_phases=2, reward_tasks=48, evaluation_tasks=128, seed=42, proposer_port=18542,
        logical_harness_rollout_cap_per_phase=13*48,
        selection='Original WHALE Pareto accuracy/turn rule, incumbent + unique valid post candidates',
        shared_prefix=old['shared_rsft_prefix'],
        editor_learning='Harness-R1-style group-relative PPO on full48 current-solver success delta; invalid/noop reward delta zero',
        order='H1 -> W1 (32 on-policy training trajectories) -> H2; all test128 stages evaluated only after freezing',
        scope='Controlled small local WHALE-style full-code search; same restricted executable code contract in all arms',
        limits=['One solver sampling seed, one draw per test question; historically used test128.',
            'Equal proposal slots and target-rollout ceiling, not necessarily equal actual rollouts after duplicate caching.',
            'External proposer is the active conversation via localhost FastAPI; its reasoning/tokens/cost are unmetered.',
            'Internal E initialization includes historical SFT/outcome training; fixed means no NEW E updates in this comparison.',
            'External proposals must not read internal candidates or this run test outcomes.',
            'Same source/weights/data/schedule may reuse exact trajectories or a training checkpoint; cache reuse is not algorithmic efficiency.',
            'No retry for invalid proposals or zero reward variance. No monotonicity claim.'])
    write_json(run / 'config.json', cfg)
    write_json(run / 'PROTOCOL.json', {'utc': utc(), 'config_sha256': sha(run / 'config.json'),
        'new_training_rollouts_per_arm': 32, 'test_feedback_used_for_selection': False,
        'phase_points': ['start', 'H1', 'W1', 'H2'], 'pre_and_post_internal_seeds_matched': True})
    source_snapshot(run, 'initial', ['three_arm_hwh.py', 'rapid_internal_alternating.py',
        'paired_joint_whale.py', 'alternating_solver_train.py', 'external_harness_replay.py', 'proposer_api.py'])
    status(run, 'prepared')


def evaluate_cached(pool, run, model, harness, data, output, schedule=None):
    schedule = schedule or [[0, i] for i in range(data['count'])]
    key = hashlib.sha256(json.dumps([model['weight_digest'], sha(harness), data['sha256'], schedule, 42, 1.0]).encode()).hexdigest()
    cache = run / 'rollout_cache' / key
    existed = (cache / 'val.json').exists()
    pool.evaluate_many([model_spec(model, harness, data, cache, schedule)])
    summarize(output, data, model, harness, rows(cache / 'trajectories.jsonl'))
    write_json(output / 'CACHE.json', {'source': str(cache), 'already_existed': existed,
        'logical_rollouts': len(schedule), 'new_physical_rollouts': 0 if existed else len(schedule)})
    return output


def score_proposals(pool, run, cfg, model, harness, directory, baseline):
    generated = rows(directory / 'generations.jsonl')
    checks = check_proposals(directory, cfg['data']['splits']['mh_val'])
    base_score = read(baseline / 'val.json')['success_rate']; scored = []
    data = cfg['data']['splits']['mh_val']; schedule = [[0, i] for i in range(data['count'])]
    jobs, mapped, existed = {}, {}, {}
    for g in generated:
        if not g['valid_source'] or not checks[g['id']]['passed']:
            mapped[g['id']] = ('invalid', None)
        elif g['source'].strip() == harness.read_text().strip():
            mapped[g['id']] = ('no_op', None)
        else:
            source = directory / (g['id'] + '.py')
            key = hashlib.sha256(json.dumps([model['weight_digest'], sha(source), data['sha256'], schedule, 42, 1.0]).encode()).hexdigest()
            cache = run / 'rollout_cache' / key
            existed[key] = (cache / 'val.json').exists()
            jobs[key] = model_spec(model, source, data, cache, schedule)
            mapped[g['id']] = ('valid', key)
    # Different candidates use the same solver; distribute their independent
    # trajectories together so a slow final shard does not idle other GPUs.
    pool.evaluate_many(list(jobs.values()))
    recorded = set()
    for g in generated:
        kind, key = mapped[g['id']]; out = None; delta = 0.
        if kind == 'valid':
            out = directory / 'scores' / g['id']; cache = run / 'rollout_cache' / key
            summarize(out, data, model, directory / (g['id'] + '.py'), rows(cache / 'trajectories.jsonl'))
            write_json(out / 'CACHE.json', {'source': str(cache), 'already_existed': existed[key] or key in recorded,
                'logical_rollouts': len(schedule), 'new_physical_rollouts': 0 if existed[key] or key in recorded else len(schedule)})
            recorded.add(key)
            delta = read(out / 'val.json')['success_rate'] - base_score
        scored.append({**g, 'outcome_kind': kind, 'reward_delta': delta,
            'reward_provenance': {'baseline': str(baseline), 'candidate': str(out) if out else None,
                'baseline_score': base_score, 'candidate_score': base_score + delta if out else None,
                'solver_weight_digest': model['weight_digest'], 'task_count': 48}})
    dump_rows(directory / 'scored.jsonl', scored)
    write_json(directory / 'score_summary.json', {'rewards': [g['reward_delta'] for g in scored],
        'valid': sum(g['outcome_kind'] == 'valid' for g in scored),
        'unique_valid': len({g['source'] for g in scored if g['outcome_kind'] == 'valid'})})
    return scored


def generate_internal(run, checkpoint, directory, requests, cfg):
    key = hashlib.sha256(json.dumps([identity(checkpoint)['weight_digest'], requests, cfg['proposal_temperature']], sort_keys=True).encode()).hexdigest()
    cache = run / 'generation_cache' / key
    existed = (cache / 'complete.json').exists()
    explored_proposals(checkpoint, cache, requests, cfg['proposal_temperature'], gpu=cfg['editor_gpu'])
    directory.mkdir(parents=True, exist_ok=True)
    for p in cache.glob('*'):
        if p.is_file(): shutil.copy2(p, directory / p.name)
    write_json(directory / 'CACHE.json', {'source': str(cache), 'already_existed': existed,
        'logical_proposals': len(requests), 'new_physical_proposals': 0 if existed else len(requests)})


def external_generate(run, cycle, url, stage, requests, harness, previous=None):
    directory = cycle / stage; directory.mkdir(parents=True, exist_ok=True)
    if (directory / 'generations.jsonl').exists(): return
    packet = {'task_prompt': 'Generate exactly the requested CandidateEnv Python modules for a controlled WHALE-style search. '
        'Use the supplied executable contract. You may edit SYSTEM_PROMPT, USER_PROMPT_TEMPLATE, MAX_TURNS (1..3), '
        'no_tools_called and get_prompt_messages. No new tools, answer access, grading/budget changes or per-task hardcoding. '
        'Return candidates with id, group, source, hypothesis. Only current DEVELOPMENT feedback may guide proposals.',
        'run_dir': str(directory), 'iteration': int(cycle.name[-2:]),
        'accepted_harness_source': harness.read_text(),
        'gate_context': {'stage': stage, 'slots': [{'id': r['id'], 'group': r['group']} for r in requests],
            'messages': requests[0]['messages'],
            'previous': [{k: g.get(k) for k in ['id', 'source', 'outcome_kind', 'reward_delta']} for g in previous or []]},
        'frontier': {}}
    receipt_file = directory / 'request_receipt.json'
    receipt = read(receipt_file) if receipt_file.exists() else api(url, '/jobs', packet)
    write_json(receipt_file, receipt)
    status(run, 'awaiting_external_proposals', phase=cycle.name, proposal_stage=stage, job_id=receipt['job_id'])
    deadline = time.monotonic() + 3600
    while True:
        response = api(url, '/jobs/' + receipt['job_id'])
        if 'candidates' in response: break
        if time.monotonic() > deadline: raise TimeoutError('Active external proposer response')
        time.sleep(1)
    expected = {(r['id'], r['group']) for r in requests}
    assert len(response['candidates']) == len(expected)
    assert {(g['id'], g['group']) for g in response['candidates']} == expected
    gs = []
    for g in response['candidates']:
        try: source, valid, error = validate_source(g['source']), True, None
        except Exception as exc: source, valid, error = g.get('source', ''), False, repr(exc)
        gs.append({**g, 'source': source, 'valid_source': valid, 'validation_error': error})
        (directory / (g['id'] + '.py')).write_text(source)
    dump_rows(directory / 'generations.jsonl', gs)
    write_json(directory / 'complete.json', {'proposer': response.get('proposer'), 'receipt': receipt,
        'response_sha256': sha(run / 'proposer_jobs' / receipt['job_id'] / 'response.json')})


def select(cycle, incumbent, baseline, post):
    choices = {'incumbent': (incumbent, baseline)}
    seen = {sha(incumbent)}
    for g in post:
        if g['outcome_kind'] != 'valid': continue
        p = cycle / 'post' / (g['id'] + '.py')
        if sha(p) in seen: continue
        seen.add(sha(p)); choices[g['id']] = (p, Path(g['reward_provenance']['candidate']))
    dest = cycle / 'selection'; values = {}
    for name, (path, out) in choices.items():
        target = dest / 'harnesses' / name / 'harness.py'; target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target); values[name] = read(out / 'val.json')
    frontier = {'_pareto': [{'harness': n, 'avg_success_rate': s, 'avg_mean_turn_count': t}
        for n, s, t in compute_pareto_frontier([(n, v['success_rate'], v['mean_turn_count']) for n, v in values.items()])]}
    _, accepted = pick_accepted_from_frontier(frontier, dest)
    assert accepted
    selected = dest / 'harnesses' / accepted / 'harness.py'
    write_json(dest / 'decision.json', {'accepted': accepted, 'results': values, 'frontier': frontier,
        'changed': sha(selected) != sha(incumbent), 'harness': str(selected), 'test_used': False})
    return selected


def execute(run):
    cfg = read(run / 'config.json')
    arms = cfg.get('arms', ARMS)
    if not arms or len(set(arms)) != len(arms) or set(arms) - set(ARMS):
        raise ValueError('arms must be a nonempty subset of external, fixed_editor, trained_editor')
    assert sha(run / 'config.json') == read(run / 'PROTOCOL.json')['config_sha256']
    if (run / 'HARDWARE_OVERRIDE.json').exists():
        cfg = {**cfg, 'pool_gpus': read(run / 'HARDWARE_OVERRIDE.json')['pool_gpus']}
    if not (run / 'START.json').exists(): write_json(run / 'START.json', {'utc': utc()})
    models = {a: Path(cfg['solver_initial_checkpoint']) for a in arms}
    harnesses = {a: Path(cfg['initial_harness']) for a in arms}
    editors = {a: Path(cfg['editor_initial_checkpoint']) for a in arms if a != 'external'}
    points = {a: [{'label': 'start', 'model': str(models[a]), 'harness': str(harnesses[a])}] for a in arms}
    pooldir = run / 'pool'
    if pooldir.exists():
        assert (pooldir / 'pool_stopped.json').exists()
        pooldir = run / f'pool_resume_{time.time_ns()}'
    try:
        proposer_context = proposer(run, cfg['proposer_port']) if 'external' in arms else nullcontext(None)
        with proposer_context as url, Pool(pooldir, cfg) as pool:
            trained_cache = {}
            for phase in (1, 2):
                for arm in arms:
                    cycle = run / arm / f'phase_{phase:02d}'; cycle.mkdir(parents=True, exist_ok=True)
                    if (cycle / 'complete.json').exists():
                        done = read(cycle / 'complete.json'); harnesses[arm] = Path(done['harness'])
                        if arm != 'external': editors[arm] = Path(done['editor'])
                    else:
                        m = identity(models[arm]); h = harnesses[arm]
                        write_json(cycle / 'START.json', {'model': m, 'harness': str(h),
                            'editor': identity(editors[arm]) if arm != 'external' else None})
                        status(run, 'development_baseline', arm=arm, phase=phase)
                        baseline = evaluate_cached(pool, run, m, h, cfg['data']['splits']['mh_val'], cycle / 'incumbent')
                        requests = open_requests(h, baseline, phase, cfg['proposal_temperature'])
                        # The historical prompt assumed a recovery hook already existed.
                        # This comparison starts from H0, so describe the actual source.
                        for request in requests:
                            request['messages'][-1]['content'] = request['messages'][-1]['content'].replace(
                                'The existing harness already recovers incomplete answers; preserve useful existing behavior.',
                                'Inspect the supplied source to determine which recovery hooks exist; preserve useful existing behavior.')
                        write_json(cycle / 'REQUESTS.json', requests)
                        status(run, 'pre_generation', arm=arm, phase=phase)
                        if arm == 'external': external_generate(run, cycle, url, 'pre', requests, h)
                        else: generate_internal(run, editors[arm], cycle / 'pre', requests, cfg)
                        status(run, 'score_pre', arm=arm, phase=phase)
                        pre = score_proposals(pool, run, cfg, m, h, cycle / 'pre', baseline)
                        before = identity(editors[arm]) if arm != 'external' else None
                        trained = {'updated': False, 'reason': 'external proposer' if arm == 'external' else 'frozen control'}
                        if arm == 'trained_editor':
                            status(run, 'editor_training', arm=arm, phase=phase)
                            trained = run_editor(run, editors[arm], cycle / 'editor_update', pre, gpu=cfg['editor_gpu'], mode='train', epochs=cfg['editor_grpo_updates'])
                            if trained['updated']: editors[arm] = Path(trained['identity']['weights'])
                        status(run, 'post_generation', arm=arm, phase=phase)
                        if arm == 'external': external_generate(run, cycle, url, 'post', requests[:4], h, pre)
                        else: generate_internal(run, editors[arm], cycle / 'post', requests[:4], cfg)
                        status(run, 'score_post', arm=arm, phase=phase)
                        post = score_proposals(pool, run, cfg, m, h, cycle / 'post', baseline)
                        harnesses[arm] = select(cycle, h, baseline, post)
                        write_json(cycle / 'EDITOR_RESULT.json', {'before': before,
                            'after': identity(editors[arm]) if arm != 'external' else None, 'training': trained,
                            'pre_rewards': [g['reward_delta'] for g in pre], 'post_rewards': [g['reward_delta'] for g in post],
                            'matched4_changed_sources': sum(a.get('source') != b.get('source') for a, b in zip(pre[:4], post)) if arm != 'external' else None})
                        write_json(cycle / 'complete.json', {'harness': str(harnesses[arm]),
                            'editor': str(editors[arm]) if arm != 'external' else None})
                    points[arm].append({'label': f'H{phase}', 'model': str(models[arm]), 'harness': str(harnesses[arm])})
                    write_json(run / 'stage_identities.json', points)
                if phase == 1:
                    for arm in arms:
                        cycle = run / arm / 'weight_phase'; cycle.mkdir(parents=True, exist_ok=True)
                        if (cycle / 'complete.json').exists(): models[arm] = Path(read(cycle / 'complete.json')['model'])
                        else:
                            status(run, 'solver_rollouts', arm=arm)
                            evaluate_cached(pool, run, identity(models[arm]), harnesses[arm], cfg['data']['splits']['train'],
                                cycle / 'solver_rollouts', cfg['train_schedule'])
                            key = (identity(models[arm])['weight_digest'], sha(harnesses[arm]), sha(cycle / 'solver_rollouts/trajectories.jsonl'))
                            if key in trained_cache:
                                models[arm] = trained_cache[key]
                                write_json(cycle / 'TRAINING_REUSED.json', {'checkpoint': str(models[arm]), 'reason': 'Identical starting W, H and exact rollout tokens'})
                            else:
                                status(run, 'solver_training', arm=arm)
                                models[arm] = train_solver(cycle, cfg, models[arm], harnesses[arm], cfg['solver_gpu'])
                                trained_cache[key] = models[arm]
                            write_json(cycle / 'complete.json', {'model': str(models[arm]), 'identity': identity(models[arm]), 'harness': str(harnesses[arm])})
                        points[arm].append({'label': 'W1', 'model': str(models[arm]), 'harness': str(harnesses[arm])})
                        write_json(run / 'stage_identities.json', points)
            write_json(run / 'FROZEN.json', {'utc': utc(), 'points': points, 'editors': {a: str(e) for a, e in editors.items()},
                'test_started': False})
            for arm in arms:
                for point in points[arm]:
                    status(run, 'frozen_test128', arm=arm, stage_label=point['label'])
                    out = run / 'test' / arm / point['label']
                    evaluate_cached(pool, run, identity(point['model']), Path(point['harness']), cfg['data']['splits']['test'], out)
                    point.update(output=str(out), solved=read(out / 'val.json')['solved_examples'], total=128)
                    write_json(run / 'CURVES.json', points)
            write_json(run / 'COMPLETE.json', {'utc': utc(), 'curves': points,
                'editors': {a: str(e) for a, e in editors.items()}})
        status(run, 'complete')
    except BaseException as exc:
        status(run, 'failed', error=repr(exc)); raise


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('mode', choices=['prepare', 'execute']); p.add_argument('--run', type=Path, required=True)
    args = p.parse_args(); (prepare if args.mode == 'prepare' else execute)(args.run.resolve())

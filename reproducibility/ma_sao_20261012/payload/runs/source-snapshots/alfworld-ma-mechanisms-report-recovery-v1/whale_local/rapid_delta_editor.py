"""Two current-solver editor phases separated by one small solver update."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess

from alternating_common import identity, check_proposals, source_snapshot
from alternating_pool import Pool
from harness_bootstrap import FAMILIES
from harness_delta import messages
from math_self_replacement import read, rows, dump_rows, utc, status, model_spec, summarize
from paired_joint_whale import train_solver
from pipeline import ROOT, sha, write_json
from meta_harness.benchmark import compute_pareto_frontier
from meta_harness.meta_harness_retool import pick_accepted_from_frontier


def editor_call(checkpoint, directory, requests, mode='generate', epochs=8, lr=3e-6):
    if (directory / 'complete.json').exists(): return read(directory / 'complete.json')
    directory.mkdir(parents=True, exist_ok=True); data = directory / 'requests.jsonl'
    dump_rows(data, requests)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='6', WHALE_RUN_ROOT=str(directory), WHALE_MODEL_ROOT=str(checkpoint),
        WHALE_PARAMETER_DTYPE='float32', HF_HUB_OFFLINE='1', TOKENIZERS_PARALLELISM='false',
        PYTHONPATH=':'.join(str(ROOT / p) for p in ['whale_local', 'WHALE/domains/math_reasoning', 'WHALE/domains/chess_puzzles']))
    with directory.with_suffix('.log').open('a') as log:
        subprocess.run([str(ROOT / '.venv-whale-portable/bin/python'), '-u', str(ROOT / 'whale_local/harness_delta_worker.py'),
            mode, '--data', str(data), '--epochs', str(epochs), '--learning-rate', str(lr)],
            env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
    return read(directory / 'complete.json')


def packet(rr, model, previous=None):
    failures = [r for r in rr if not r['reward']]
    result = {'solver_weight_digest': model['weight_digest'], 'tasks': len(rr), 'solved': int(sum(r['reward'] for r in rr)),
        'failed_without_final': sum('\\boxed{' not in r['last_assistant'] for r in failures),
        'failed_with_boxed_answer': sum('\\boxed{' in r['last_assistant'] for r in failures), 'failure_observations': []}
    old = {r['task_id']: r for r in previous or []}
    if old:
        result['since_solver_update'] = {'newly_solved': sum(r['reward'] > old[r['task_id']]['reward'] for r in rr),
            'newly_failed': sum(r['reward'] < old[r['task_id']]['reward'] for r in rr),
            'note': 'Old rewards are not training labels for this checkpoint. All candidates are rerun on the current solver.'}
    for r in failures[:4]:
        question = next((m['content'] for m in r['policy_calls'][0]['messages'] if m['role'] == 'user'), '')
        tool_text = [m.get('content', '') for c in r['policy_calls'] for m in c['messages'] if m['role'] == 'tool']
        result['failure_observations'].append({'question': question[:360], 'assistant_tail': r['last_assistant'][-600:],
            'tool_outputs': [str(t)[-180:] for t in tool_text[-2:]], 'assistant_tokens': r['assistant_tokens']})
    return result


def choose_batch(rr):
    categories = [[], [], []]
    for r in rr:
        if r['reward']: categories[2].append(r)
        elif '\\boxed{' in r['last_assistant']: categories[1].append(r)
        else: categories[0].append(r)
    failed = []
    while (categories[0] or categories[1]) and len(failed) < 6:
        for category in categories[:2]:
            if category and len(failed) < 6: failed.append(category.pop(0))
    chosen = failed + categories[2][:6]; seen = {r['task_id'] for r in chosen}
    chosen += [r for r in rr if r['task_id'] not in seen][:12 - len(chosen)]
    return sorted(chosen, key=lambda r: r['example_index'])


def curriculum(source):
    result = []
    for family, (feedback, targets) in FAMILIES.items():
        key = {'finalize': 'partial_response_hint', 'repair': 'tool_error_hint'}.get(family, 'SYSTEM_PROMPT')
        for i in range(6):
            result.append({'id': f'{family}_{i}', 'messages': messages(source,
                {'synthetic_interface_example': True, 'failure_observations': [feedback[i]], 'instruction': 'Replace the relevant guidance.'}),
                'completion': json.dumps({key: targets[i % len(targets)]})})
    return result


def score(pool, run, cfg, model, harness, directory, baseline):
    gs = rows(directory / 'generations.jsonl'); data = cfg['data']['splits']['mh_val']
    checks = check_proposals(directory, data); base = rows(baseline / 'trajectories.jsonl')
    schedule = [[0, r['example_index']] for r in base]; jobs = {}; mapped = {}
    for g in gs:
        if not g['valid_source'] or not checks[g['id']]['passed']: mapped[g['id']] = ('invalid', None); continue
        if not g['edit_changed']: mapped[g['id']] = ('no_op', None); continue
        path = directory / (g['id'] + '.py')
        key = hashlib.sha256(json.dumps([model['weight_digest'], sha(path), sha(harness), data['sha256'], schedule, 42, 1.0]).encode()).hexdigest()
        output = run / 'outcome_cache' / key
        jobs[key] = model_spec(model, path, data, output, schedule); mapped[g['id']] = ('valid', output)
    pool.evaluate_many(list(jobs.values())); scored = []
    for g in gs:
        kind, output = mapped[g['id']]
        result = rows(output / 'trajectories.jsonl') if output else base
        assert [r['task_id'] for r in result] == [r['task_id'] for r in base]
        delta = sum(r['reward'] - b['reward'] for r, b in zip(result, base)) / len(base)
        scored.append({**g, 'outcome_kind': kind, 'reward_delta': delta,
            'reward_provenance': {'candidate': str(output) if output else None, 'baseline': str(baseline),
                'solver_weight_digest': model['weight_digest'], 'solved': int(sum(r['reward'] for r in result)),
                'baseline_solved': int(sum(r['reward'] for r in base)), 'task_count': len(base),
                'rescued': sum(r['reward'] > b['reward'] for r, b in zip(result, base)),
                'regressed': sum(r['reward'] < b['reward'] for r, b in zip(result, base))}})
    dump_rows(directory / 'scored.jsonl', scored)
    return scored


def summary(scored):
    valid = [g for g in scored if g['outcome_kind'] == 'valid']
    return {'slots': len(scored), 'valid_changed': len(valid), 'unique_changed': len({g['source'] for g in valid}),
        'positive_gain': sum(g['reward_delta'] > 0 for g in scored), 'rewards': [g['reward_delta'] for g in scored],
        'mean_reward': sum(g['reward_delta'] for g in scored) / len(scored),
        'mean_tokens': sum(len(g['response_ids']) for g in scored) / len(scored)}


def select(pool, cfg, cycle, model, harness, scored, full_result_cache=()):
    dest = cycle / 'selection'; dest.mkdir(exist_ok=True)
    ranked = sorted([r for r in scored if r['outcome_kind'] == 'valid'], key=lambda r: (-r['reward_delta'], r['id']))
    choices = {'incumbent': (harness, cycle / 'incumbent')}
    if ranked:
        g = ranked[0]; path = cycle / 'post' / (g['id'] + '.py'); out = dest / g['id']
        part = rows(Path(g['reward_provenance']['candidate']) / 'trajectories.jsonl'); used = {r['example_index'] for r in part}
        remainder = out / 'remaining'; data = cfg['data']['splits']['mh_val']
        cached = Path(g['reward_provenance']['candidate']) if len(used) == data['count'] else None
        for previous in full_result_cache:
            previous = Path(previous)
            if not (previous / 'val.json').exists(): continue
            value = read(previous / 'val.json'); meta = value['metadata']
            if (meta['weight_digest'] == model['weight_digest'] and meta['harness_sha256'] == sha(path)
                and meta['dataset_sha256'] == data['sha256'] and meta['seed'] == 42 and meta['temperature'] == 1.0
                and meta['sampling_schedule'] == [[0, i] for i in range(data['count'])]):
                cached = previous; break
        if cached is not None:
            summarize(out, data, model, path, rows(cached / 'trajectories.jsonl'))
            write_json(out / 'REUSED.json', {'source': str(cached), 'new_rollouts': 0})
        else:
            pool.evaluate_many([model_spec(model, path, data, remainder, [[0, i] for i in range(data['count']) if i not in used])])
            summarize(out, data, model, path, part + rows(remainder / 'trajectories.jsonl'))
        choices[g['id']] = (path, out)
    values = {name: read(out / 'val.json') for name, (_, out) in choices.items()}
    for name, (path, _) in choices.items():
        d = dest / 'harnesses' / name; d.mkdir(parents=True, exist_ok=True); shutil.copy2(path, d / 'harness.py')
    frontier = {'_pareto': [{'harness': n, 'avg_success_rate': s, 'avg_mean_turn_count': t}
        for n, s, t in compute_pareto_frontier([(n, v['success_rate'], v['mean_turn_count']) for n, v in values.items()])]}
    _, accepted = pick_accepted_from_frontier(frontier, dest); assert accepted
    selected = dest / 'harnesses' / accepted / 'harness.py'
    write_json(dest / 'decision.json', {'accepted': accepted, 'results': values, 'frontier': frontier,
        'changed': sha(selected) != sha(harness), 'harness': str(selected), 'test_used': False})
    return selected


def prepare(run):
    run.mkdir(parents=True, exist_ok=False)
    prior = ROOT / 'runs/whale_internal_alternating_rapid_20260917'; old = read(prior / 'config.json'); models = read(prior / 'CURRENT_MODELS.json')
    shutil.copy2(models['harness'], run / 'initial_harness.py')
    cfg = {'reference': str(prior), 'solver_initial_checkpoint': models['solver'], 'editor_initial_checkpoint': models['editor'],
        'initial_harness': str(run / 'initial_harness.py'), 'data': old['data'], 'pool_gpus': old['pool_gpus'],
        'checkpoint_allowlist': [models['solver'], str(run / 'cycle_01/solver_update/round_01/checkpoint')],
        'solver_rsft_lr': 1e-7, 'new_solver_rollouts': 32, 'editor_phases': 2, 'reward_batch': 12,
        'editor_update_steps': 8, 'editor_lr': 3e-6, 'proposal_slots': 8, 'post_slots': 4, 'seed': 42,
        'warmup_if_needed': '48 existing synthetic format/mechanism demonstrations, two epochs, lr1e-5; charged separately',
        'objective': 'clipped PPO on gain over current incumbent / 0.125, clipped +/-2; no group mean; invalid/no-op advantage0; KL .02',
        'limits': ['Delta actions change SYSTEM_PROMPT and two existing recovery instructions, not arbitrary Python hooks.',
            'Pilot combines representation and learning changes; not a factorial ablation or a proof against Harness-R1.',
            'Reward and failure packets refreshed after solver update; no stale reward reuse.', 'Historical test128 reused, one seed and one draw.']}
    write_json(run / 'config.json', cfg); write_json(run / 'PROTOCOL.json', {'utc': utc(), 'config_sha256': sha(run / 'config.json')})
    source_snapshot(run, 'initial', ['rapid_delta_editor.py', 'harness_delta.py', 'harness_delta_worker.py'])
    status(run, 'prepared')


def execute(run):
    cfg = read(run / 'config.json'); assert sha(run / 'config.json') == read(run / 'PROTOCOL.json')['config_sha256']
    if not (run / 'START.json').exists(): write_json(run / 'START.json', {'utc': utc()})
    else: write_json(run / 'RESUME.json', {'utc': utc(), 'reason': 'Correct tokenizer return type before the first SFT update; reuse completed development and interface probe.'})
    model = Path(cfg['solver_initial_checkpoint']); editor = Path(cfg['editor_initial_checkpoint']); harness = Path(cfg['initial_harness'])
    points = [{'label': 'initial', 'model': str(model), 'editor': str(editor), 'harness': str(harness)}]; previous = None
    pool_dir = run / 'pool'
    if pool_dir.exists():
        assert (pool_dir / 'pool_stopped.json').exists()
        index = 1
        while (run / f'pool_resume_{index:02d}').exists(): index += 1
        pool_dir = run / f'pool_resume_{index:02d}'
    try:
        with Pool(pool_dir, cfg) as pool:
            for n in [1, 2]:
                cycle = run / f'cycle_{n:02d}'; cycle.mkdir(exist_ok=True)
                m = identity(model); write_json(cycle / 'START.json', {'solver': m, 'editor': identity(editor), 'harness': str(harness)})
                status(run, 'refresh_current_solver_failures', cycle=n)
                data = cfg['data']['splits']['mh_val']; base = cycle / 'incumbent'
                pool.evaluate_many([model_spec(m, harness, data, base)])
                baseline = rows(base / 'trajectories.jsonl'); chosen = choose_batch(baseline); reward_base = cycle / 'reward_baseline'
                summarize(reward_base, data, m, harness, chosen)
                feedback = packet(chosen, m, previous); write_json(cycle / 'FAILURE_PACKET.json', feedback)
                req = [{'id': f'edit_{i:02d}', 'group': f'current_solver_{n}', 'harness': str(harness),
                    'messages': messages(harness.read_text(), feedback), 'seed': 121400 + n * 1000 + i} for i in range(8)]
                if n == 1:
                    status(run, 'delta_interface_probe')
                    editor_call(editor, run / 'interface_probe', req[:4])
                    probe = rows(run / 'interface_probe/generations.jsonl')
                    unique = {g['source'] for g in probe if g['valid_source'] and g['edit_changed']}
                    if len(unique) < 2:
                        status(run, 'delta_format_warmup', probe_distinct=len(unique))
                        warm = curriculum(harness.read_text())
                        for row in warm: row['provenance'] = 'Existing Codex-authored synthetic interface examples; no measured task gains or benchmark answers'
                        result = editor_call(editor, run / 'format_warmup', warm, 'sft', epochs=2, lr=1e-5)
                        editor = Path(result['identity']['weights'])
                status(run, 'delta_pre_generation', cycle=n)
                editor_call(editor, cycle / 'pre', req)
                before = score(pool, run, cfg, m, harness, cycle / 'pre', reward_base)
                status(run, 'paired_gain_editor_update', cycle=n, pre=summary(before))
                initial_e = identity(editor); trained = editor_call(editor, cycle / 'editor_update', before, 'train')
                if trained['updated']: editor = Path(trained['identity']['weights'])
                status(run, 'delta_post_generation', cycle=n, updated=trained['updated'])
                editor_call(editor, cycle / 'post', req[:4])
                after = score(pool, run, cfg, m, harness, cycle / 'post', reward_base)
                write_json(cycle / 'EDITOR_RESULT.json', {'before': initial_e, 'after': identity(editor), 'updated': trained['updated'],
                    'pre': summary(before), 'pre_matched4': summary(before[:4]), 'post': summary(after),
                    'target_solver_weight_digest': m['weight_digest'], 'test_used': False})
                status(run, 'select_delta_harness', cycle=n, post=summary(after))
                harness = select(pool, cfg, cycle, m, harness, after)
                points.append({'label': f'cycle_{n}_H', 'model': str(model), 'editor': str(editor), 'harness': str(harness)})
                if n == 1:
                    # Refresh incumbent under the accepted H after W changes in the next phase.
                    selected = read(cycle / 'selection/decision.json')['accepted']
                    previous = rows((base if selected == 'incumbent' else cycle / 'selection' / selected) / 'trajectories.jsonl')
                    status(run, 'small_solver_rollout_update', trajectories=32)
                    pool.evaluate_many([model_spec(m, harness, cfg['data']['splits']['train'], cycle / 'solver_rollouts', [[0, i] for i in range(32)])])
                    model = train_solver(cycle, cfg, model, harness, '6')
                    points.append({'label': 'cycle_1_W', 'model': str(model), 'editor': str(editor), 'harness': str(harness)})
                write_json(run / 'STAGE_STATES.json', points)
                write_json(run / 'CURRENT_MODELS.json', {'solver': str(model), 'editor': str(editor), 'harness': str(harness), 'cycle': n})
            write_json(run / 'FROZEN.json', {'utc': utc(), 'points': points, 'test_used_for_selection': False})
            prior = Path(cfg['reference']); initial = rows(prior / 'final128/trajectories.jsonl')
            summarize(run / 'monitor/initial', cfg['data']['splits']['test'], identity(cfg['solver_initial_checkpoint']),
                Path(cfg['initial_harness']), initial[:64])
            cache = {(identity(points[0]['model'])['weight_digest'], sha(points[0]['harness'])): run / 'monitor/initial'}; curve = []
            for point in points:
                key = (identity(point['model'])['weight_digest'], sha(point['harness']))
                if key not in cache:
                    output = run / 'monitor' / point['label']; status(run, 'stage_monitor', label=point['label'])
                    pool.evaluate_many([model_spec(identity(point['model']), Path(point['harness']), cfg['data']['splits']['test'], output, [[0, i] for i in range(64)])]); cache[key] = output
                value = read(cache[key] / 'val.json')
                curve.append({**point, 'output': str(cache[key]), 'solved': value['solved_examples'], 'total': 64})
                write_json(run / 'CURVE.json', curve)
            status(run, 'final128_completion')
            rest = run / 'final_remaining'
            pool.evaluate_many([model_spec(identity(model), harness, cfg['data']['splits']['test'], rest, [[0, i] for i in range(64, 128)])])
            summarize(run / 'final128', cfg['data']['splits']['test'], identity(model), harness,
                rows(Path(curve[-1]['output']) / 'trajectories.jsonl') + rows(rest / 'trajectories.jsonl'))
        write_json(run / 'COMPLETE.json', {'utc': utc(), 'solver': identity(model), 'editor': identity(editor)})
        status(run, 'complete_pending_basic_checks')
    except BaseException as exc:
        status(run, 'failed', error=repr(exc)); raise


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('action', choices=['prepare', 'run']); p.add_argument('--run', type=Path, required=True)
    args = p.parse_args(); (prepare if args.action == 'prepare' else execute)(args.run.resolve())

"""Two small internal-editor GRPO -> harness selection -> solver RSFT cycles."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import time

from alternating_common import identity, run_editor, source_snapshot
from alternating_cycles import explored_proposals
from alternating_pool import Pool
from harness_bootstrap import messages
from math_self_replacement import read, rows, utc, status, score, summarize, model_spec
from paired_joint_whale import train_solver
from meta_harness.benchmark import compute_pareto_frontier
from meta_harness.meta_harness_retool import pick_accepted_from_frontier
from pipeline import ROOT, sha, write_json


def prepare(run):
    run.mkdir(parents=True, exist_ok=False)
    prior = ROOT / 'runs/whale_math_self_replacement_20260917'
    result = read(prior / 'RAPID_RESULT.json'); assert result['basic_checks_passed']
    previous = read(prior / 'config.json'); frozen = read(prior / 'FROZEN.json')
    harness = run / 'initial_harness.py'; shutil.copy2(frozen['internal_harness'], harness)
    initial_eval = read(prior / 'TEST_STATES.json')['W1_H_internal']['output']
    original = rows(Path(initial_eval) / 'trajectories.jsonl')
    model = identity(previous['solver_initial_checkpoint'])
    cfg = {'cycles': 2, 'reference': str(prior), 'solver_initial_checkpoint': model['model'],
        'editor_initial_checkpoint': frozen['editor_checkpoint'], 'initial_harness': str(harness),
        'initial_solver': model, 'initial_editor': identity(frozen['editor_checkpoint']),
        'data': previous['data'], 'pool_gpus': ['7', '8', '9'] + [f'node1:{i}' for i in range(4, 10)],
        'editor_gpu': '6', 'solver_gpu': '6', 'solver_rsft_lr': 3e-7,
        'editor_grpo_updates': 4, 'proposal_temperature': 1.0, 'fallback_temperature': 1.3,
        'max_extra_zero_variance_groups': 1, 'pre_candidates': 8, 'post_candidates': 4,
        'reward_tasks': 8, 'selection_candidates': 2, 'monitor_tasks': 64, 'seed': 42,
        'checkpoint_allowlist': [model['model']] + [str(run / f'cycle_{i:02d}/solver_update/round_01/checkpoint') for i in [1, 2]],
        'initial_development': str(prior / 'selection/reward_0_00'),
        'initial_monitor': str(initial_eval),
        'order': 'E proposes -> outcome GRPO E -> E proposes again -> select H -> on-policy RSFT W; repeat.',
        'limits': ['Rapid pilot; one seed, one draw per evaluation question.',
            'Historical baseline and test questions reused; no new independent-test claim.',
            'Editor grammar supports prompts and executable recovery hooks, max3 turns.',
            'No external model proposes or edits candidate harness code in this run.',
            'A zero-variance reward group gets at most one extra exploration group; skip RL if still zero variance.']}
    write_json(run / 'config.json', cfg)
    write_json(run / 'PROTOCOL.json', {'utc': utc(), 'config_sha256': sha(run / 'config.json'),
        'solver_rollouts': 256, 'new_monitor_draws_max': 256, 'final_extra_draws': 64,
        'no_cross_ablation': True, 'no_multi_seed': True, 'evaluation_used_for_selection': False,
        'two_separate_checkpoint_roles': True, 'phase_losses_are_retained': True})
    summarize(run / 'monitor/initial', cfg['data']['splits']['test'], model, harness, original[:64])
    source_snapshot(run, 'initial', ['rapid_internal_alternating.py', 'math_self_replacement.py',
        'alternating_cycles.py', 'paired_joint_whale.py', 'alternating_solver_train.py'])
    status(run, 'prepared')


def open_requests(harness, baseline, number, temperature):
    rr = rows(baseline / 'trajectories.jsonl'); failed = [r for r in rr if not r['reward']]
    feedback = {'tasks': len(rr), 'solved': int(sum(r['reward'] for r in rr)), 'failure_observations': []}
    for r in failed[:3]:
        tool_text = [m.get('content', '') for c in r['policy_calls'] for m in c['messages'] if m.get('role') == 'tool']
        feedback['failure_observations'].append({'assistant_tail': r['last_assistant'][-450:],
            'recent_tool_outputs': [str(x)[-250:] for x in tool_text[-2:]]})
    prompt = messages(harness.read_text(), json.dumps(feedback, ensure_ascii=False))
    prompt[-1]['content'] += (
        '\nChoose the improvement mechanism yourself from the observed failures. '
        'You may change reasoning guidance, exact arithmetic or useful Python verification, '
        'interpreter-error feedback, interpretation of tool outputs, or incomplete-response recovery. '
        'The existing harness already recovers incomplete answers; preserve useful existing behavior. '
        'Choose one substantive mechanism. Do not merely paraphrase the current nudge. '
        'Return the full executable CandidateEnv module, not a patch. All reference answers are hidden.')
    return [{'id': f'reward_0_{i:02d}', 'group': 'reward_0', 'messages': prompt,
        'seed': 95400 + number * 1000 + i, 'proposal_temperature': temperature} for i in range(8)]


def selection(pool, cycle, cfg, incumbent, after, indices):
    unique = {}
    for p in sorted(after, key=lambda p: (-p['reward_delta'], p['id'])):
        if p['outcome_kind'] == 'valid': unique.setdefault(hashlib.sha256(p['source'].encode()).hexdigest(), p)
    shortlisted = list(unique.values())[:2]
    data = cfg['data']['splits']['mh_val']; model = cfg['models']['W1']
    choices = {'incumbent': (incumbent, cycle / 'incumbent')}
    for p in shortlisted:
        harness = cycle / 'post_proposals' / (p['id'] + '.py')
        output = cycle / 'selection' / p['id']
        remainder = output / 'remaining'
        pool.evaluate_many([model_spec(model, harness, data, remainder,
            [[0, i] for i in range(data['count']) if i not in indices])])
        rr = rows(Path(p['reward_provenance']['candidate']) / 'trajectories.jsonl') + rows(remainder / 'trajectories.jsonl')
        summarize(output, data, model, harness, rr); choices[p['id']] = (harness, output)
    dest = cycle / 'selection'; dest.mkdir(exist_ok=True)
    values = {name: read(out / 'val.json') for name, (_, out) in choices.items()}
    for name, (source, _) in choices.items():
        target = dest / 'harnesses' / name / 'harness.py'; target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    frontier = {'_pareto': [{'harness': n, 'avg_success_rate': s, 'avg_mean_turn_count': t}
        for n, s, t in compute_pareto_frontier([(n, v['success_rate'], v['mean_turn_count']) for n, v in values.items()])]}
    _, accepted = pick_accepted_from_frontier(frontier, dest); assert accepted
    harness = dest / 'harnesses' / accepted / 'harness.py'
    write_json(dest / 'decision.json', {'accepted': accepted, 'results': values, 'frontier': frontier,
        'harness': str(harness), 'harness_sha256': sha(harness), 'changed': sha(harness) != sha(incumbent)})
    return harness


def execute(run):
    cfg = read(run / 'config.json'); assert sha(run / 'config.json') == read(run / 'PROTOCOL.json')['config_sha256']
    if not (run / 'START.json').exists(): write_json(run / 'START.json', {'utc': utc()})
    model = Path(cfg['solver_initial_checkpoint']); editor = Path(cfg['editor_initial_checkpoint'])
    harness = Path(cfg['initial_harness'])
    points = [{'label': 'initial', 'model': str(model), 'harness': str(harness), 'editor': str(editor)}]
    pool_dir = run / 'pool'
    if pool_dir.exists():
        assert (pool_dir / 'pool_stopped.json').exists(), 'Stop the previous pool before resuming.'
        i = 1
        while (run / f'pool_resume_{i:02d}').exists(): i += 1
        pool_dir = run / f'pool_resume_{i:02d}'
    try:
        with Pool(pool_dir, cfg) as pool:
            for number in range(1, cfg['cycles'] + 1):
                cycle = run / f'cycle_{number:02d}'; cycle.mkdir(exist_ok=True)
                if (cycle / 'complete.json').exists():
                    previous = read(cycle / 'complete.json')
                    model = Path(previous['solver']['model']); editor = Path(previous['editor']['model'])
                    harness = Path(previous['harness']); points += previous['points']; continue
                write_json(cycle / 'START.json', {'utc': utc(), 'solver': identity(model), 'editor': identity(editor),
                    'harness': str(harness), 'harness_sha256': sha(harness)})
                local = {**cfg, 'models': {'W1': identity(model)}, 'initial_harness': str(harness)}
                incumbent = cycle / 'incumbent'; data = cfg['data']['splits']['mh_val']
                status(run, 'harness_development', cycle=number)
                if number == 1 and not (incumbent / 'val.json').exists():
                    inherited = Path(cfg['initial_development']); value = read(inherited / 'val.json')
                    assert value['metadata']['weight_digest'] == identity(model)['weight_digest']
                    assert value['metadata']['harness_sha256'] == sha(harness)
                    incumbent.mkdir(); shutil.copy2(inherited / 'val.json', incumbent / 'val.json')
                    shutil.copy2(inherited / 'trajectories.jsonl', incumbent / 'trajectories.jsonl')
                    write_json(incumbent / 'REUSED.json', {'source': str(inherited), 'new_rollouts': 0})
                else: pool.evaluate_many([model_spec(identity(model), harness, data, incumbent)])
                base = rows(incumbent / 'trajectories.jsonl')
                selected = [r for r in base if not r['reward']][:4] + [r for r in base if r['reward']][:4]
                chosen = {r['example_index'] for r in selected}
                selected += [r for r in base if r['example_index'] not in chosen][:8 - len(selected)]
                indices = sorted(r['example_index'] for r in selected)
                reward_base = cycle / 'reward_baseline'; summarize(reward_base, data, identity(model), harness, selected)
                write_json(cycle / 'REWARD_BATCH.json', {'indices': indices, 'baseline_source': str(incumbent)})
                requests = open_requests(harness, reward_base, number, cfg['proposal_temperature'])
                status(run, 'editor_pre_generation', cycle=number)
                pre = cycle / 'pre_proposals'; explored_proposals(editor, pre, requests, cfg['proposal_temperature'])
                scored = score(pool, cycle, local, pre, indices, reward_base)
                fallback = False
                if len({r['reward_delta'] for r in scored}) == 1:
                    fallback = True; status(run, 'one_zero_variance_exploration', cycle=number)
                    requests = [dict(r, seed=r['seed'] + 10000) for r in requests]
                    pre = cycle / 'pre_exploration'
                    explored_proposals(editor, pre, requests, cfg['fallback_temperature'])
                    scored = score(pool, cycle, local, pre, indices, reward_base)
                status(run, 'editor_grpo', cycle=number, reward_deltas=[r['reward_delta'] for r in scored])
                trained = run_editor(run, editor, cycle / 'editor_grpo', scored, mode='train', epochs=4)
                before_editor = identity(editor)
                if trained['updated']:
                    assert trained['reload_verified']; editor = Path(trained['identity']['weights'])
                    assert identity(editor)['weight_digest'] != before_editor['weight_digest']
                status(run, 'updated_editor_generation', cycle=number, editor_updated=trained['updated'])
                post = cycle / 'post_proposals'
                explored_proposals(editor, post, requests[:4], cfg['fallback_temperature'] if fallback else cfg['proposal_temperature'])
                after = score(pool, cycle, local, post, indices, reward_base)
                status(run, 'harness_selection', cycle=number)
                harness = selection(pool, cycle, local, harness, after, indices)
                h_point = {'label': f'cycle_{number}_H', 'model': str(model), 'harness': str(harness), 'editor': str(editor)}
                points.append(h_point)
                write_json(cycle / 'EDITOR_RESULT.json', {'before': before_editor, 'after': identity(editor),
                    'updated': trained['updated'], 'extra_exploration': fallback, 'training_proposals': str(pre),
                    'pre_valid': sum(r['outcome_kind'] != 'invalid' for r in scored),
                    'pre_unique_sources': len({r.get('source') for r in scored if r['outcome_kind'] == 'valid'}),
                    'post_valid': sum(r['outcome_kind'] != 'invalid' for r in after),
                    'post_unique_sources': len({r.get('source') for r in after if r['outcome_kind'] == 'valid'}),
                    'pre_rewards': [r['reward_delta'] for r in scored], 'post_rewards': [r['reward_delta'] for r in after]})
                status(run, 'solver_on_policy_sampling', cycle=number)
                pool.evaluate_many([model_spec(identity(model), harness, cfg['data']['splits']['train'], cycle / 'solver_rollouts')])
                status(run, 'solver_rsft', cycle=number)
                if (cycle / 'solver_update/complete.json').exists():
                    model = Path(read(cycle / 'solver_update/complete.json')['identity']['weights'])
                else: model = train_solver(cycle, cfg, model, harness, cfg['solver_gpu'])
                assert read(cycle / 'solver_update/complete.json')['reload_verified']
                w_point = {'label': f'cycle_{number}_W', 'model': str(model), 'harness': str(harness), 'editor': str(editor)}
                points.append(w_point)
                write_json(cycle / 'complete.json', {'utc': utc(), 'solver': identity(model), 'editor': identity(editor),
                    'harness': str(harness), 'harness_sha256': sha(harness), 'points': [h_point, w_point]})
                write_json(run / 'STAGE_STATES.json', points)
                write_json(run / 'CURRENT_MODELS.json', {'solver': str(model), 'editor': str(editor),
                    'harness': str(harness), 'cycle': number})
            write_json(run / 'FROZEN.json', {'utc': utc(), 'points': points, 'monitor_used_for_selection': False})
            initial = points[0]
            cache = {(identity(initial['model'])['weight_digest'], sha(initial['harness'])): str(run / 'monitor/initial')}
            curve = []
            for p in points:
                key = (identity(p['model'])['weight_digest'], sha(p['harness']))
                if key not in cache:
                    output = run / 'monitor' / p['label']; status(run, 'stage_monitor', point=p['label'])
                    pool.evaluate_many([model_spec(identity(p['model']), Path(p['harness']), cfg['data']['splits']['test'],
                        output, [[0, i] for i in range(64)])]); cache[key] = str(output)
                value = read(Path(cache[key]) / 'val.json')
                curve.append({**p, 'output': cache[key], 'solved': value['solved_examples'], 'total': 64})
                write_json(run / 'CURVE.json', curve)
            status(run, 'final_remaining_64')
            remainder = run / 'final_remaining'
            pool.evaluate_many([model_spec(identity(model), harness, cfg['data']['splits']['test'], remainder,
                [[0, i] for i in range(64, 128)])])
            final_rows = rows(Path(curve[-1]['output']) / 'trajectories.jsonl') + rows(remainder / 'trajectories.jsonl')
            summarize(run / 'final128', cfg['data']['splits']['test'], identity(model), harness, final_rows)
        write_json(run / 'COMPLETE.json', {'utc': utc(), 'solver': identity(model), 'editor': identity(editor)})
        status(run, 'complete_pending_basic_checks')
    except BaseException as exc:
        status(run, 'failed', error=repr(exc)); raise


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('action', choices=['prepare', 'run']); p.add_argument('--run', type=Path, required=True)
    args = p.parse_args(); (prepare if args.action == 'prepare' else execute)(args.run.resolve())

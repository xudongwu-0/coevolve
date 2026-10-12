"""Relabel the same editor samples on the entire current development population."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import time

from alternating_common import identity, source_snapshot
from alternating_pool import Pool
from delta_rank_trial import train as rank_train, evaluate as rank_evaluate
from math_self_replacement import read, rows, dump_rows, utc, status, model_spec, summarize
from pipeline import ROOT, write_json, sha
from rapid_delta_editor import score, summary, select


def prepare(run, reference, rank_run):
    cycle = reference / 'cycle_02'; cfg = read(reference / 'config.json'); initial = read(cycle / 'START.json')
    run.mkdir(parents=True, exist_ok=False)
    cfg.update(source_run=str(reference), source_replay_cycle=str(cycle), source_cycle=str(run), rank_run=str(rank_run),
        solver_initial_checkpoint=initial['solver']['model'], checkpoint_allowlist=[initial['solver']['model']],
        editor_initial_checkpoint=rows(cycle / 'pre/scored.jsonl')[0]['sampling_identity']['weights'],
        initial_harness=initial['harness'], post_slots=2, reward_batch=48, new_solver_rollouts=0, editor_phases=1,
        objective='Current-solver full48 valid-edit gain ranking; nonnegative mean baseline; std floor1/12; clipped +/-2; invalid/no-op0; KL .02',
        full_reward_rule='Use all48 development tasks on the current solver; reuse old12 and already completed full48 candidate results.',
        adaptive_reason='The +3/12 bounds edit was -3/48 overall; repair the reward population before more optimization.')
    write_json(run / 'config.json', cfg); write_json(run / 'START.json', {'utc': utc(), **initial})
    write_json(run / 'PROTOCOL.json', {'utc': utc(), 'config_sha256': sha(run / 'config.json'),
        'pre_samples_reused': 8, 'fresh_post_samples': 2, 'editor_updates': 8, 'new_solver_updates': 0,
        'test_rule': 'Reuse main final128 when the selected code is identical; otherwise run128 once after full48 selection.'})
    for directory, names in [('pre', ['generations.jsonl', 'requests.jsonl']), ('incumbent', ['val.json', 'trajectories.jsonl'])]:
        target = run / directory; target.mkdir()
        for name in names: shutil.copy2(cycle / directory / name, target / name)
    for path in (cycle / 'pre').glob('edit_*.py'): shutil.copy2(path, run / 'pre' / path.name)
    source_snapshot(run, 'initial', ['harness_delta.py', 'harness_delta_worker.py', 'rapid_delta_editor.py', 'delta_rank_trial.py', 'delta_full_reward_trial.py'])
    status(run, 'prepared')


def calibrate(run, pool):
    cfg = read(run / 'config.json'); reference = Path(cfg['source_run']); cycle = Path(cfg['source_replay_cycle'])
    model = identity(cfg['solver_initial_checkpoint']); harness = Path(cfg['initial_harness']); data = cfg['data']['splits']['mh_val']
    original = rows(cycle / 'pre/scored.jsonl'); schedule = [[0, i] for i in range(data['count'])]
    full_known = [cycle / 'incumbent'] + [p.parent for p in (cycle / 'selection').glob('*/val.json')]
    rank_run = Path(cfg['rank_run'])
    if (rank_run / 'selection').exists(): full_known += [p.parent for p in (rank_run / 'selection').glob('*/val.json')]
    plans = []; jobs = []; unique = set()
    for g in original:
        if g['outcome_kind'] != 'valid': continue
        candidate = run / 'pre' / (g['id'] + '.py'); key = hashlib.sha256(json.dumps([model['weight_digest'], sha(candidate), sha(harness), data['sha256'], schedule, 42, 1.0]).encode()).hexdigest()
        if key in unique: continue
        unique.add(key); output = reference / 'outcome_cache' / key
        if (output / 'val.json').exists(): continue
        known = None
        for directory in full_known:
            value = read(directory / 'val.json'); meta = value['metadata']
            if meta['weight_digest'] == model['weight_digest'] and meta['harness_sha256'] == sha(candidate) and meta['sampling_schedule'] == schedule:
                assert meta['dataset_sha256'] == data['sha256'] and meta['seed'] == 42 and meta['temperature'] == 1.0
                known = directory; break
        if known:
            summarize(output, data, model, candidate, rows(known / 'trajectories.jsonl'))
            write_json(output / 'REUSED.json', {'source': str(known), 'new_rollouts': 0}); continue
        part = rows(Path(g['reward_provenance']['candidate']) / 'trajectories.jsonl'); used = {r['example_index'] for r in part}
        remainder = output / 'remaining'
        jobs.append(model_spec(model, candidate, data, remainder, [[0, i] for i in range(data['count']) if i not in used]))
        plans.append((output, candidate, part, remainder))
    status(run, 'complete_current_solver_rewards', new_rollouts=sum(len(j['schedule']) for j in jobs))
    pool.evaluate_many(jobs)
    for output, candidate, part, remainder in plans:
        summarize(output, data, model, candidate, part + rows(remainder / 'trajectories.jsonl'))
    corrected = score(pool, reference, cfg, model, harness, run / 'pre', run / 'incumbent')
    comparison = [{'id': a['id'], 'kind': a['outcome_kind'], 'sample12_gain': a['reward_delta'],
        'full48_gain': b['reward_delta'], 'full48_solved': b['reward_provenance']['solved'], 'patch': a.get('patch')}
        for a, b in zip(original, corrected)]
    write_json(run / 'CALIBRATION.json', {'new_rollouts': sum(len(j['schedule']) for j in jobs), 'labels': comparison})
    return corrected


def execute(run):
    cfg = read(run / 'config.json'); reference = Path(cfg['source_run']); rank_run = Path(cfg['rank_run'])
    while not (reference / 'COMPLETE.json').exists():
        if read(reference / 'STATUS.json')['stage'] == 'failed': raise RuntimeError('Main run failed')
        time.sleep(3)
    while read(rank_run / 'STATUS.json')['stage'] != 'ready_for_evaluation': time.sleep(3)
    with Pool(run / 'pool', cfg) as pool:
        rank_evaluate(rank_run, shared_pool=pool)
        corrected = calibrate(run, pool)
        rank_train(run)
        model = identity(cfg['solver_initial_checkpoint']); harness = Path(cfg['initial_harness'])
        status(run, 'full_reward_post_evaluation')
        after = score(pool, reference, cfg, model, harness, run / 'post', run / 'incumbent')
        chosen = select(pool, cfg, run, model, harness, after)
        selection = read(run / 'selection/decision.json')
        write_json(run / 'FROZEN.json', {'utc': utc(), 'harness': str(chosen), 'selection': selection['accepted'], 'test_used': False})
        main_models = read(reference / 'CURRENT_MODELS.json')
        if identity(main_models['solver']) == model and sha(main_models['harness']) == sha(chosen):
            output = run / 'final128'; output.mkdir()
            for name in ['val.json', 'trajectories.jsonl']: shutil.copy2(reference / 'final128' / name, output / name)
            write_json(output / 'REUSED.json', {'source': str(reference / 'final128'), 'new_rollouts': 0})
        else:
            status(run, 'full_reward_final128')
            pool.evaluate_many([model_spec(model, chosen, cfg['data']['splits']['test'], run / 'final128')])
        trained = read(run / 'editor_update/complete.json')
        result = {'pre_full48': summary(corrected), 'post_full48': summary(after), 'selection_scores': {k:v['solved_examples'] for k,v in selection['results'].items()},
            'selected': selection['accepted'], 'final128': read(run / 'final128/val.json')['solved_examples'],
            'editor': trained['identity']['weights'], 'solver': model['model'], 'harness': str(chosen),
            'new_editor_updates': trained['new_updates'], 'new_solver_updates': 0,
            'new_reward_calibration_rollouts': read(run / 'CALIBRATION.json')['new_rollouts']}
        write_json(run / 'RESULT.json', result); write_json(run / 'CURRENT_MODELS.json', {k:result[k] for k in ['editor','solver','harness']})
    write_json(run / 'COMPLETE.json', {'utc': utc()}); status(run, 'complete')


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('action', choices=['prepare', 'run']); p.add_argument('--run', type=Path, required=True)
    p.add_argument('--reference', type=Path); p.add_argument('--rank-run', type=Path); args = p.parse_args()
    run = args.run.resolve()
    if args.action == 'prepare': prepare(run, args.reference.resolve(), args.rank_run.resolve())
    else: execute(run)

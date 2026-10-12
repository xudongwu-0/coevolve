"""One adaptive, same-data editor update comparison; never trains the solver."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
from contextlib import nullcontext

from alternating_common import identity, source_snapshot
from alternating_pool import Pool
from harness_delta import gain_rank_advantages
from math_self_replacement import read, rows, dump_rows, model_spec, utc, status
from pipeline import ROOT, write_json, sha
from rapid_delta_editor import editor_call, score, summary, select


def prepare(run, reference, phase):
    cycle = reference / f'cycle_{phase:02d}'
    sampling_checkpoint = rows(cycle / 'pre/scored.jsonl')[0]['sampling_identity']['weights']
    run.mkdir(parents=True, exist_ok=False)
    cfg = read(reference / 'config.json'); start = read(cycle / 'START.json')
    cfg.update(solver_initial_checkpoint=start['solver']['model'], checkpoint_allowlist=[start['solver']['model']],
        source_run=str(reference), source_cycle=str(cycle), editor_initial_checkpoint=sampling_checkpoint,
        objective='Current-solver valid-edit gain ranking; nonnegative mean baseline; std floor1/12; clipped +/-2; invalid/no-op0; KL .02',
        editor_phases=1, new_solver_rollouts=0,
        initial_harness=start['harness'], comparison='Same eight sampled edits and measured current-solver outcomes; change advantage only.',
        rank_test_rule='Evaluate chosen harness on64. Complete128 only if full-development score exceeds the paired-update choice; never choose using test scores.')
    write_json(run / 'config.json', cfg); write_json(run / 'START.json', {'utc': utc()})
    write_json(run / 'PROTOCOL.json', {'utc': utc(), 'config_sha256': sha(run / 'config.json'), 'phase': phase,
        'reason': 'Paired-only update lowered the first-phase matched proposal gains; test relative preference among valid measured edits.'})
    source_snapshot(run, 'initial', ['harness_delta.py', 'harness_delta_worker.py', 'rapid_delta_editor.py', 'delta_rank_trial.py'])
    status(run, 'prepared')


def train(run):
    cfg = read(run / 'config.json'); cycle = Path(cfg['source_cycle']); calls = rows(cycle / 'pre/scored.jsonl')
    adv = gain_rank_advantages(calls)
    if not any(adv):
        write_json(run / 'NO_SIGNAL.json', {'advantages': adv, 'reason': 'No usable gain contrast; no optimizer step requested.'}); return
    directory = run / 'editor_update'; directory.mkdir(exist_ok=True); data = directory / 'requests.jsonl'; dump_rows(data, calls)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='6', WHALE_RUN_ROOT=str(directory), WHALE_MODEL_ROOT=cfg['editor_initial_checkpoint'],
        WHALE_PARAMETER_DTYPE='float32', HF_HUB_OFFLINE='1', TOKENIZERS_PARALLELISM='false',
        PYTHONPATH=':'.join(str(ROOT / p) for p in ['whale_local', 'WHALE/domains/math_reasoning', 'WHALE/domains/chess_puzzles']))
    status(run, 'rank_editor_update', advantages=adv)
    if not (directory / 'complete.json').exists():
        with directory.with_suffix('.log').open('a') as log:
            subprocess.run([str(ROOT / '.venv-whale-portable/bin/python'), '-u', str(ROOT / 'whale_local/harness_delta_worker.py'),
                'train', '--data', str(data), '--epochs', '8', '--learning-rate', '3e-6', '--advantage-mode', 'rank'],
                env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
    result = read(directory / 'complete.json'); assert result['updated'] and result['reload_verified']
    editor = Path(result['identity']['weights']); status(run, 'rank_post_generation')
    requests = rows(cycle / 'pre/requests.jsonl')[:cfg.get('post_slots', 4)]
    editor_call(editor, run / 'post', requests)
    status(run, 'ready_for_evaluation')


def evaluate(run, shared_pool=None):
    cfg = read(run / 'config.json'); reference = Path(cfg['source_run']); cycle = Path(cfg['source_cycle'])
    assert (reference / 'COMPLETE.json').exists(), 'Let the main run release its sampler pool first.'
    model = identity(cfg['solver_initial_checkpoint']); harness = Path(cfg['initial_harness'])
    incumbent = run / 'incumbent'; incumbent.mkdir(exist_ok=True)
    for name in ['val.json', 'trajectories.jsonl']: shutil.copy2(cycle / 'incumbent' / name, incumbent / name)
    with (nullcontext(shared_pool) if shared_pool is not None else Pool(run / 'pool', cfg)) as pool:
        status(run, 'rank_candidate_rewards')
        scored = score(pool, reference, cfg, model, harness, run / 'post', cycle / 'reward_baseline')
        status(run, 'rank_harness_selection', post=summary(scored))
        prior_decision = read(cycle / 'selection/decision.json')
        full_cache = [cycle / 'selection' / name for name in prior_decision['results'] if name != 'incumbent'] + [cycle / 'incumbent']
        chosen = select(pool, cfg, run, model, harness, scored, full_result_cache=full_cache)
        decision = read(run / 'selection/decision.json'); paired = read(cycle / 'selection/decision.json')
        selected_score = decision['results'][decision['accepted']]['solved_examples']
        paired_score = paired['results'][paired['accepted']]['solved_examples']
        # Choose whether this branch merits the second half before observing test scores.
        complete128 = selected_score > paired_score
        write_json(run / 'FROZEN.json', {'utc': utc(), 'harness': str(chosen), 'selected_dev48': selected_score,
            'paired_dev48': paired_score, 'complete128_by_development_only': complete128})
        reused = None
        for point in read(reference / 'CURVE.json'):
            if identity(point['model']) == model and sha(point['harness']) == sha(chosen): reused = Path(point['output']); break
        if reused is not None:
            monitor = run / 'monitor64'; monitor.mkdir(exist_ok=True)
            for name in ['val.json', 'trajectories.jsonl']: shutil.copy2(reused / name, monitor / name)
        else:
            status(run, 'rank_monitor64')
            pool.evaluate_many([model_spec(model, chosen, cfg['data']['splits']['test'], run / 'monitor64', [[0, i] for i in range(64)])])
        if complete128:
            status(run, 'rank_complete128')
            pool.evaluate_many([model_spec(model, chosen, cfg['data']['splits']['test'], run / 'remaining64', [[0, i] for i in range(64, 128)])])
        result = {'post': summary(scored), 'paired_post': read(cycle / 'EDITOR_RESULT.json')['post'],
            'pre_matched4': read(cycle / 'EDITOR_RESULT.json')['pre_matched4'], 'selected_dev48': selected_score,
            'paired_dev48': paired_score, 'rank_solved64': read(run / 'monitor64/val.json')['solved_examples'],
            'rank_solved128': (read(run / 'monitor64/val.json')['solved_examples'] + read(run / 'remaining64/val.json')['solved_examples']) if complete128 else None,
            'editor': read(run / 'editor_update/complete.json')['identity']['weights'], 'solver': model['model'], 'harness': str(chosen),
            'new_editor_updates': 8, 'new_solver_updates': 0, 'same_source_data_and_sampling_checkpoint': True}
        write_json(run / 'RESULT.json', result)
    write_json(run / 'COMPLETE.json', {'utc': utc()}); status(run, 'complete')


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('action', choices=['prepare', 'train', 'evaluate']); p.add_argument('--run', type=Path, required=True)
    p.add_argument('--reference', type=Path); p.add_argument('--phase', type=int, default=2)
    args = p.parse_args(); run = args.run.resolve()
    if args.action == 'prepare': prepare(run, args.reference.resolve(), args.phase)
    else: (train if args.action == 'train' else evaluate)(run)

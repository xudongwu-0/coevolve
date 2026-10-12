"""Evaluate an immutable stage using the existing serialized worker queues.

This schedules no extra evaluation: the main runner reuses the resulting val.
Only use while workers hold the same solver and the editor/next solver trains.
"""
import argparse
import os
import signal
from pathlib import Path
from alternating_common import identity
from alternating_pool import Pool
from math_self_replacement import read, model_spec
from pipeline import write_json


class ExistingWorker:
    def __init__(self, path): self.path = path
    def poll(self): return 0 if (self.path / 'stop').exists() else None


def main(run, point, pool_path):
    cfg = read(run / 'config.json')
    assert Path(point['model']).exists() and Path(point['harness']).exists()
    pool = Pool(pool_path, cfg)
    directories = sorted((pool_path / 'replicas').glob('slot_*'))
    assert len(directories) == len(cfg['pool_gpus'])
    # Each live worker executes its queue serially. UUID-like nanosecond job
    # filenames and separate output paths keep the two coordinators independent.
    pool.processes = [(d, ExistingWorker(d), None) for d in directories]
    output = run / 'monitor' / point['label']
    write_json(run / (point['label'] + '_PREFETCH.json'), {'point': point,
        'pool': str(pool_path), 'same_evaluation_reused_by_main': True, 'scores_not_used_for_training': True})
    pool.evaluate_many([model_spec(identity(point['model']), Path(point['harness']),
        cfg['data']['splits']['test'], output, [[0, i] for i in range(64)])])
    print('Prefetched fixed64 stage; scores left for the final report.', flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--run', type=Path, required=True)
    p.add_argument('--point', required=True); p.add_argument('--model'); p.add_argument('--harness')
    p.add_argument('--pause-coordinator', type=int)
    args = p.parse_args(); run = args.run.resolve()
    if args.model and args.harness:
        point = {'label': args.point, 'model': args.model, 'harness': args.harness}
    else: point = next(x for x in read(run / 'STAGE_STATES.json') if x['label'] == args.point)
    pid = args.pause_coordinator
    if pid:
        command = Path(f'/proc/{pid}/cmdline').read_bytes()
        assert b'rapid_internal_alternating.py' in command and str(run.name).encode() in command
        assert read(run / 'STATUS.json')['stage'] == 'solver_rsft'
        # Pause only the waiting coordinator. Its GPU-training child and sampler
        # workers continue, and cannot race a second evaluation of this output.
        os.kill(pid, signal.SIGSTOP)
    try:
        main(run, point, run / 'pool')
    finally:
        if pid: os.kill(pid, signal.SIGCONT)

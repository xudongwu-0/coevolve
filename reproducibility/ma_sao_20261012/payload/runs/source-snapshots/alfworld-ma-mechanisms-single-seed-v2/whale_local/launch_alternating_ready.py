"""Launch the fixed two-cycle trial after the registered editor readiness test."""
import argparse
import datetime
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

from pipeline import ROOT, sha, write_json


def read(p): return json.loads(Path(p).read_text())


def main():
    p = argparse.ArgumentParser(); p.add_argument('--run', type=Path, required=True)
    p.add_argument('--composition', type=Path, required=True); args = p.parse_args()
    run = args.run.resolve(); composition = args.composition.resolve()
    deadline = time.monotonic() + 3600
    while not (composition / 'READY.json').exists():
        if (composition / 'NOT_READY.json').exists(): raise RuntimeError('Editor failed registered readiness')
        if time.monotonic() > deadline: raise TimeoutError('Editor readiness did not finish')
        time.sleep(2)
    ready = read(composition / 'READY.json'); assert ready['passed_readiness']
    origin = ROOT / 'runs/whale_alternating_grpo_20260916'
    if run.exists(): raise FileExistsError('Use a fresh run to preserve partial attempts')
    run.mkdir()
    for name in ['baselines', 'editor_probes', 'replicas', 'data']:
        shutil.copytree(origin / name, run / name)
    shutil.copytree(origin / 'source_snapshots/initial', run / 'source_snapshots/initial')
    for name in ['INITIAL_COMPLETE.json', 'initial_checkpoint_results.json', 'initial_verification.json',
        'base_editable.py', 'pool_stopped.json', 'R1_REFERENCE.json', 'GRPO_OBJECTIVE_CHECK.json']:
        shutil.copy2(origin / name, run / name)
    # Copy only cached outcomes actually referenced by the initial probes.
    outcomes = set()
    for file in (origin / 'editor_probes').glob('*/scored.jsonl'):
        for row in map(json.loads, file.read_text().splitlines()):
            if row['reward_provenance']['candidate']: outcomes.add(Path(row['reward_provenance']['candidate']))
    for path in outcomes: shutil.copytree(path, run / path.relative_to(origin))
    config = read(origin / 'config.json')
    config.update(initial_probe_origin=str(origin), initial_probe_verification_sha256=sha(origin / 'initial_verification.json'),
        initial_probe_E1_checkpoint=config['editor_initial_checkpoint'], editor_initial_checkpoint=ready['checkpoint'],
        composition_bootstrap=str(composition), composition_readiness_sha256=sha(composition / 'READY.json'),
        harness_reward_temperature=1.0, harness_selection_temperature=1.0,
        training_proposal_temperature=1.0, fallback_proposal_temperature=1.2, editor_request_mode='strategy')
    config['limits'][-1] = 'Editor continued on synthetic composition SFT; generic guidance-edit interface, not unrestricted code diagnosis.'
    config['previous_stopped_attempts'] = [str(origin), str(ROOT / 'runs/whale_alternating_stochastic_20260916')]
    write_json(run / 'config.json', config)
    write_json(run / 'PROTOCOL.json', {'registered_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'config_sha256': sha(run / 'config.json'), 'cycles': 2, 'before_any_new_rollout': True,
        'monitor_only_never_reward_or_selection': True, 'initial_probe_origin': str(origin),
        'composition_readiness_required': True, 'composition_protocol_sha256': sha(composition / 'PROTOCOL.json'),
        'harness_r1_paper': 'https://arxiv.org/html/2608.02276v1#S3.SS2',
        'harness_r1_local_commit': '411bb5489ada85e6118897039eab84b4e4aaa4f0',
        'change_reason': 'Earlier attempts had no editor update: copying incumbent/invalid code. Continued synthetic composition SFT before outcome RL.'})
    env = dict(os.environ, PYTHONPATH=':'.join(str(ROOT / d) for d in ['whale_local', 'WHALE/domains/math_reasoning']))
    with (run / 'cycles.log').open('a') as log:
        subprocess.run([str(ROOT / '.venv-whale-search/bin/python'), '-u', str(ROOT / 'whale_local/alternating_cycles.py'),
            '--run', str(run)], env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
    with (run / 'audit.log').open('a') as log:
        subprocess.run([str(ROOT / '.venv-whale-portable/bin/python'), '-u', str(ROOT / 'whale_local/audit_alternating.py'),
            '--run', str(run)], env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
    subprocess.run([str(ROOT / '.venv-whale-search/bin/python'), str(ROOT / 'whale_local/report_alternating.py'),
        '--run', str(run)], env=env, check=True)


if __name__ == '__main__': main()

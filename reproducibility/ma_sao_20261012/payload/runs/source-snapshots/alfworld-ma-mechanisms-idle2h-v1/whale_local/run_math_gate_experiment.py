"""Run and audit one qualified, matched-budget Math gate experiment."""
import argparse
import datetime
import json
import os
from pathlib import Path
import subprocess

from pipeline import ROOT, sha, write_json


def read(path):
    return json.loads(Path(path).read_text())


def free_devices(command, expected):
    output = subprocess.check_output(command, text=True, timeout=20)
    rows = {int(parts[0]): (int(parts[1]), int(parts[2]))
            for line in output.splitlines() if (parts := line.split(','))}
    for gpu in expected:
        memory, utilization = rows[gpu]
        assert memory < 1000 and utilization == 0, f'GPU {gpu} is occupied: {rows[gpu]}'
    return {str(gpu): {'memory_used_mib': rows[gpu][0], 'utilization_percent': rows[gpu][1]}
            for gpu in expected}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--run-dir', type=Path, required=True)
    args = parser.parse_args()
    reference, run = args.reference.resolve(), args.run_dir.resolve()
    qualification = read(reference / 'verification.json')
    assert qualification['passed_audit'] and qualification['qualified_for_gate'], \
        'An independently qualified baseline is required before any gate work'
    assert not run.exists(), 'Use a fresh run; existing runs retain their exact launch_command.json'
    query = ['nvidia-smi', '--query-gpu=index,memory.used,utilization.gpu',
             '--format=csv,noheader,nounits']
    local = free_devices(query, [6, 7, 8, 9])
    from math_remote import HOST, MANAGEMENT_SSH
    if not HOST:
        raise RuntimeError('Set COEVOLVE_REMOTE_HOST before running this historical two-node gate launcher')
    remote = free_devices([*MANAGEMENT_SSH, HOST, *query],
        [4, 5, 6, 7, 8, 9])
    env = dict(os.environ, PYTHONPATH=str(ROOT / 'whale_local') + ':' +
               str(ROOT / 'WHALE/domains/math_reasoning'))
    subprocess.run([str(ROOT / '.venv-whale-portable/bin/python'), '-u',
        str(ROOT / 'whale_local/prepare_math_gate_experiment.py'),
        '--reference', str(reference), '--run-dir', str(run)],
        env=dict(env, CUDA_VISIBLE_DEVICES=''), check=True)
    write_json(run / 'GPU_PREFLIGHT.json', {'checked_before_launch': True,
        'local': local, 'node1': remote, 'other_gpu_jobs_not_modified': True})
    started = datetime.datetime.now(datetime.timezone.utc).isoformat()
    command = read(run / 'launch_command.json')['argv']
    with (run / 'pipeline.log').open('w') as log:
        process = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT)
        write_json(run / 'LAUNCH.json', {'started_utc': started, 'coordinator_pid': process.pid,
            'argv': command, 'launcher_sha256': sha(__file__),
            'qualification_sha256': sha(reference / 'verification.json')})
        write_json(run / 'STATUS.json', {'stage': 'gate_pipeline_running',
            'started_utc': started, 'coordinator_pid': process.pid,
            'teacher': 'active Codex via the local FastAPI queue'})
        try:
            result = process.wait()
            assert result == 0, f'Gate pipeline exited {result}; inspect {run / "pipeline.log"}'
        finally:
            if process.poll() is None:
                process.terminate()
                process.wait(timeout=55)
    write_json(run / 'STATUS.json', {'stage': 'gate_pipeline_finished_audit_pending',
        'started_utc': started, 'test_used': True})
    with (run / 'audit.log').open('w') as log:
        subprocess.run([str(ROOT / '.venv-whale-portable/bin/python'), '-u',
            str(ROOT / 'whale_local/verify_math_gate_complete.py'), str(run)],
            env=dict(env, CUDA_VISIBLE_DEVICES=''), stdout=log, stderr=subprocess.STDOUT, check=True)
    audit = read(run / 'verification.json')
    assert audit['passed_audit'] and audit['frozen_sources_and_service_cleanup_verified']
    ended = datetime.datetime.now(datetime.timezone.utc)
    write_json(run / 'STATUS.json', {'stage': 'complete_audited', 'test_used': True,
        'started_utc': started, 'finished_utc': ended.isoformat(),
        'elapsed_seconds': (ended - datetime.datetime.fromisoformat(started)).total_seconds(),
        'gate_minus_whale_pp': audit['contrasts']['gate_minus_whale']['difference_pp']})
    print(json.dumps({'complete_audited': True, 'scores': audit['scores'],
        'gate_minus_whale': audit['contrasts']['gate_minus_whale']}, indent=2))


if __name__ == '__main__':
    main()

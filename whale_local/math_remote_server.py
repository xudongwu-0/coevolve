"""Own exactly one remote policy/sandbox pair; standard-library supervisor."""
import argparse
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--stop', action='store_true')
    parser.add_argument('--port', type=int)
    args = parser.parse_args()
    run = args.run.resolve()
    if args.stop:
        path = run / 'remote_supervisor.json'
        if not path.exists():
            return
        pid = json.loads(path.read_text())['pid']
        proc = Path(f'/proc/{pid}/cmdline')
        if proc.exists():
            command = proc.read_bytes().split(b'\0')
            assert str(Path(__file__).resolve()).encode() in command
            assert str(run).encode() in command
            os.kill(pid, signal.SIGTERM)
        return
    assert args.port is not None
    for port in [args.port, args.port + 1000]:
        with socket.socket() as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            sock.bind(('127.0.0.1', port))
    run.mkdir(parents=True, exist_ok=True)
    (run / 'remote_supervisor.json').write_text(json.dumps({'pid': os.getpid(), 'run': str(run)}))
    root = Path(__file__).resolve().parents[1]
    stop = False
    def request_stop(*_):
        nonlocal stop
        stop = True
    for sig in [signal.SIGTERM, signal.SIGHUP, signal.SIGINT]:
        signal.signal(sig, request_stop)
    processes = []
    with (run/'worker.log').open('a') as worker_log, (run/'sandbox.log').open('a') as sandbox_log:
        try:
            sandbox = subprocess.Popen([sys.executable, '-u', str(root/'whale_local/math_sandbox.py'),
                '--port', str(args.port+1000), '--log-dir', str(run/'sandbox_calls')], stdout=sandbox_log, stderr=subprocess.STDOUT)
            processes.append(sandbox)
            worker = subprocess.Popen([sys.executable, '-u', str(root/'whale_local/math_worker.py')], stdout=worker_log, stderr=subprocess.STDOUT)
            processes.append(worker)
            (run/'services.json').write_text(json.dumps({'worker_pid':worker.pid,'sandbox_pid':sandbox.pid,'remote':True}))
            while not stop:
                if any(p.poll() is not None for p in processes):
                    raise RuntimeError('Remote worker or sandbox exited; inspect copied logs')
                time.sleep(.5)
        finally:
            for p in processes:
                if p.poll() is None:
                    p.terminate()
            for p in processes:
                try:
                    p.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    p.kill();p.wait()
            (run/'services_stopped.json').write_text(json.dumps({'worker_pid':worker.pid,
                'sandbox_pid':sandbox.pid,'worker_returncode':worker.returncode,'sandbox_returncode':sandbox.returncode,'remote':True}))

if __name__ == '__main__':
    main()

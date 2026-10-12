"""Authenticated SSH transport for remote frozen samplers; credentials never stored."""
from contextlib import contextmanager
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import time

import pipeline

HOST = os.environ.get('COEVOLVE_REMOTE_HOST', '')
CONTROL = os.environ.get('COEVOLVE_SSH_CONTROL', '/tmp/coevolve-ssh/control')
SSH = ['ssh', '-S', CONTROL, '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=10']
MANAGEMENT_SSH = ['ssh', '-S', str(Path(CONTROL).with_name('management')),
                  '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=10']
RSYNC_SSH = shlex.join(MANAGEMENT_SSH)
ROOT = pipeline.ROOT


def ssh_command(command, **kwargs):
    return subprocess.run(MANAGEMENT_SSH + [HOST, shlex.join(command)], check=True, **kwargs)


def ensure_model(model):
    """One immutable copy per model path, shared by all remote GPU replicas."""
    if not HOST:
        raise RuntimeError('Set COEVOLVE_REMOTE_HOST=user@host before using node1 GPU replicas')
    Path(CONTROL).parent.mkdir(parents=True, exist_ok=True)
    model = Path(model).resolve()
    identity = [(p.name,p.stat().st_size,p.stat().st_mtime_ns) for p in sorted(model.iterdir()) if p.is_file()]
    digest = hashlib.sha256(json.dumps([str(model),identity]).encode()).hexdigest()
    cache = Path(CONTROL).parent
    marker = cache / (digest + '.ready')
    with (cache/(digest+'.lock')).open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if marker.exists():
            return
        ssh_command(['mkdir','-p',str(model)])
        subprocess.run(['rsync','-aL','--partial','-e',RSYNC_SSH,str(model)+'/',HOST+':'+str(model)+'/'],check=True)
        marker.write_text(json.dumps({'source':str(model),'signature':digest}))


@contextmanager
def service(run, model, gpu, port, learning_rate, parameter_dtype):
    assert gpu.startswith('node1:') and gpu[6:].isdigit() and 0 <= int(gpu[6:]) < 10
    run, model = Path(run).resolve(), Path(model).resolve()
    run.mkdir(parents=True,exist_ok=True)
    ensure_model(model)
    env = {'CUDA_VISIBLE_DEVICES':gpu[6:],'WHALE_RUN_ROOT':str(run),'WHALE_MODEL_ROOT':str(model),
        'WHALE_POLICY_PORT':str(port),'WHALE_LEARNING_RATE':str(learning_rate),'WHALE_PARAMETER_DTYPE':parameter_dtype,
        'HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1','TOKENIZERS_PARALLELISM':'false',
        'PYTHONPATH':':'.join(str(p) for p in [ROOT/'whale_local',ROOT/'WHALE/domains/math_reasoning',pipeline.DOMAIN]),
        'NO_PROXY':'127.0.0.1,localhost','no_proxy':'127.0.0.1,localhost'}
    if os.environ.get('WHALE_CHECKPOINT_ALLOWLIST'):
        env['WHALE_CHECKPOINT_ALLOWLIST']=os.environ['WHALE_CHECKPOINT_ALLOWLIST']
    script = str(ROOT/'whale_local/math_remote_server.py')
    python = str(ROOT/'.venv-whale-portable/bin/python')
    forwards = ['-L',f'127.0.0.1:{port}:127.0.0.1:{port}',
                '-L',f'127.0.0.1:{port+1000}:127.0.0.1:{port+1000}']
    opened, child = False, None
    with (run/'remote_transport.log').open('a') as log:
        try:
            subprocess.run(SSH+['-O','forward',*forwards,HOST],check=True,stdout=log,stderr=subprocess.STDOUT)
            opened = True
            command = ['env',*(f'{k}={v}' for k,v in env.items()),python,'-u',script,'--run',str(run),'--port',str(port)]
            child = subprocess.Popen(SSH+[HOST,'exec '+shlex.join(command)],stdout=log,stderr=subprocess.STDOUT)
            base_url = f'http://127.0.0.1:{port}'
            identity = pipeline.wait_ready(child,base_url)
            pipeline.wait_ready(child,f'http://127.0.0.1:{port+1000}')
            pipeline.write_json(run/'remote_transport.json',{'host':HOST,'gpu':gpu[6:],'port':port,
                'ssh_pid':child.pid,'weight_digest':identity['weight_digest'],'ssh_forwarded_loopback_only':True})
            yield base_url, identity
        finally:
            try:
                if child is not None:
                    ssh_command([python,script,'--run',str(run),'--stop'],stdout=log,stderr=subprocess.STDOUT)
                    try:
                        child.wait(timeout=45)
                    except subprocess.TimeoutExpired:
                        child.terminate();child.wait(timeout=10)
                    # Only this service's evidence is copied; no credentials or other runs.
                    subprocess.run(['rsync','-a','-e',RSYNC_SSH,HOST+':'+str(run)+'/',str(run)+'/'],check=True,
                                   stdout=log,stderr=subprocess.STDOUT)
            finally:
                if opened:
                    subprocess.run(SSH+['-O','cancel',*forwards,HOST],check=True,stdout=log,stderr=subprocess.STDOUT)

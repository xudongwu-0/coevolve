"""Shared identities, proposal rewards, and real-rollout checkpoint probes."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

from alternating_pool import rows,read
from harness_bootstrap import START
from harness_focused_bootstrap import request
from math_runtime import BASE_HARNESS
from pipeline import ROOT,sha,write_json


def identity(path):
    path=Path(path).resolve();s=read(path/'training_state.json')
    return {'model':str(path),'weight_digest':s['weight_digest']}


def run_editor(run,checkpoint,directory,requests,*,gpu='6',mode='generate',epochs=4):
    directory=Path(directory)
    if (directory/'complete.json').exists():return read(directory/'complete.json')
    directory.mkdir(parents=True,exist_ok=True)
    data=directory/'requests.jsonl' if mode=='generate' else directory/'rewards.jsonl'
    data.write_text(''.join(json.dumps(r)+'\n' for r in requests))
    env=dict(os.environ,CUDA_VISIBLE_DEVICES=gpu,WHALE_RUN_ROOT=str(directory),WHALE_MODEL_ROOT=str(checkpoint),
        WHALE_PARAMETER_DTYPE='float32',WHALE_LEARNING_RATE='1e-6',HF_HUB_OFFLINE='1',TOKENIZERS_PARALLELISM='false',
        PYTHONPATH=':'.join(str(ROOT/p) for p in ['whale_local','WHALE/domains/math_reasoning','WHALE/domains/chess_puzzles']))
    command=[str(ROOT/'.venv-whale-portable/bin/python'),'-u',str(ROOT/'whale_local/harness_grpo.py'),mode,
        '--data',str(data),'--epochs',str(epochs),'--temperature','0.7','--learning-rate','1e-6']
    with directory.with_suffix('.log').open('a') as log:
        subprocess.run(command,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
    return read(directory/'complete.json')


def evidence(baseline):
    data=rows(Path(baseline)/'trajectories.jsonl');failed=[r for r in data if not r['reward']]
    packet={'tasks':len(data),'solved':int(sum(r['reward'] for r in data)),
        'missing_boxed':sum('\\boxed{' not in r['last_assistant'] for r in failed),
        'failure_observations':[]}
    for r in failed[:2]:
        packet['failure_observations'].append({'assistant_tail':r['last_assistant'][-350:],
            'tool_feedback':[str(t.get('env_feedback',''))[-200:] for t in r.get('harness_trace',[])][-1:]})
    return json.dumps(packet,ensure_ascii=False)


def requests_for(group,source,baseline,focus,count,seed):
    messages=request(source,evidence(baseline),focus)
    return [{'id':f'{group}_{i:02d}','group':group,'messages':messages,'seed':seed+i} for i in range(count)]


def spec(model,harness,data,output,*,draws=1,seed=42,temperature=0):
    return {**identity(model),'harness':str(Path(harness).resolve()),'data':data,'output':str(output),
        'draws':draws,'seed':seed,'temperature':temperature}


def check_proposals(directory,data):
    result=directory/'runtime_checks.json'
    if not result.exists():
        with directory.with_suffix('.probe.log').open('a') as log:
            subprocess.run([sys.executable,str(ROOT/'whale_local/harness_probe.py'),'--input',str(directory/'generations.jsonl'),
                '--data',data['path'],'--output',str(result)],stdout=log,stderr=subprocess.STDOUT,check=True)
    return {r['id']:r for r in read(result)['results']}


def evaluate_proposals(pool,run,directory,model,cases):
    """Score all samples, including invalid/no-op samples (reward zero, R1)."""
    generations=rows(directory/'generations.jsonl')
    checks=check_proposals(directory,next(iter(cases.values()))['data'])
    cache=run/'outcome_cache';cache.mkdir(exist_ok=True)
    jobs={};mapped={}
    for g in generations:
        case=cases[g['group']]
        task_temperature=read(Path(case['baseline'])/'val.json')['metadata']['temperature']
        if not g['valid_source'] or not checks[g['id']]['passed']:
            mapped[g['id']]=('invalid',None);continue
        source=g['source'];current=Path(case['harness']).read_text()
        if source.strip()==current.strip():mapped[g['id']]=('no_op',None);continue
        harness=directory/(g['id']+'.py')
        assert harness.read_text()==source
        key=hashlib.sha256(json.dumps({'model':identity(model)['weight_digest'],'harness':sha(harness),
            'data':case['data']['sha256'],'seed':case.get('seed',42),'temperature':task_temperature},sort_keys=True).encode()).hexdigest()
        output=cache/key
        mapped[g['id']]=('valid',output)
        jobs[key]=spec(model,harness,case['data'],output,seed=case.get('seed',42),temperature=task_temperature)
    pool.evaluate_many(list(jobs.values()))
    scored=[]
    for g in generations:
        case=cases[g['group']];kind,output=mapped[g['id']]
        base=read(Path(case['baseline'])/'val.json')
        assert base['metadata']['weight_digest']==identity(model)['weight_digest']
        if kind=='valid':
            result=read(output/'val.json');reward=result['success_rate']-base['success_rate']
        else:result=None;reward=0.0
        scored.append({**g,'reward_delta':reward,'outcome_kind':kind,
            'reward_provenance':{'baseline':str(case['baseline']),'baseline_val_sha256':sha(Path(case['baseline'])/'val.json'),
                'candidate':str(output) if output else None,'candidate_val_sha256':sha(output/'val.json') if output else None,
                'fixed_solver':identity(model),'data':case['data'],'candidate_score':result['success_rate'] if result else None,
                'baseline_score':base['success_rate'],'all_batch_tasks_evaluated':kind=='valid'}})
    (directory/'scored.jsonl').write_text(''.join(json.dumps(g)+'\n' for g in scored))
    summary={'proposals':len(scored),'valid_complete':sum(g['outcome_kind']=='valid' for g in scored),
        'positive_delta':sum(g['reward_delta']>0 for g in scored),'negative_delta':sum(g['reward_delta']<0 for g in scored),
        'mean_reward_delta':sum(g['reward_delta'] for g in scored)/len(scored),
        'unique_valid_sources':len({g['source'] for g in scored if g['valid_source']}),
        'generation_sha256':sha(directory/'generations.jsonl'),'scored_sha256':sha(directory/'scored.jsonl')}
    write_json(directory/'score_summary.json',summary)
    return scored,summary


def probe_cases(run,config):
    return {'probe_finalize':{'data':config['data']['splits']['probe_finalize'],
        'harness':str(run/'base_editable.py'),'baseline':str(run/'baselines/probe_finalize')},
        'probe_concise':{'data':config['data']['splits']['probe_concise'],
        'harness':config['initial_harness'],'baseline':str(run/'baselines/probe_concise')}}


def probe_requests(run,config):
    result=[]
    for group,case in probe_cases(run,config).items():
        focus=group.removeprefix('probe_')
        result+=requests_for(group,Path(case['harness']).read_text(),case['baseline'],focus,4,
            config['editor_probe_seed']+(0 if focus=='finalize' else 100))
    return result


def source_snapshot(run,stage,extra=()):
    directory=run/'source_snapshots'/stage;directory.mkdir(parents=True,exist_ok=True)
    paths=[ROOT/'whale_local'/name for name in ['alternating_pool.py','alternating_common.py','harness_grpo.py',
        'worker.py','math_worker.py','math_runtime.py','math_remote.py','math_env.py','math_reward.py',
        'harness_code_contract.py','harness_probe.py','harness_focused_bootstrap.py','harness_bootstrap.py',
        'safe_gradient_norm.py','qwen35_padding.py',*extra]]
    manifest={}
    for path in paths:
        (directory/path.name).write_bytes(path.read_bytes());manifest[str(path)]=sha(path)
    write_json(directory/'manifest.json',manifest)

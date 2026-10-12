"""Corrected, baseline-first native WHALE alternation on fresh HotpotQA."""
import argparse
from contextlib import contextmanager
import datetime
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time
from types import SimpleNamespace

from pipeline import ROOT, api, sha, wait_ready, write_json
from search_pool_v2 import Pool
from search_runtime_v2 import BASE_HARNESS, DOMAIN
from meta_harness import meta_harness_search_r1 as mh


def read(path): return json.loads(Path(path).read_text())
def rows(path): return [json.loads(s) for s in Path(path).read_text().splitlines()]
def utc(): return datetime.datetime.now(datetime.timezone.utc).isoformat()
def status(run, stage, **extra):
    result={'stage':stage,'utc':utc(),**extra}; write_json(run/'STATUS.json',result)
    print(json.dumps(result),flush=True)


@contextmanager
def proposer(run,port):
    env=dict(os.environ,WHALE_PROPOSER_JOBS=str(run/'proposer_jobs'),WHALE_PROPOSER_PORT=str(port))
    with (run/'proposer.log').open('a') as log:
        process=subprocess.Popen([str(ROOT/'.venv-whale-portable/bin/python'),'-u',
            str(ROOT/'whale_local/proposer_api.py')],env=env,stdout=log,stderr=subprocess.STDOUT)
        try:
            url=f'http://127.0.0.1:{port}'; wait_ready(process,url); yield url
        finally:
            process.terminate()
            try:process.wait(timeout=20)
            except subprocess.TimeoutExpired:process.kill();process.wait()
            write_json(run/'proposer_stopped.json',{'pid':process.pid,'returncode':process.returncode})


def prepare(run,manifest):
    run.mkdir(parents=True,exist_ok=False)
    model=Path((ROOT/'runs/whale_setup/base_model_path.txt').read_text().strip())
    initial=run/'initial_harness.py';shutil.copy2(BASE_HARNESS,initial)
    predicted=[str(run/f'cycle_{n:02d}/solver_update/round_01/checkpoint') for n in [1,2]]
    config={'data':read(manifest),'solver_initial_checkpoint':str(model),'initial_harness':str(initial),
        'pool_gpus':['7','8','9']+[f'node1:{i}' for i in range(10)],'training_gpu':'6',
        'checkpoint_allowlist':[str(model)]+predicted,'cycles':2,'train_draws':4,'train_temperature':1.0,
        'evaluation_draws':1,'evaluation_seed':42,'evaluation_temperature':1.0,
        'learning_rate':1e-6,'parameter_dtype':'float32','assistant_token_budget':2048,'policy_max_tokens':768,
        'mh_iterations':2,'mh_proposals':2,'proposer_port':18642,
        'qualification_minimum_conditional_gain_pp':5.0,
        'algorithm':'WHALE native online RSFT then native full-code MH; W1-H1-W2-H2',
        'proposer':'active Codex conversation through local FastAPI queue',
        'search_space':'full SearchR1Env harness: prompts, search tool, retrieval queries/topk, context formatting, stop and recovery logic; MAX_TURNS <= 16; no reward/data/runtime edits',
        'test_unopened':True,
        'limits':['HotpotQA distractor BM25 instead of full Wikipedia E5/FAISS',
            'Fixed native exact match instead of paper LLM judge',
            '128 train questions, 4 rollout candidates, 48 harness development questions',
            '2048 assistant tokens, 768 per call, 4096 context; thinking disabled',
            'FP32 master/Adam and native RSFT minibatch 4, microbatch 1, LR 1e-6; paper uses different scale',
            'Proposer is active agent, not an unattended external API; its FLOPs are unmetered']}
    write_json(run/'config.json',config)
    write_json(run/'PROTOCOL.json',{'registered_utc':utc(),'config_sha256':sha(run/'config.json'),
        'method_selection':'task qualification uses monitor only; test untouched',
        'curve':'W0H0 -> W1H0 -> W1H1 -> W2H1 -> W2H2; raw scores, never best-so-far substitution',
        'conditional_checks':['W2H2-W0H2 >= 5pp','W2H2-W2H0 >= 5pp'],
        'internal_replacement':'only after external baseline qualification; not launched by this driver',
        'seed_policy':'single fixed seed 42; one evaluation per question; four RSFT training samples are training candidates'})
    snapshot=run/'source_snapshot';snapshot.mkdir()
    files=['search_baseline_v2.py','search_worker_v2.py','search_runtime_v2.py','search_pool_v2.py',
        'search_solver_train_v2.py','search_remote_v2.py','search_remote_server_v2.py',
        'worker.py','pipeline.py','qwen35_padding.py','safe_gradient_norm.py','math_remote.py','proposer_api.py']
    records=[]
    for name in files:
        source=ROOT/'whale_local'/name;shutil.copy2(source,snapshot/name)
        records.append({'path':str(source),'sha256':sha(source)})
    for rel in ['meta_harness/meta_harness_search_r1.py','meta_harness/benchmark.py',
                'environments/search_r1/env.py','environments/search_r1/search_r1_env.py','verl/workers/actor/dp_actor.py']:
        source=DOMAIN/rel;records.append({'path':str(source),'sha256':sha(source)})
    write_json(snapshot/'manifest.json',records);status(run,'prepared')


def spec(model,ident,harness,data,output,draws=1,temperature=1.0):
    return {'model':str(model),'weight_digest':ident['weight_digest'],'harness':str(harness),
        'data':data,'output':str(output),'draws':draws,'seed':42,'temperature':temperature}


def train(run,cycle,config,model,ident,harness):
    output=cycle/'solver_update';output.mkdir();(output/'calls').mkdir()
    source=cycle/'train_rollouts/trajectories.jsonl'
    for row in rows(source):
        for call in row['policy_calls']:write_json(output/'calls'/(call['call_id']+'.json'),call)
    target=output/'rollouts.jsonl';shutil.copy2(source,target)
    env=dict(os.environ,CUDA_VISIBLE_DEVICES=config['training_gpu'],WHALE_RUN_ROOT=str(output),
        WHALE_MODEL_ROOT=str(model),WHALE_SEARCH_CORPORA=config['data']['corpora']['path'],
        WHALE_LEARNING_RATE=str(config['learning_rate']),WHALE_PARAMETER_DTYPE='float32',WHALE_SAFE_GRAD_NORM='1',
        HF_HUB_OFFLINE='1',TOKENIZERS_PARALLELISM='false',PYTHONPATH=':'.join(str(p) for p in
        [ROOT/'whale_local',DOMAIN,ROOT/'WHALE/domains/chess_puzzles']))
    started=time.monotonic()
    with (output/'train.log').open('a') as log:
        subprocess.run([str(ROOT/'.venv-whale-portable/bin/python'),'-u',str(ROOT/'whale_local/search_solver_train_v2.py'),
            '--rollouts',str(target),'--data',config['data']['splits']['train']['path'],
            '--harness',str(harness),'--expected-digest',ident['weight_digest']],
            env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
    result=read(output/'complete.json');assert result['reload_verified']
    write_json(output/'TIMING.json',{'elapsed_seconds':time.monotonic()-started})
    return Path(result['identity']['weights']),result['identity']


def harness_search(run,cycle,config,pool,url,model,ident,incumbent):
    mh.RUNS_DIR=run/'mh';mh.BASELINE_HARNESS=incumbent
    mhconfig={'meta':{'proposer_model':'active-codex','proposer_effort':'high',
        'proposals_per_iter':config['mh_proposals'],'propose_timeout_s':3600},
        'task_profiles':{'mh_val':{}},'models':[{'model':ident['model']}],'seeds':[42],'eval':{'enable_thinking':False}}
    path=cycle/'mh_config.json';write_json(path,mhconfig)
    def sweep(_config,harnesses,logs_dir,**kwargs):
        specifications=[spec(model,ident,h,config['data']['splits']['mh_val'],
            logs_dir/'mh_val'/name/(ident['model'].lower()+'_seed42')) for name,_,h in harnesses]
        pool.evaluate_many(specifications)
        return [(name,True) for name,_,_ in harnesses]
    def propose(*,task_prompt,iteration,run_dir,timeout_seconds,**kwargs):
        names=re.findall(r'harnesses/(h\d+)/harness\.py',task_prompt)
        current=mh.get_accepted_harness(run_dir)
        payload={'task_prompt':task_prompt+'\nLocal fixed budget: 2048 assistant tokens per trajectory, 768 per call, 4096 context, max16 turns. Local retriever is BM25 over official distractor passages. Do not read monitor or test data/results. Use only train and mh_val failures. No hardcoded task answers, corpus reads, reward edits or policy calls outside the environment.',
            'run_dir':str(run_dir),'iteration':iteration,'accepted_harness_source':(run_dir/'harnesses'/current/'harness.py').read_text(),
            'frontier':read(run_dir/'logs/frontier_val.json')}
        job=api(url,'/jobs',payload);status(run,'awaiting_active_agent',cycle=cycle.name,iteration=iteration,job_id=job['job_id'])
        deadline=time.monotonic()+timeout_seconds
        while True:
            response=api(url,'/jobs/'+job['job_id'])
            if 'candidates' in response:break
            if time.monotonic()>deadline:raise TimeoutError('Active proposer response')
            time.sleep(1)
        assert [c['name'] for c in response['candidates']]==names
        metadata=[]
        for candidate in response['candidates']:
            directory=run_dir/'harnesses'/candidate['name'];directory.mkdir(exist_ok=True)
            compile(candidate['source'],str(directory/'harness.py'),'exec')
            (directory/'harness.py').write_text(candidate['source'])
            metadata.append({k:v for k,v in candidate.items() if k!='source'})
        write_json(run_dir/'pending_eval.json',{'candidates':metadata})
        return SimpleNamespace(exit_code=0,text='active Codex via local FastAPI',stderr='',show=lambda:None)
    mh.run_sweep=sweep;mh.propose_claude=propose
    saved=sys.argv
    try:
        sys.argv=['native_mh','--run-name',cycle.name,'--config',str(path),
            '--iterations',str(config['mh_iterations']),'--proposals-per-iter',str(config['mh_proposals'])]
        mh.main()
    finally:sys.argv=saved
    directory=run/'mh'/cycle.name;selected=mh.get_accepted_harness(directory)
    harness=directory/'harnesses'/selected/'harness.py'
    write_json(cycle/'mh_result.json',{'selected':selected,'harness':str(harness),'sha256':sha(harness),
        'weight_digest':ident['weight_digest'],'frontier':read(directory/'logs/frontier_val.json')})
    return harness


def execute(run):
    config=read(run/'config.json');assert sha(run/'config.json')==read(run/'PROTOCOL.json')['config_sha256']
    for item in read(run/'source_snapshot/manifest.json'):assert sha(item['path'])==item['sha256']
    started=time.monotonic();write_json(run/'START.json',{'utc':utc()})
    model=Path(config['solver_initial_checkpoint']);harness=Path(config['initial_harness'])
    try:
        with proposer(run,config['proposer_port']) as url,Pool(run/'pool',config) as pool:
            ident=read(pool.processes[0][0]/'ready.json');initial_ident=ident
            write_json(run/'initial_identity.json',ident)
            points=[{'label':'W0H0','model':str(model),'identity':ident,'harness':str(harness)}]
            for number in range(1,config['cycles']+1):
                cycle=run/f'cycle_{number:02d}';cycle.mkdir()
                status(run,'sampling_train',cycle=number)
                pool.evaluate_many([spec(model,ident,harness,config['data']['splits']['train'],
                    cycle/'train_rollouts',draws=config['train_draws'],temperature=config['train_temperature'])])
                status(run,'training_weights',cycle=number)
                model,ident=train(run,cycle,config,model,ident,harness)
                points.append({'label':f'W{number}H{number-1}','model':str(model),'identity':ident,'harness':str(harness)})
                status(run,'searching_harness',cycle=number)
                harness=harness_search(run,cycle,config,pool,url,model,ident,harness)
                points.append({'label':f'W{number}H{number}','model':str(model),'identity':ident,'harness':str(harness)})
                write_json(run/'CURVE_STATES.json',points)
            write_json(run/'FROZEN_BASELINE.json',{'utc':utc(),'points':points,'test_opened':False})
            status(run,'development_qualification')
            checks=points+[{'label':'W0H2','model':config['solver_initial_checkpoint'],'identity':initial_ident,'harness':str(harness)},
                {'label':'W2H0','model':str(model),'identity':ident,'harness':config['initial_harness']}]
            results=pool.evaluate_many([spec(Path(p['model']),p['identity'],Path(p['harness']),
                config['data']['splits']['monitor'],run/'monitor'/p['label']) for p in checks])
            values=dict(zip([p['label'] for p in checks],results))
            wg=100*(values['W2H2']['success_rate']-values['W0H2']['success_rate'])
            hg=100*(values['W2H2']['success_rate']-values['W2H0']['success_rate'])
            write_json(run/'QUALIFICATION.json',{'split':'monitor/development, not test','scores':values,
                'conditional_weight_gain_pp':wg,'conditional_harness_gain_pp':hg,
                'qualified':wg>=5 and hg>=5,'test_opened':False})
        status(run,'complete',elapsed_seconds=time.monotonic()-started)
    except BaseException as exc:
        status(run,'failed',error=repr(exc));raise


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['prepare','run'])
    parser.add_argument('--run',type=Path,required=True);parser.add_argument('--manifest',type=Path)
    args=parser.parse_args()
    if args.action=='prepare':prepare(args.run.resolve(),args.manifest)
    else:execute(args.run.resolve())

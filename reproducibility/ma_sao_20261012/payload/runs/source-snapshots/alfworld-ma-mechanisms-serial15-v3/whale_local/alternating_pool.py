"""Persistent frozen solver replicas; each process owns its sandbox URL.

Jobs contain immutable model/code/data identities and disjoint rollout schedules.
Workers never train. Remote evidence is collected by the existing SSH supervisor.
"""
import argparse
import asyncio
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from urllib.error import HTTPError

from math_runtime import evaluate, service
from pipeline import ROOT, api, sha, write_json


def read(path):
    return json.loads(Path(path).read_text())


def rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines()]


def worker(args):
    run=args.worker.resolve();config=read(run/'worker_config.json')
    os.environ['WHALE_CHECKPOINT_ALLOWLIST']=json.dumps(config.get('checkpoint_allowlist',[]))
    with service(run,Path(config['model']),config['gpu'],config['port'],config['data'],parameter_dtype='float32') as (url,identity):
        write_json(run/'ready.json',identity)
        while not (run/'stop').exists():
            pending=sorted((run/'jobs').glob('*.json'))
            pending=[p for p in pending if not p.with_suffix('.done').exists()]
            if not pending:
                time.sleep(0.2);continue
            path=pending[0];job=read(path)
            try:
                if identity['weight_digest']!=job['weight_digest']:
                    if config['gpu'].startswith('node1:'):
                        from math_remote import ensure_model
                        ensure_model(Path(job['model']))
                    identity=api(url,'/checkpoint/load',{'checkpoint':job['model']},timeout=300)
                assert identity['weight_digest']==job['weight_digest']
                assert sha(job['harness'])==job['harness_sha256']
                output=Path(job['output'])
                if not (output/'val.json').exists():
                    asyncio.run(evaluate(base_url=url,identity=identity,harness=Path(job['harness']),
                        data_entry=job['data'],output=output,temperature=job['temperature'],seed=job['seed'],
                        schedule=job['schedule'],resume=True))
                assert api(url,'/health')['weight_digest']==job['weight_digest']
                write_json(path.with_suffix('.done'),{'passed':True,'val_sha256':sha(output/'val.json')})
            except BaseException as exc:
                detail={'error':repr(exc)}
                if isinstance(exc,HTTPError):detail['response_body']=exc.read().decode(errors='replace')[:4000]
                write_json(path.with_suffix('.error'),detail)
                raise


class Pool:
    def __init__(self,run,config):
        self.run=Path(run);self.config=config;self.processes=[];self.counter=0

    def __enter__(self):
        initial=Path(self.config['solver_initial_checkpoint'])
        for i,gpu in enumerate(self.config['pool_gpus']):
            directory=self.run/'replicas'/f'slot_{i:02d}'
            directory.mkdir(parents=True,exist_ok=True);(directory/'jobs').mkdir(exist_ok=True)
            for name in ['stop','ready.json']:
                (directory/name).unlink(missing_ok=True)
            write_json(directory/'worker_config.json',{'model':str(initial),'gpu':gpu,
                'port':20000+i,'data':self.config['data'],
                'checkpoint_allowlist':self.config.get('checkpoint_allowlist',[])})
            log=(directory/'coordinator.log').open('a')
            p=subprocess.Popen([sys.executable,'-u',str(Path(__file__).resolve()),'--worker',str(directory)],
                env=dict(os.environ,CUDA_VISIBLE_DEVICES=''),stdout=log,stderr=subprocess.STDOUT)
            self.processes.append((directory,p,log))
        try:
            deadline=time.monotonic()+600
            while not all((d/'ready.json').exists() for d,_,_ in self.processes):
                self.check()
                if time.monotonic()>deadline:raise TimeoutError('Frozen replica startup')
                time.sleep(1)
            return self
        except BaseException:
            self.__exit__(*sys.exc_info());raise

    def check(self):
        for d,p,_ in self.processes:
            if p.poll() is not None:raise RuntimeError(f'Replica exited: {d}, code={p.returncode}')
            errors=list((d/'jobs').glob('*.error'))
            if errors:raise RuntimeError(f'Replica job failed: {errors[0]}: {read(errors[0])}')

    def evaluate_many(self,specs):
        """Evaluate each spec once, partitioning its plan into short worker jobs."""
        work=[]
        for spec in specs:
            output=Path(spec['output']);plan=spec.get('schedule') or [
                [draw,i] for draw in range(spec.get('draws',1)) for i in range(spec['data']['count'])]
            if (output/'val.json').exists():
                old=read(output/'val.json')['metadata']
                assert old['weight_digest']==spec['weight_digest'] and old['harness_sha256']==sha(spec['harness'])
                assert old['sampling_schedule']==plan and old['seed']==spec['seed'] and old['temperature']==spec['temperature']
                continue
            chunks=[plan[i:i+2] for i in range(0,len(plan),2)]
            assignments=[]
            for i,chunk in enumerate(chunks):
                job=dict(spec,schedule=chunk,harness_sha256=sha(spec['harness']),
                    output=str(output/'shards'/f'{i:04d}'))
                work.append((job,assignments))
            spec['_assignments']=assignments;spec['_plan']=plan
        available=list(range(len(self.processes)));running={}
        while work or running:
            self.check()
            while available and work:
                slot=available.pop(0);job,assignments=work.pop(0)
                directory=self.processes[slot][0];self.counter+=1
                path=directory/'jobs'/f'{time.time_ns()}_{self.counter:06d}.json'
                write_json(path,job);assignments.append(str(path));running[slot]=path
            for slot,path in list(running.items()):
                if path.with_suffix('.done').exists():
                    del running[slot];available.append(slot)
            if running:time.sleep(0.5)
        for spec in specs:
            if '_assignments' not in spec:continue
            output=Path(spec['output']);result=[]
            for path in spec.pop('_assignments'):
                job=read(path);part=rows(Path(job['output'])/'trajectories.jsonl')
                for r in part:r['physical_worker_run']=str(Path(path).parent.parent)
                result.extend(part)
            by_key={(r['draw'],r['example_index']):r for r in result};plan=spec.pop('_plan')
            assert len(by_key)==len(result)==len(plan)
            result=[by_key[tuple(k)] for k in plan]
            for r in result:
                assert r['sampling_identity']['weight_digest']==spec['weight_digest']
                assert r['harness_sha256']==sha(spec['harness'])
            output.mkdir(parents=True,exist_ok=True)
            (output/'trajectories.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in result))
            write_json(output/'val.json',{'num_examples':len(result),'solved_examples':int(sum(r['reward'] for r in result)),
                'success_rate':sum(r['reward'] for r in result)/len(result),
                'mean_turn_count':sum(len(r['policy_calls']) for r in result)/len(result),
                'metadata':{'weight_digest':spec['weight_digest'],'harness_sha256':sha(spec['harness']),
                    'dataset_path':spec['data']['path'],'dataset_sha256':spec['data']['sha256'],
                    'sampling_schedule':plan,'temperature':spec['temperature'],'seed':spec['seed'],
                    'assistant_token_budget':2048,'policy_max_tokens':1024}})
        return [read(Path(spec['output'])/'val.json') for spec in specs]

    def __exit__(self,*exc):
        for d,p,_ in self.processes:
            if p.poll() is None:(d/'stop').touch()
        for d,p,log in self.processes:
            try:p.wait(timeout=120)
            except subprocess.TimeoutExpired:
                p.send_signal(signal.SIGINT)
                try:p.wait(timeout=60)
                except subprocess.TimeoutExpired:p.terminate();p.wait(timeout=45)
            log.close()
        write_json(self.run/'pool_stopped.json',{'workers':[{'directory':str(d),'returncode':p.returncode}
            for d,p,_ in self.processes]})


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--worker',type=Path,required=True);worker(p.parse_args())

"""Register a two-cycle trial and first measure the two existing checkpoints."""
from concurrent.futures import ThreadPoolExecutor
import argparse
import datetime
import json
from pathlib import Path

import pandas as pd

from alternating_common import *
from alternating_pool import Pool


def prepare(run):
    if (run/'config.json').exists():return read(run/'config.json')
    run.mkdir(parents=True)
    prior=ROOT/'runs/whale_dual2b_focused_20260916';p=read(prior/'config.json');models=read(prior/'CURRENT_MODELS.json')
    directory=run/'data';directory.mkdir()
    mh=pd.read_parquet(p['data']['splits']['mh_val']['path'])
    mh=mh.iloc[sorted(range(len(mh)),key=lambda i:mh.iloc[i]['extra_info']['task_id'])]
    splits={}
    for name,start,end in [('reward_0',0,8),('reward_1',8,16),('probe_finalize',16,20),('probe_concise',20,24),('selection',24,48)]:
        path=directory/(name+'.parquet');mh.iloc[start:end].to_parquet(path,index=False)
        splits[name]={'path':str(path),'sha256':sha(path),'count':end-start}
    splits['train']=p['data']['splits']['train'];splits['monitor']=p['data']['splits']['test']
    config={'cycles':2,'editor_initial_checkpoint':models['harness_generator']['checkpoint'],
        'editor_comparison_checkpoint':p['editor_initial_checkpoint'],
        'solver_initial_checkpoint':models['task_solver']['checkpoint'],
        'initial_harness':models['execution_harness']['path'],'data':{'splits':splits},
        'pool_gpus':['8','9',*[f'node1:{i}' for i in range(10)]],
        'editor_probe_seed':71337,'editor_training_group_size':8,'editor_training_groups':2,
        'editor_grpo_epochs':4,'editor_grpo_lr':1e-6,'solver_rsft_lr':3e-7,
        'solver_draws':1,'monitor_draws':2,'monitor_temperature':1.0,'monitor_seed':42,
        'initial_reference':str(prior),'initial_verification_sha256':sha(prior/'verification.json'),
        'selection_candidates_per_cycle':2,'teacher_runtime_calls':0,
        'zero_variance_policy':'Record no update, do not fabricate weight learning; continue fixed experiment budget.',
        'model_update_policy':'Carry each numerically verified trained solver; select harness by development results including incumbent.',
        'editor_policy_probe':'Two fixed four-task packets, four samples each, same fixed W2 solver and seeds for every editor checkpoint.',
        'harness_training':'Two groups of eight on-policy code proposals. Reward is same-batch solver success delta. Invalid/no-op zero. GRPO + TIS.',
        'limits':['same MATH scenario; previous test32 reused only as monitoring, not fresh independent confirmation',
            'probe/selection/reward are disjoint within this run but came from previously used MH development tasks',
            'small local implementation of R1 objective, not the original Relax/Megatron distributed runtime',
            'editor starts from bounded SFT; focus remains supplied from failure telemetry']}
    write_json(run/'config.json',config);(run/'base_editable.py').write_text(START)
    write_json(run/'PROTOCOL.json',{'registered_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'config_sha256':sha(run/'config.json'),'run_on_same_task':True,'cycles':2,'before_any_new_rollout':True,
        'harness_r1_paper':'https://arxiv.org/html/2608.02276v1#S3.SS2',
        'harness_r1_local_commit':read(ROOT/'deliverables/harness_r1_training_pack/VERSION.json')['sources']['harness_r1_commit']
            if 'sources' in read(ROOT/'deliverables/harness_r1_training_pack/VERSION.json') else '411bb5489ada85e6118897039eab84b4e4aaa4f0',
        'monitor_only_never_reward_or_selection':True})
    return config


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);args=p.parse_args();run=args.run.resolve()
    config=prepare(run);source_snapshot(run,'initial',['alternating_initial.py'])
    write_json(run/'STATUS.json',{'stage':'initial_checkpoint_probes_running'})
    cases=probe_cases(run,config);model=config['solver_initial_checkpoint']
    with Pool(run,config) as pool:
        pool.evaluate_many([spec(model,case['harness'],case['data'],case['baseline']) for case in cases.values()])
        requests=probe_requests(run,config)
        checkpoints=[('E0',config['editor_comparison_checkpoint'],'6'),('E1',config['editor_initial_checkpoint'],'7')]
        with ThreadPoolExecutor(max_workers=2) as executor:
            futures=[executor.submit(run_editor,run,checkpoint,run/'editor_probes'/name,requests,gpu=gpu)
                for name,checkpoint,gpu in checkpoints]
            for future in futures:future.result()
        scores={}
        for name,_,_ in checkpoints:
            _,scores[name]=evaluate_proposals(pool,run,run/'editor_probes'/name,model,cases)
        # Exact cached first two draws: fixed in protocol before inspecting scores.
        prior=Path(config['initial_reference']);monitor={}
        for cell,label in [('Wf_H0','W2_H0'),('Wf_Hf','W2_Hcurrent'),('W1_Hf','W1_Hcurrent')]:
            original=prior/'test'/cell/'trajectories.jsonl';data=rows(original)
            data=[r for r in data if r['draw']<config['monitor_draws']]
            assert len(data)==64
            monitor[label]={'successful_draws':sum(r['reward'] for r in data),'draws':len(data),
                'source':str(original),'source_sha256':sha(original),'reused_draws':[0,1]}
        write_json(run/'initial_checkpoint_results.json',{'editor_probe':scores,'solver_monitor':monitor,
            'editor_comparison_same_prompts_seeds_solver_and_tasks':True,'config_sha256':sha(run/'config.json')})
    write_json(run/'INITIAL_COMPLETE.json',{'passed_execution':True,'results_sha256':sha(run/'initial_checkpoint_results.json')})
    write_json(run/'STATUS.json',{'stage':'initial_probes_complete_pending_audit'})
    print(json.dumps(read(run/'initial_checkpoint_results.json'),indent=2),flush=True)


if __name__=='__main__':main()

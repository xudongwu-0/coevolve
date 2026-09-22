"""Two genuine E-GRPO -> regenerate H -> W-RSFT -> new-failures cycles."""
import argparse
import datetime
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

from alternating_common import *
from alternating_pool import Pool
from harness_focused_bootstrap import choose_focus
from math_reward import binary_reward
from meta_harness.benchmark import compute_pareto_frontier
from meta_harness.meta_harness_retool import pick_accepted_from_frontier


def explored_proposals(checkpoint,directory,requests,temperature,gpu='6'):
    if (directory/'complete.json').exists():return
    directory.mkdir(parents=True,exist_ok=True)
    data=directory/'requests.jsonl';data.write_text(''.join(json.dumps(r)+'\n' for r in requests))
    env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),WHALE_RUN_ROOT=str(directory),WHALE_MODEL_ROOT=str(checkpoint),
        WHALE_PARAMETER_DTYPE='float32',WHALE_LEARNING_RATE='1e-6',HF_HUB_OFFLINE='1',TOKENIZERS_PARALLELISM='false',
        PYTHONPATH=':'.join(str(ROOT/p) for p in ['whale_local','WHALE/domains/math_reasoning','WHALE/domains/chess_puzzles']))
    with directory.with_suffix('.log').open('a') as log:
        subprocess.run([str(ROOT/'.venv-whale-portable/bin/python'),'-u',str(ROOT/'whale_local/harness_grpo.py'),
            'generate','--data',str(data),'--temperature',str(temperature)],env=env,stdout=log,stderr=subprocess.STDOUT,check=True)


def has_reward_variance(scored):
    return any(len({r['reward_delta'] for r in scored if r['group']==g})>1 for g in {r['group'] for r in scored})


def status(run,stage,**details):
    write_json(run/'STATUS.json',{'stage':stage,'updated_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),**details})
    print(stage,details,flush=True)


def train_solver(run,cycle,config,model,harness,rollout_dir):
    output=cycle/'solver_update';output.mkdir(exist_ok=True)
    if (output/'complete.json').exists():return Path(read(output/'complete.json')['identity']['weights'])
    samples=rows(rollout_dir/'trajectories.jsonl')
    # These are real received API records; compare them to original server files
    # once the pool closes and copies all remote evidence back.
    (output/'calls').mkdir(exist_ok=True)
    for row in samples:
        for call in row['policy_calls']:write_json(output/'calls'/(call['call_id']+'.json'),call)
    target=output/'rollouts.jsonl';shutil.copy2(rollout_dir/'trajectories.jsonl',target)
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='7',WHALE_RUN_ROOT=str(output),WHALE_MODEL_ROOT=str(model),
        WHALE_PARAMETER_DTYPE='float32',WHALE_LEARNING_RATE=str(config['solver_rsft_lr']),WHALE_SAFE_GRAD_NORM='1',
        HF_HUB_OFFLINE='1',TOKENIZERS_PARALLELISM='false',
        PYTHONPATH=':'.join(str(ROOT/p) for p in ['whale_local','WHALE/domains/math_reasoning','WHALE/domains/chess_puzzles']))
    command=[str(ROOT/'.venv-whale-portable/bin/python'),'-u',str(ROOT/'whale_local/alternating_solver_train.py'),
        '--rollouts',str(target),'--data',config['data']['splits']['train']['path'],'--harness',str(harness),
        '--expected-digest',identity(model)['weight_digest']]
    with (output/'train.log').open('a') as log:subprocess.run(command,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
    return Path(read(output/'complete.json')['identity']['weights'])


def monitor(pool,run,config,model,harness,label,curve):
    key=(identity(model)['weight_digest'],sha(harness))
    same=next((point for point in curve if (point['weight_digest'],point['harness_sha256'])==key),None)
    if same:
        point={**same,'label':label,'reuses_identical_model_harness_monitor':same['label']}
    else:
        output=run/'monitor'/label
        value=pool.evaluate_many([spec(model,harness,config['data']['splits']['monitor'],output,
            draws=config['monitor_draws'],temperature=config['monitor_temperature'],seed=config['monitor_seed'])])[0]
        point={'label':label,'successful_draws':value['solved_examples'],'draws':value['num_examples'],
            'weight_digest':key[0],'harness_sha256':key[1],'trajectories':str(output/'trajectories.jsonl')}
    curve.append(point);write_json(run/'monitor_curve.json',curve)
    return point


def select_harness(pool,run,cycle,config,model,incumbent,post):
    unique={}
    for proposal in sorted(post,key=lambda p:(-p['reward_delta'],p['id'])):
        if proposal['outcome_kind']!='valid':continue
        digest=hashlib.sha256(proposal['source'].encode()).hexdigest()
        if digest not in unique:unique[digest]=proposal
    shortlist=list(unique.values())[:config['selection_candidates_per_cycle']]
    directory=cycle/'selection';directory.mkdir(exist_ok=True)
    choices={'incumbent':Path(incumbent)}
    for p in shortlist:choices[p['id']]=cycle/'editor_post_proposals'/(p['id']+'.py')
    for name,path in choices.items():
        destination=directory/'harnesses'/name/'harness.py';destination.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(path,destination)
    results=pool.evaluate_many([spec(model,path,config['data']['splits']['selection'],directory/'results'/name,
        temperature=config.get('harness_selection_temperature',0))
        for name,path in choices.items()])
    values=dict(zip(choices,results));points=[(name,r['success_rate'],r['mean_turn_count']) for name,r in values.items()]
    frontier={'_pareto':[{'harness':n,'avg_success_rate':s,'avg_mean_turn_count':t}
        for n,s,t in compute_pareto_frontier(points)]}
    _,accepted=pick_accepted_from_frontier(frontier,directory);assert accepted
    selected=directory/'harnesses'/accepted/'harness.py'
    write_json(directory/'decision.json',{'accepted':accepted,'results':values,'frontier':frontier,
        'all_candidates_from_updated_editor':True,'shortlist_ids':[p['id'] for p in shortlist],
        'monitor_not_used':True,'selected_sha256':sha(selected)})
    return selected


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--wait-initial',action='store_true')
    args=p.parse_args();run=args.run.resolve()
    if args.wait_initial:
        while not (run/'INITIAL_COMPLETE.json').exists():time.sleep(2)
    assert read(run/'INITIAL_COMPLETE.json')['passed_execution']
    assert read(run/'initial_verification.json')['passed_audit']
    config=read(run/'config.json')
    protocol=run/'CYCLES_PROTOCOL.json'
    if not protocol.exists():
        write_json(protocol,{'registered_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
            'config_sha256':sha(run/'config.json'),'cycles':2,'post_update_proposals_per_group':4,
            'regenerate_with_updated_editor_before_selecting_execution_harness':True,
            'editor_grpo_epochs':4,'solver_learning_rate':config['solver_rsft_lr'],
            'solver_initial_checkpoint':config['solver_initial_checkpoint'],'editor_initial_checkpoint':config['editor_initial_checkpoint'],
            'monitor_used_for_decisions':False,'carry_numerically_verified_updates_even_if_monitor_drops':True,
            'post_update_editor_probe_uses_fixed_initial_solver':True})
        registered=read(protocol)
        registered.update(training_proposal_temperature=config.get('training_proposal_temperature',1.2),
            zero_variance_fallback_temperature=config.get('fallback_proposal_temperature',1.5),
            max_training_bundles_per_cycle=2,policy_probe_temperature=0.7,
            exploration_reason='Initial E0/E1 produced identical repeated code at temperature 0.7; choose exploration before any cycle rewards.')
        write_json(protocol,registered)
        source_snapshot(run,'cycles',['alternating_cycles.py','alternating_solver_train.py'])
    model=Path(config['solver_initial_checkpoint']);editor=Path(config['editor_initial_checkpoint']);harness=Path(config['initial_harness'])
    cached=read(run/'initial_checkpoint_results.json')['solver_monitor']['W2_Hcurrent']
    curve=[{'label':'initial','successful_draws':cached['successful_draws'],'draws':cached['draws'],
        'weight_digest':identity(model)['weight_digest'],'harness_sha256':sha(harness),'trajectories':cached['source'],
        'reused_draws':[0,1]}]
    write_json(run/'monitor_curve.json',curve)
    for round_number in range(1,config['cycles']+1):
        cycle=run/f'cycle_{round_number:02d}';cycle.mkdir(exist_ok=True)
        if (cycle/'complete.json').exists():
            done=read(cycle/'complete.json');model=Path(done['solver']);editor=Path(done['editor']);harness=Path(done['harness'])
            curve+=done['monitor_points'];continue
        cycle_config={**config,'solver_initial_checkpoint':str(model),
            'checkpoint_allowlist':[config['solver_initial_checkpoint'],
                *(str(run/f'cycle_{i:02d}/solver_update/round_01/checkpoint') for i in range(1,config['cycles']+1))]}
        write_json(cycle/'START.json',{'solver':identity(model),'editor':identity(editor),'harness':str(harness),'harness_sha256':sha(harness)})
        pool_directory=cycle/'pool'
        if pool_directory.exists():
            # A stopped partial cycle keeps all jobs, receipts and errors.
            assert (pool_directory/'pool_stopped.json').exists(),'Existing pool must be confirmed stopped before resume'
            attempt=1
            while (cycle/f'pool_resume_{attempt:02d}').exists():attempt+=1
            pool_directory=cycle/f'pool_resume_{attempt:02d}'
        with Pool(pool_directory,cycle_config) as pool:
            if round_number==1 and config.get('composition_bootstrap'):
                status(run,'composition_editor_fixed_probe',cycle=round_number)
                probe=run/'editor_probes/E1c';run_editor(run,editor,probe,probe_requests(run,config))
                evaluate_proposals(pool,run,probe,config['solver_initial_checkpoint'],probe_cases(run,config))
            status(run,'cycle_reward_baselines',cycle=round_number)
            cases={}
            for i in range(2):
                name=f'reward_{i}';cases[name]={'data':config['data']['splits'][name],
                    'harness':str(harness),'baseline':str(cycle/'baselines'/name),'seed':142+round_number}
            pool.evaluate_many([spec(model,harness,c['data'],c['baseline'],seed=c['seed'],
                temperature=config.get('harness_reward_temperature',0)) for c in cases.values()])
            requests=[]
            for i,(group,case) in enumerate(cases.items()):
                observed=rows(Path(case['baseline'])/'trajectories.jsonl')
                focus=config.get('editor_request_mode') or choose_focus(observed,harness.read_text(),i)
                requests+=requests_for(group,harness.read_text(),case['baseline'],focus,8,81200+round_number*1000+i*100)
            status(run,'cycle_editor_on_policy_proposals',cycle=round_number)
            temperature=config.get('training_proposal_temperature',1.2)
            before=cycle/'editor_pre_proposals';explored_proposals(editor,before,requests,temperature)
            scored,pre_summary=evaluate_proposals(pool,run,before,model,cases)
            if not has_reward_variance(scored):
                status(run,'cycle_zero_variance_exploration_fallback',cycle=round_number)
                temperature=config.get('fallback_proposal_temperature',1.5)
                requests=[dict(r,seed=r['seed']+500000) for r in requests]
                before=cycle/'editor_pre_fallback';explored_proposals(editor,before,requests,temperature)
                scored,pre_summary=evaluate_proposals(pool,run,before,model,cases)
            write_json(cycle/'TRAIN_SOURCE.json',{'directory':str(before),'temperature':temperature,
                'scored_sha256':sha(before/'scored.jsonl'),'has_group_variance':has_reward_variance(scored)})
            status(run,'cycle_editor_grpo',cycle=round_number)
            trained=run_editor(run,editor,cycle/'editor_grpo',scored,mode='train',epochs=config['editor_grpo_epochs'])
            prior_editor=editor
            if trained['updated']:editor=Path(trained['identity']['weights'])
            # Updated E generates the actual proposals used by the next W update.
            status(run,'cycle_updated_editor_regeneration',cycle=round_number,editor_updated=trained['updated'])
            post_requests=[r for r in requests if int(r['id'].rsplit('_',1)[1])<4]
            after=cycle/'editor_post_proposals';explored_proposals(editor,after,post_requests,temperature)
            post,post_summary=evaluate_proposals(pool,run,after,model,cases)
            probe=run/'editor_probes'/f'E{round_number+1}';run_editor(run,editor,probe,probe_requests(run,config))
            _,probe_summary=evaluate_proposals(pool,run,probe,config['solver_initial_checkpoint'],probe_cases(run,config))
            status(run,'cycle_harness_selection',cycle=round_number)
            harness=select_harness(pool,run,cycle,config,model,harness,post)
            hpoint=monitor(pool,run,config,model,harness,f'cycle_{round_number:02d}_after_harness',curve)
            status(run,'cycle_solver_sampling',cycle=round_number)
            rollout_dir=cycle/'solver_rollouts'
            pool.evaluate_many([spec(model,harness,config['data']['splits']['train'],rollout_dir,
                draws=config['solver_draws'],seed=431+round_number,temperature=0.8)])
            status(run,'cycle_solver_rsft',cycle=round_number)
            prior_solver=model;model=train_solver(run,cycle,config,model,harness,rollout_dir)
            status(run,'cycle_monitor_new_solver',cycle=round_number)
            wpoint=monitor(pool,run,config,model,harness,f'cycle_{round_number:02d}_after_weights',curve)
        write_json(cycle/'complete.json',{'solver':str(model),'editor':str(editor),'harness':str(harness),
            'prior_solver':str(prior_solver),'prior_editor':str(prior_editor),'editor_updated':trained['updated'],
            'editor_pre_summary':pre_summary,'editor_post_summary':post_summary,'editor_probe_summary':probe_summary,
            'monitor_points':[hpoint,wpoint],'solver_digest':identity(model)['weight_digest'],'editor_digest':identity(editor)['weight_digest']})
    write_json(run/'monitor_curve.json',curve)
    write_json(run/'FINAL_MODELS.json',{'solver':identity(model),'editor':identity(editor),'harness':str(harness),'harness_sha256':sha(harness)})
    status(run,'alternation_complete_pending_audit')


if __name__=='__main__':main()

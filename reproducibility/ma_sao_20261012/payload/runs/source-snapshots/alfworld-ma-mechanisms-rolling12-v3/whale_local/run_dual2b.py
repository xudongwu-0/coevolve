"""Pilot: internal editor MH -> reward-backed editor SFT -> new solver RSFT.

The cached native W1 is an explicit warm start. All new harness proposals come
from the local 2B checkpoint; no active-agent queue or source rewriting.
"""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from harness_bootstrap import START, messages
from math_runtime import BASE_HARNESS, service
from math_sampling import sample_parallel
from math_evaluation import evaluate_four_cells
from meta_harness import meta_harness_retool as mh
from meta_harness.benchmark import compute_pareto_frontier
from pipeline import ROOT, api, sha, write_json
from prepare_dual2b import feedback
from math_reward import binary_reward

def read(p): return json.loads(Path(p).read_text())
def rows(p): return [json.loads(s) for s in Path(p).read_text().splitlines()]

def editor(run, checkpoint, directory, data, mode='generate', epochs=1):
    if (directory / 'complete.json').exists(): return
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='6', WHALE_RUN_ROOT=str(directory),
        WHALE_MODEL_ROOT=str(checkpoint), WHALE_PARAMETER_DTYPE='float32', WHALE_LEARNING_RATE='1e-6',
        PYTHONPATH=':'.join(str(ROOT/p) for p in ['whale_local','WHALE/domains/math_reasoning','WHALE/domains/chess_puzzles']),
        TOKENIZERS_PARALLELISM='false', HF_HUB_OFFLINE='1')
    seed=271+int(hashlib.sha256(directory.name.encode()).hexdigest()[:8],16)%100000
    command = [str(ROOT/'.venv-whale-portable/bin/python'), '-u', str(ROOT/'whale_local/harness_learner.py'), mode,
        '--data', str(data), '--epochs', str(epochs), '--learning-rate', '1e-6', '--temperature', '0.7', '--seed', str(seed)]
    with directory.with_suffix('.log').open('a') as log:
        subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)

def runtime_probe(run, generated, out, data):
    command = [sys.executable, str(ROOT/'whale_local/harness_probe.py'), '--input', str(generated),
        '--data', data['path'], '--output', str(out)]
    with out.with_suffix('.log').open('a') as log:
        subprocess.run(command, env=dict(os.environ, CUDA_VISIBLE_DEVICES=''),
            stdout=log, stderr=subprocess.STDOUT, check=True, timeout=180)
    return {r['id']: r for r in read(out)['results']}

def main():
    p=argparse.ArgumentParser(); p.add_argument('--run',type=Path,required=True)
    p.add_argument('--new-solver-round',action='store_true'); args=p.parse_args()
    run=args.run.resolve(); config=read(run/'config.json'); reference=Path(config['solver_prefix_reference'])
    baseline=reference/'mh/round_01/logs/mh_val/h0'
    source_val=next(baseline.rglob('val.json')); source_rows=source_val.parent/'trajectories.jsonl'
    solver=reference/'round_01/checkpoint'
    solver_training=read(solver/'training_state.json')
    assert read(reference/'verification.json')['passed_audit']
    assert read(source_val)['metadata']['weight_digest']==solver_training['weight_digest']
    assert read(source_val)['metadata']['harness_sha256']==sha(BASE_HARNESS)
    assert read(source_val)['metadata']['dataset_path']==config['data']['splits']['mh_val']['path']
    mhdir=run/'internal_mh'; (mhdir/'harnesses/h0').mkdir(parents=True,exist_ok=True)
    shutil.copy2(BASE_HARNESS,mhdir/'harnesses/h0/harness.py')
    results={'h0':read(source_val)}; trajectories={'h0':str(source_rows)}
    checkpoint=Path(config['editor_initial_checkpoint']); accepted='h0'; positive=[]; proposals=[]
    def status(stage, **kwargs):
        write_json(run/'STATUS.json',dict(stage=stage,updated_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),**kwargs))
        print(stage,kwargs,flush=True)
    status('starting',editor_checkpoint=str(checkpoint),solver_checkpoint=str(solver))
    with service(run,solver,'8',30081,config['data'],parameter_dtype='float32') as (base_url,identity):
        write_json(run/'solver_initial_identity.json',identity)
        warm_solver_identity=identity
        for iteration in range(1,3):
            status('internal_proposals',iteration=iteration,accepted=accepted)
            phase=run/f'editor_proposals_{iteration:02d}'; request_path=run/f'editor_requests_{iteration:02d}.jsonl'
            source=START if accepted=='h0' else (mhdir/f'harnesses/{accepted}/harness.py').read_text()
            failures=rows(trajectories[accepted])
            requests=[]
            for i in range(2):
                if config.get('editor_focus_mode')=='telemetry_conditioned':
                    from harness_focused_bootstrap import request,choose_focus
                    focus=choose_focus(failures,source,i)
                    prompt=request(source,feedback(failures),focus)
                else:
                    focus=None;prompt=messages(source,feedback(failures))
                requests.append({'id':f'h{2*(iteration-1)+i+1}','messages':prompt,'focus':focus})
            request_path.write_text(''.join(json.dumps(x)+'\n' for x in requests))
            editor(run,checkpoint,phase,request_path)
            generated=rows(phase/'generations.jsonl')
            checks=runtime_probe(run,phase/'generations.jsonl',phase/'runtime_checks.json',config['data']['splits']['mh_val'])
            incumbent=accepted
            for candidate in generated:
                name=candidate['id']; proposals.append(candidate)
                if not candidate['valid_source'] or not checks[name]['passed']:
                    status('candidate_rejected_contract',candidate=name); continue
                duplicates=[n for n in results if (mhdir/f'harnesses/{n}/harness.py').read_text()==candidate['source']]
                if duplicates:
                    write_json(phase/(name+'_duplicate.json'),{'candidate':name,'duplicate_of':duplicates[0],
                        'evaluation_calls':0,'counts_as_proposal_attempt':True})
                    continue
                target=mhdir/f'harnesses/{name}/harness.py';target.parent.mkdir(parents=True,exist_ok=True)
                target.write_text(candidate['source'])
                phase_eval=run/'mh_sampling'/name;phase_eval.mkdir(parents=True,exist_ok=True)
                output=run/'mh_results'/name
                status('candidate_evaluation',candidate=name)
                if not (output/'val.json').exists():
                    sample_parallel(run=run,phase=phase_eval,output=output,base_url=base_url,identity=identity,
                        harness=target,data_entry=config['data']['splits']['mh_val'],rollouts=1,seed=42,
                        temperature=0,split='mh_val',resume=True,port_base=22000+int(name[1:])*30)
                results[name]=read(output/'val.json');trajectories[name]=str(output/'trajectories.jsonl')
                if results[name]['solved_examples']>results[incumbent]['solved_examples']:
                    positive.append(dict(candidate,provenance={'kind':'internal_generation_positive_dev_reward',
                        'candidate':name,'incumbent':incumbent,'candidate_val':str(output/'val.json'),
                        'candidate_val_sha256':sha(output/'val.json'),
                        'incumbent_solved':results[incumbent]['solved_examples'],
                        'candidate_solved':results[name]['solved_examples'],'harness_sha256':sha(target)}))
            points=[(name,r['success_rate'],r['mean_turn_count']) for name,r in results.items()]
            frontier={'_pareto':[{'harness':n,'avg_success_rate':a,'avg_mean_turn_count':t}
                for n,a,t in compute_pareto_frontier(points)]}
            _,accepted=mh.pick_accepted_from_frontier(frontier,mhdir)
            assert accepted is not None
            write_json(run/f'mh_iteration_{iteration:02d}.json',{'accepted':accepted,'frontier':frontier,
                'results':results,'proposer_identity':read(phase/'complete.json')['identity']})
        status('mh_finished',accepted=accepted,positive_proposals=len(positive))
        selected=mhdir/f'harnesses/{accepted}/harness.py'
        write_json(run/'MH_RESULT.json',{'accepted':accepted,'results':results,'positive_proposals':len(positive),
            'all_candidates_generated_by_internal_2b':True,'external_runtime_calls':0})
        if not positive and config.get('require_harness_improvement_before_test',True):
            status('stopped_no_harness_gain_test_unopened')
            return
        if positive:
            import pandas as pd
            targets={r['extra_info']['task_id']:r['reward_model']['ground_truth']['target']
                for r in pd.read_parquet(config['data']['splits']['mh_val']['path']).to_dict('records')}
            for proposal in positive:
                provenance=proposal['provenance']; val=Path(provenance['candidate_val'])
                assert sha(val)==provenance['candidate_val_sha256']
                verified=rows(val.parent/'trajectories.jsonl')
                assert len(verified)==len(targets) and {r['task_id'] for r in verified}==set(targets)
                assert all(binary_reward(r['last_assistant'],targets[r['task_id']])==r['reward'] for r in verified)
                assert sum(r['reward'] for r in verified)==provenance['candidate_solved']>provenance['incumbent_solved']
                assert all(r['sampling_identity']['weight_digest']==identity['weight_digest'] for r in verified)
            dataset=run/'editor_online_data.jsonl'
            dataset.write_text(''.join(json.dumps(x)+'\n' for x in positive))
            online=run/'editor_online'; status('editor_reward_sft',examples=len(positive))
            editor(run,checkpoint,online,dataset,mode='train')
            checkpoint=online/'checkpoint'
            path=run/'checkpoints/harness_E1'
            if not path.exists():path.symlink_to(checkpoint,target_is_directory=True)
            # Prove that the newly updated editor can be loaded and used again.
            next_requests=run/'editor_next_requests.jsonl'
            previous=rows(trajectories[accepted]);source=selected.read_text()
            if config.get('editor_focus_mode')=='telemetry_conditioned':
                from harness_focused_bootstrap import request,choose_focus
                prompt=request(source,feedback(previous),choose_focus(previous,source,0))
            else:prompt=messages(source,feedback(previous))
            next_requests.write_text(json.dumps({'id':'next_harness','messages':prompt})+'\n')
            editor(run,checkpoint,run/'editor_next_generation',next_requests)
            runtime_probe(run,run/'editor_next_generation/generations.jsonl',
                run/'editor_next_generation/runtime_checks.json',config['data']['splits']['mh_val'])
        # One fresh task-weight update under the internally selected harness.
        if args.new_solver_round:
            phase=run/'round_01';phase.mkdir(exist_ok=True);status('solver_sampling',harness=accepted)
            if not (phase/'rollouts/val.json').exists():
                sample_parallel(run=run,phase=phase,output=phase/'rollouts',base_url=base_url,identity=identity,
                    harness=selected,data_entry=config['data']['splits']['train'],rollouts=2,seed=91,resume=True)
            if not (phase/'checkpoint/training_state.json').exists():
                status('solver_weight_sft')
                api(base_url,'/rsft/train',{'round':1,'rollouts':str(phase/'rollouts/trajectories.jsonl'),
                    'train_data':config['data']['splits']['train']['path'],'harness':str(selected),
                    'expected_weight_digest':identity['weight_digest']},timeout=7200)
            identity=api(base_url,'/checkpoint/load',{'checkpoint':str(phase/'checkpoint')},timeout=300)
            path=run/'checkpoints/solver_W2'
            if not path.exists():path.symlink_to(phase/'checkpoint',target_is_directory=True)
        frozen={'editor_checkpoint':str(checkpoint),'solver_identity':identity,'harness':str(selected),
            'harness_sha256':sha(selected),'positive_editor_proposals':len(positive),
            'new_solver_update':args.new_solver_round,'frozen_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
        write_json(run/'FROZEN.json',frozen);status('frozen_fresh_test')
        initial=read(reference/'initial_identity.json')
        cells=[('Wf_H0',identity,BASE_HARNESS),('W0_H0',initial,BASE_HARNESS),
            ('W0_Hf',initial,selected),('W1_Hf',warm_solver_identity,selected)]
        evaluate_four_cells(run=run,config=config,split='test',base_url=base_url,
            initial=initial,final=identity,final_harness=selected,resume=True,remote_cells=cells)
        write_json(run/'five_cell_result.json',{name:read(run/'test'/name/'val.json')
            for name in ['Wf_H0','W0_H0','W0_Hf','W1_Hf','Wf_Hf']})
    status('complete_pending_audit')

if __name__=='__main__':main()

"""Independently recheck two-learner trial artifacts and physical API receipts."""
import argparse
import datetime
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np
import pandas as pd
import torch
from safetensors import safe_open
from transformers import AutoTokenizer

from assess_viability import paired_change
from math_reward import binary_reward
from math_context_audit import recheck_context
from pipeline import sha,write_json
from harness_code_contract import validate_source


def read(path):return json.loads(Path(path).read_text())
def rows(path):return [json.loads(line) for line in Path(path).read_text().splitlines()]


def checkpoint(path,proof=None):
    state=read(path/'training_state.json');seen={}
    for file in path.glob('*.safetensors'):
        with safe_open(file,framework='pt',device='cpu') as handle:
            for name in handle.keys():
                if name not in state['parameter_hashes']:continue
                t=handle.get_tensor(name);assert t.dtype==torch.float32 and torch.isfinite(t).all()
                seen[name]=hashlib.sha256(t.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()
    assert seen==state['parameter_hashes']
    digest=hashlib.sha256(json.dumps(seen,sort_keys=True).encode()).hexdigest();assert digest==state['weight_digest']
    parent=Path(state['sampling_identity']['weights']);previous=read(parent/'training_state.json')
    changed=[key for key in seen if seen[key]!=previous['parameter_hashes'][key]]
    assert changed==state['changed_parameter_tensors'] or set(changed)==set(state['changed_parameter_tensors'])
    assert changed and not any('visual' in key for key in changed)
    optimizer=torch.load(path/'optimizer.pt',weights_only=True,map_location='cpu',mmap=True)
    assert {int(s['step'].item()) for s in optimizer['state'].values()}==set(state['optimizer_steps_after'])
    assert all(torch.isfinite(v).all() for s in optimizer['state'].values() for v in s.values())
    assert {str(v.dtype) for s in optimizer['state'].values() for k,v in s.items() if k!='step'}=={'torch.float32'}
    for group in optimizer['param_groups']:
        assert group['lr']==state['learning_rate']
        assert group['weight_decay']==state.get('weight_decay',0.01)
        if 'adam_betas' in state:assert list(group['betas'])==state['adam_betas']
    assert (proof if proof is not None else read(path.parent/'complete.json'))['reload_verified']
    return {'weight_digest':digest,'optimizer_steps':state['optimizer_steps_after'],
        'changed_parameter_tensors':len(changed),'fp32_finite_model_and_optimizer':True,'reload_verified':True}


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--initial-only',action='store_true');args=parser.parse_args()
    torch.set_num_threads(4);run=args.run.resolve();config=read(run/'config.json')
    upstream=Path(__file__).resolve().parents[1]/'WHALE'
    upstream_commit=subprocess.check_output(['git','-C',str(upstream),'rev-parse','HEAD'],text=True).strip()
    assert upstream_commit=='fbe125eb7abea7f760c99ab9acc1a6261e708fc6'
    subprocess.run(['git','-C',str(upstream),'diff','--quiet','HEAD'],check=True)
    assert read(run/'PROTOCOL.json')['config_sha256']==sha(run/'config.json')
    prior=Path(config['initial_reference'])
    assert sha(prior/'verification.json')==config['initial_verification_sha256']
    assert read(prior/'verification.json')['passed_audit']
    if config.get('initial_probe_origin'):
        origin=Path(config['initial_probe_origin'])
        assert sha(origin/'initial_verification.json')==config['initial_probe_verification_sha256']
        assert read(origin/'initial_verification.json')['passed_audit']
    latest_sources={}
    for manifest in sorted((run/'source_snapshots').glob('*/manifest.json'),
            key=lambda p:({'initial':0,'cycles':1}.get(p.parent.name,2),p.parent.name)):
        for source,digest in read(manifest).items():
            assert sha(manifest.parent/Path(source).name)==digest,(source,manifest)
            latest_sources[source]=digest
    for source,digest in latest_sources.items():assert sha(source)==digest,source
    data={};targets={};ids={}
    for split,entry in config['data']['splits'].items():
        assert sha(entry['path'])==entry['sha256']
        rr=pd.read_parquet(entry['path']).to_dict('records');assert len(rr)==entry['count']
        data[entry['path']]=entry
        targets[entry['path']]={r['extra_info']['task_id']:r['reward_model']['ground_truth']['target'] for r in rr}
        ids[split]=[r['extra_info']['task_id'] for r in rr]
    assert len(set(sum(ids.values(),[])))==sum(len(v) for v in ids.values()),'Within-run split overlap'
    receipt_files={};receipt_hashes={}
    for file in run.rglob('calls/*.json'):
        key=file.stem;digest=sha(file)
        if key in receipt_hashes:assert receipt_hashes[key]==digest
        receipt_files[key]=file;receipt_hashes[key]=digest
    used=set();verified={};context_ids=set()
    for file in run.rglob('val.json'):
        if 'shards' in file.parts:continue
        value=read(file);meta=value['metadata'];entry=data[meta['dataset_path']]
        assert meta['dataset_sha256']==entry['sha256']
        rr=rows(file.parent/'trajectories.jsonl');plan=meta['sampling_schedule']
        assert len(rr)==value['num_examples']==len(plan)
        split=next(k for k,v in config['data']['splits'].items() if v['path']==entry['path'])
        for row,(draw,index) in zip(rr,plan):
            assert row['draw']==draw and row['example_index']==index and row['task_id']==ids[split][index]
            assert row['rollout_seed']==meta['seed']*100003+draw*1009+index
            assert row['reward']==binary_reward(row['last_assistant'],targets[entry['path']][row['task_id']])
            assert row['sampling_identity']['weight_digest']==meta['weight_digest']
            assert row['harness_sha256']==meta['harness_sha256'] and row['answer_redacted']
            assert row['assistant_tokens']==sum(len(c['response_ids']) for c in row['policy_calls'])<=2048
            physical=Path(row['physical_worker_run'])
            for c in row['policy_calls']:
                original=physical/'calls'/(c['call_id']+'.json')
                assert original.exists() and read(original)==c and read(receipt_files[c['call_id']])==c
                assert c['sampling_identity']['weight_digest']==meta['weight_digest']
                assert c['stop_at_chat_end'] and len(c['response_ids'])<=c['requested_max_tokens']<=1024
                assert len(c['response_ids'])==len(c['response_logprobs']) and np.isfinite(c['response_logprobs']).all()
                used.add(c['call_id'])
            if row.get('context_rejection_id'):
                key=row['context_rejection_id']
                if key not in context_ids:recheck_context(physical/'context_rejections'/(key+'.json'));context_ids.add(key)
        assert sum(r['reward'] for r in rr)==value['solved_examples']
        verified[str(file.parent)]=value
    assert used==set(receipt_files),'Unaccounted task-model API calls'
    def outcome(path):
        key=str(path)
        if key not in verified:
            origin=Path(config['initial_probe_origin'])
            assert Path(path).is_relative_to(origin)
            copied=run/Path(path).relative_to(origin)
            assert sha(Path(path)/'val.json')==sha(copied/'val.json')
            assert sha(Path(path)/'trajectories.jsonl')==sha(copied/'trajectories.jsonl')
            return verified[str(copied)]
        return verified[key]
    tokenizer=AutoTokenizer.from_pretrained(config['editor_initial_checkpoint'],local_files_only=True)
    editor_checks={};all_generations={}
    for file in run.rglob('generations.jsonl'):
        directory=file.parent;complete=read(directory/'complete.json');assert sha(file)==complete['generations_sha256']
        requests=rows(directory/'requests.jsonl');generated=rows(file)
        assert len(generated)==complete['requests']==len(requests)
        for request,g in zip(requests,generated):
            assert g['id']==request['id'] and g['group']==request['group'] and g['messages']==request['messages']
            assert g['seed']==request['seed'] and g['sampling_identity']['weight_digest']==complete['identity']['weight_digest']
            text=tokenizer.apply_chat_template(g['messages'],tokenize=False,add_generation_prompt=True,enable_thinking=False)
            assert tokenizer(text,add_special_tokens=False)['input_ids']==g['prompt_ids']
            assert tokenizer.decode(g['response_ids'],skip_special_tokens=True)==g['content']
            assert len(g['response_logprobs'])==len(g['response_ids']) and np.isfinite(g['response_logprobs']).all()
            if g['valid_source']:
                assert validate_source(g['content'])==g['source']==(directory/(g['id']+'.py')).read_text()
        scored=rows(directory/'scored.jsonl');assert len(scored)==len(generated)
        for g,s in zip(generated,scored):
            assert all(s[k]==v for k,v in g.items())
            provenance=s['reward_provenance'];baseline=Path(provenance['baseline'])
            assert sha(baseline/'val.json')==provenance['baseline_val_sha256']
            base=outcome(baseline)
            if s['outcome_kind']=='valid':
                candidate=Path(provenance['candidate']);value=outcome(candidate)
                assert sha(candidate/'val.json')==provenance['candidate_val_sha256']
                assert base['metadata']['weight_digest']==value['metadata']['weight_digest']==provenance['fixed_solver']['weight_digest']
                assert base['metadata']['dataset_sha256']==value['metadata']['dataset_sha256']
                assert base['metadata']['sampling_schedule']==value['metadata']['sampling_schedule']
                assert base['metadata']['temperature']==value['metadata']['temperature']
                assert base['metadata']['seed']==value['metadata']['seed']
                assert s['reward_delta']==value['success_rate']-base['success_rate']
            else:assert s['reward_delta']==0 and provenance['candidate'] is None
        editor_checks[str(directory)]={'generated':len(generated),'mean_reward_delta':np.mean([s['reward_delta'] for s in scored]),
            'positive_proposals':sum(s['reward_delta']>0 for s in scored),'identity':complete['identity']['weight_digest']}
        all_generations[str(directory)]=generated
    e0=rows(run/'editor_probes/E0/requests.jsonl');e1=rows(run/'editor_probes/E1/requests.jsonl');assert e0==e1
    expected={'E0':config['editor_comparison_checkpoint'],
        'E1':config.get('initial_probe_E1_checkpoint',config['editor_initial_checkpoint'])}
    if config.get('composition_bootstrap'):expected['E1c']=config['editor_initial_checkpoint']
    for name,path in expected.items():
        assert read(run/'editor_probes'/name/'complete.json')['identity']['weight_digest']==read(Path(path)/'training_state.json')['weight_digest']
    for file in run.rglob('services.json'):
        stopped=read(file.with_name('services_stopped.json'));started=read(file)
        assert stopped['worker_pid']==started['worker_pid'] and stopped['sandbox_pid']==started['sandbox_pid']
        assert stopped['worker_returncode'] in (0,-15) and stopped['sandbox_returncode'] in (0,-15)
    report={'passed_audit':True,'phase':'initial' if args.initial_only else 'full',
        'verified_unique_task_calls':len(used),'verified_outcomes':len(verified),'editor_probes':editor_checks,
        'verified_task_trajectories':sum(v['num_examples'] for v in verified.values()),
        'within_run_splits_disjoint':True,'monitor_is_reused_not_fresh':True,'teacher_runtime_calls':0,
        'exact_model_generated_code_tokens_verified':True,'same_batch_delta_rewards_recomputed':True,
        'context_rejections':len(context_ids),'original_server_receipts_verified':True,'services_stopped':True}
    report.update(upstream_whale_commit=upstream_commit,upstream_tracked_files_unchanged=True)
    if config.get('initial_probe_origin'):
        reused={p.stem for p in (run/'replicas').rglob('calls/*.json')}
        report.update(reused_initial_task_calls=len(reused),new_cycle_task_calls=len(used-reused))
    if (run/'TRANSPORT_AMENDMENT.json').exists():
        amendment=read(run/'TRANSPORT_AMENDMENT.json')
        assert read(run/'CHECKPOINT_LOAD_CONTRACT_CHECK.json')['passed']
        assert amendment['preserved_editor_digest']==read(run/'cycle_01/editor_grpo/complete.json')['identity']['weight_digest']
        assert amendment['preserved_solver_digest']==read(run/'cycle_01/solver_update/complete.json')['identity']['weight_digest']
        loads=list(run.rglob('checkpoint_loads/*.json'))
        for file in loads:
            record=read(file);assert record['all_parameter_hashes_equal']
            consumer=read(file.parent.parent/'worker_config.json')
            assert record['weights'] in consumer['checkpoint_allowlist']
            assert read(Path(record['weights'])/'training_state.json')['weight_digest']==record['weight_digest']
        report.update(transport_amendment_preserves_training=True,verified_cross_checkpoint_loads=len(loads))
    if not args.initial_only:
        checks={};curve=read(run/'monitor_curve.json');contrasts=[]
        assert read(run/'CYCLES_PROTOCOL.json')['config_sha256']==sha(run/'config.json')
        assert len(curve)==1+2*config['cycles']
        if config.get('composition_bootstrap'):
            bridge=Path(config['composition_bootstrap']);ready=read(bridge/'READY.json')
            assert ready['passed_readiness'] and ready['checkpoint']==config['editor_initial_checkpoint']
            assert ready['executable_preserving_changed_guidance']>=7 and ready['distinct_system_prompts']>=2
            assert sha(bridge/'PROTOCOL.json')==ready['protocol_sha256']
            for path in sorted(bridge.glob('editor_sft*/checkpoint')):
                state=read(path/'training_state.json');checks[str(path)]=checkpoint(path)
                assert state['epochs']==2 and state['learning_rate']==1e-5 and state['examples']==64
                assert sha(state['data'])==state['data_sha256']==read(bridge/'PROTOCOL.json')['train_sha256']
                assert all(np.isfinite(np.asarray(v)).all() for m in state['metrics'] for v in m.values())
        previous=None;initial_outcome=None
        for point in curve:
            rr=[r for r in rows(point['trajectories']) if r['draw']<config['monitor_draws']]
            assert len(rr)==64 and {r['task_id'] for r in rr}==set(ids['monitor'])
            assert sum(r['reward'] for r in rr)==point['successful_draws']
            outcome=np.array([sum(r['reward'] for r in rr if r['task_id']==task)/2 for task in ids['monitor']])
            if initial_outcome is None:initial_outcome=outcome.copy()
            if previous is not None:contrasts.append({'from':previous[0],'to':point['label'],**paired_change(previous[1],outcome)})
            previous=(point['label'],outcome)
        expected_editor=read(Path(config['editor_initial_checkpoint'])/'training_state.json')['weight_digest']
        expected_solver=read(Path(config['solver_initial_checkpoint'])/'training_state.json')['weight_digest']
        expected_harness=sha(config['initial_harness'])
        cycle_paths=sorted(run.glob('cycle_*'));assert len(cycle_paths)==config['cycles']
        for cycle in cycle_paths:
            done=read(cycle/'complete.json');start=read(cycle/'START.json')
            assert start['editor']['weight_digest']==expected_editor
            assert start['solver']['weight_digest']==expected_solver
            assert start['harness_sha256']==expected_harness
            trained=read(cycle/'editor_grpo/complete.json')
            if trained['updated']:
                path=cycle/'editor_grpo/checkpoint';state=read(path/'training_state.json');checks[str(path)]=checkpoint(path)
                assert state['sampling_identity']['weight_digest']==start['editor']['weight_digest']
                assert state['epochs']==config['editor_grpo_epochs']==4
                training=rows(cycle/'editor_grpo/rewards.jsonl');assert sha(cycle/'editor_grpo/rewards.jsonl')==state['data_sha256']
                train_source=read(cycle/'TRAIN_SOURCE.json')
                assert training==rows(Path(train_source['directory'])/'scored.jsonl')
                assert sha(Path(train_source['directory'])/'scored.jsonl')==train_source['scored_sha256']
                batch=torch.load(cycle/'editor_grpo/grpo_batch.pt',weights_only=True,map_location='cpu')
                recomputed=np.zeros(len(training))
                for group in {r['group'] for r in training}:
                    ix=[i for i,r in enumerate(training) if r['group']==group];assert len(ix)==8
                    rewards=np.array([training[i]['reward_delta'] for i in ix]);std=rewards.std()
                    if std>1e-8:recomputed[ix]=(rewards-rewards.mean())/std
                assert np.allclose(batch['advantages'].numpy(),recomputed) and np.any(recomputed)
                assert batch['ids']==[r['id'] for r in training]
                for lp,sampled,r in zip(batch['old_training_logprobs'],batch['rollout_logprobs'],training):
                    assert torch.isfinite(lp).all() and len(lp)==len(r['response_ids'])
                    assert sampled==r['response_logprobs']
                assert len(state['metrics'])==4 and all(np.isfinite(list(m.values())).all() for m in state['metrics'])
                assert state['optimizer_steps_after']==[(max(state['optimizer_steps_before']) if state['optimizer_steps_before'] else 0)+4]
            post=read(cycle/'editor_post_proposals/complete.json')
            assert post['identity']['weight_digest']==read(Path(done['editor'])/'training_state.json')['weight_digest']
            decision=read(cycle/'selection/decision.json')
            assert sha(done['harness'])==decision['selected_sha256']
            if decision['accepted']!='incumbent':
                assert Path(done['harness']).read_text()==(cycle/'editor_post_proposals'/(decision['accepted']+'.py')).read_text()
            path=Path(done['solver']);state=read(path/'training_state.json')
            # The solver's completion record is outside round_01; validate here.
            complete=read(cycle/'solver_update/complete.json');assert complete['reload_verified']
            # checkpoint() accepts the editor-style adjacent proof; solver proof
            # is checked separately and passed via the helper below.
            checks[str(path)]=solver_checkpoint(path,complete)
            assert state['learning_rate']==config['solver_rsft_lr']
            assert state['sampling_identity']['weight_digest']==start['solver']['weight_digest']
            training=rows(cycle/'solver_rollouts/trajectories.jsonl')
            accepted=[c for r in training if r['reward'] for c in r['policy_calls']]
            tensor=torch.load(path.parent/'accepted_batch.pt',weights_only=True,map_location='cpu')
            assert len(accepted)==len(tensor['responses'])==state['accepted_policy_calls']
            pmax=tensor['input_ids'].shape[1]-tensor['responses'].shape[1]
            for i,c in enumerate(accepted):
                assert tensor['input_ids'][i,:pmax][tensor['attention_mask'][i,:pmax].bool()].tolist()==c['prompt_ids']
                assert tensor['responses'][i][tensor['response_mask'][i].bool()].tolist()==c['response_ids']
            progress=read(path.parent/'training_progress.json');updates=(len(accepted)+3)//4
            assert progress['completed_optimizer_updates']==state['metrics']['actor/sft_updates']==updates
            assert state['optimizer_steps_after']==[max(state['optimizer_steps_before'])+updates]
            assert all(np.isfinite(np.asarray(v)).all() for v in state['metrics'].values())
            expected_editor=done['editor_digest'];expected_solver=done['solver_digest'];expected_harness=sha(done['harness'])
        final=read(run/'FINAL_MODELS.json')
        assert final['editor']['weight_digest']==expected_editor and final['solver']['weight_digest']==expected_solver
        assert final['harness_sha256']==expected_harness
        report.update(checkpoints=checks,monitor_curve=curve,adjacent_monitor_changes=contrasts,
            total_monitor_change=paired_change(initial_outcome,previous[1]),
            actual_editor_weight_updates=sum(read(p)['updated'] for p in run.glob('cycle_*/editor_grpo/complete.json')),
            limitation='Repeated monitoring and small policy probes do not prove generalization or monotonic self-improvement.')
    target=run/('initial_verification.json' if args.initial_only else 'verification.json');write_json(target,report)
    if not args.initial_only:
        write_json(run/'STATUS.json',{'stage':'alternation_complete_audited','updated_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
            'verification_sha256':sha(target)})
    print(json.dumps(report,indent=2),flush=True)


def solver_checkpoint(path,complete):
    # Preserve checkpoint artifacts: use the same tensor/optimizer audit while
    # supplying the solver's actual reload-proof location explicitly.
    return checkpoint(path,complete)


if __name__=='__main__':main()

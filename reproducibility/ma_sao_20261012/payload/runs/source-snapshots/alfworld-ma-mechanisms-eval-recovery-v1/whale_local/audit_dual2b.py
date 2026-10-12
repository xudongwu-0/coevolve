"""Independent saved-artifact audit for the internal-editor pilot."""
import argparse
import datetime
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from safetensors import safe_open
from assess_viability import paired_change
from math_context_audit import recheck_context
from math_reward import binary_reward
from pipeline import sha, write_json

def read(p): return json.loads(Path(p).read_text())
def rows(p): return [json.loads(x) for x in Path(p).read_text().splitlines()]

def checkpoint(path):
    state=read(path/'training_state.json')
    seen={}
    for file in path.glob('*.safetensors'):
        with safe_open(file,framework='pt',device='cpu') as f:
            for name in f.keys():
                if name not in state['parameter_hashes']: continue
                t=f.get_tensor(name)
                seen[name]=hashlib.sha256(t.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()
                assert t.dtype==torch.float32
                assert torch.isfinite(t).all(),f'Nonfinite saved parameter: {name}'
    # HF tied tensors can be saved under a single alias; named_parameters
    # used in training must still be covered by the persisted tensor names.
    assert seen==state['parameter_hashes']
    digest=hashlib.sha256(json.dumps(seen,sort_keys=True).encode()).hexdigest()
    assert digest==state['weight_digest']
    opt=torch.load(path/'optimizer.pt',map_location='cpu',weights_only=True,mmap=True)
    assert {int(v['step'].item()) for v in opt['state'].values()}==set(state['optimizer_steps_after'])
    assert {str(v.dtype) for s in opt['state'].values() for k,v in s.items() if k!='step'}=={'torch.float32'}
    assert all(torch.isfinite(v).all() for s in opt['state'].values() for v in s.values())
    assert all(g['lr']==state['learning_rate'] and g['weight_decay']==0.01 for g in opt['param_groups'])
    assert state['changed_parameter_tensors'] and not any('visual' in n for n in state['changed_parameter_tensors'])
    assert state['qwen35_padding_compatibility_fix']['name']=='qwen35-singleton-padding-mask-v1'
    return {'weight_digest':digest,'optimizer_steps':state['optimizer_steps_after'],
        'changed_tensors':len(state['changed_parameter_tensors']),'fp32_parameters_and_moments':True,
        'all_saved_parameters_and_optimizer_states_finite':True}

def main():
    torch.set_num_threads(4)
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);args=p.parse_args()
    run=args.run.resolve();config=read(run/'config.json');frozen=read(run/'FROZEN.json')
    assert read(run/'STATUS.json')['stage'] in {'complete_pending_audit','complete_audited'}
    config_hash=read(run/'PROTOCOL_AMENDMENT_BEFORE_MH.json')['config_sha256']
    sources=read(run/'ONLINE_SOURCE_MANIFEST.json')
    for path in sorted(run.glob('NUMERICAL_AMENDMENT_*.json')):
        amendment=read(path);assert amendment['old_config_sha256']==config_hash
        config_hash=amendment['new_config_sha256']
        for name,change in amendment['source_changes'].items():
            assert sources.get(name)==change['old_sha256']
            sources[name]=change['new_sha256']
        assert amendment['fresh_test_unopened'] and amendment['new_rollouts']==0
    assert sha(run/'config.json')==config_hash
    for path,digest in sources.items(): assert sha(path)==digest
    reference=Path(config['solver_prefix_reference']);prior=read(reference/'config.json')
    assert sha(reference/'verification.json')==read(run/'PROTOCOL.json')['reference_verification_sha256']
    assert read(reference/'verification.json')['passed_audit']
    identities=read(run/'solver_initial_identity.json')
    assert identities['weight_digest']==read(reference/'round_01/checkpoint/training_state.json')['weight_digest']
    targets={};ids={}
    for split,entry in config['data']['splits'].items():
        assert sha(entry['path'])==entry['sha256']
        items=pd.read_parquet(entry['path']).to_dict('records')
        targets[split]={r['extra_info']['task_id']:r['reward_model']['ground_truth']['target'] for r in items}
        ids[split]=[r['extra_info']['task_id'] for r in items]
        assert len(set(ids[split]))==entry['count']
    assert not set(ids['train'])&set(ids['mh_val']) and not set(ids['test'])&(set(ids['train'])|set(ids['mh_val']))
    old_keys={info['task_id'] for e in prior['data']['splits'].values() for info in pd.read_parquet(e['path'])['extra_info']}
    assert not old_keys&set(ids['test'])
    official_train={hashlib.sha256(r['problem'].strip().encode()).hexdigest()
        for r in pd.read_parquet(prior['data']['source_files']['train']['path']).to_dict('records')}
    assert not official_train&set(ids['test'])
    receipts={};receipt_sha={}
    for file in run.rglob('calls/*.json'):
        key=file.stem;digest=sha(file)
        if key in receipt_sha: assert receipt_sha[key]==digest
        receipts[key]=file;receipt_sha[key]=digest
    used_calls=set();contexts=[]
    def verify_rows(path,split,weight,harness,draws,seed):
        data=rows(path);assert len(data)==len(ids[split])*draws
        for pos,r in enumerate(data):
            draw,index=divmod(pos,len(ids[split]))
            assert (r['draw'],r['example_index'],r['task_id'])==(draw,index,ids[split][index])
            assert r['rollout_seed']==seed*100003+draw*1009+index
            assert r['reward']==binary_reward(r['last_assistant'],targets[split][r['task_id']])
            assert r['sampling_identity']['weight_digest']==weight and r['harness_sha256']==harness
            assert r['answer_redacted']
            if r['policy_calls']:
                content=r['policy_calls'][-1]['parsed_message']['content']
                assert r['last_assistant']==(content if isinstance(content,str) else '')
            assert r['assistant_tokens']==sum(len(c['response_ids']) for c in r['policy_calls'])<=2048
            for c in r['policy_calls']:
                assert c==read(receipts[c['call_id']]);used_calls.add(c['call_id'])
                assert c['sampling_identity']['weight_digest']==weight
                assert c['stop_at_chat_end'] and len(c['response_ids'])<=c['requested_max_tokens']<=1024
                assert len(c['response_ids'])==len(c['response_logprobs'])
                assert not any(t in (248044,248046) for t in c['response_ids'][:-1])
            if r.get('context_rejection_id'):
                files=list(run.rglob(r['context_rejection_id']+'.json'));assert files
                contexts.append(recheck_context(files[0]))
        return data
    mh=read(run/'MH_RESULT.json'); assert mh['all_candidates_generated_by_internal_2b'] and mh['external_runtime_calls']==0
    generators={}
    from transformers import AutoTokenizer
    editor_tokenizer=AutoTokenizer.from_pretrained(config['editor_initial_checkpoint'],local_files_only=True)
    for phase in sorted(run.glob('editor_proposals_*')):
        if not phase.is_dir(): continue
        proof=read(phase/'complete.json')
        assert sha(phase/'generations.jsonl')==proof['generations_sha256']
        assert proof['identity']['weight_digest']==read(Path(config['editor_initial_checkpoint'])/'training_state.json')['weight_digest']
        for generated in rows(phase/'generations.jsonl'):
            text=editor_tokenizer.apply_chat_template(generated['messages'],tokenize=False,
                add_generation_prompt=True,enable_thinking=False)
            assert editor_tokenizer(text,add_special_tokens=False)['input_ids']==generated['prompt_ids']
            assert editor_tokenizer.decode(generated['response_ids'],skip_special_tokens=True)==generated['content']
            generators[generated['id']]=generated
    assert len(generators)==4
    for name,value in mh['results'].items():
        if name=='h0':continue
        path=run/'mh_results'/name/'trajectories.jsonl'
        code=run/'internal_mh/harnesses'/name/'harness.py'
        assert code.read_text()==generators[name]['source']
        verified=verify_rows(path,'mh_val',identities['weight_digest'],sha(code),1,42)
        assert sum(r['reward'] for r in verified)==value['solved_examples']
    checks={'editor_E0':checkpoint(Path(config['editor_initial_checkpoint']))}
    if (run/'editor_online').exists():
        online=rows(run/'editor_online_data.jsonl')
        training=read(run/'editor_online/checkpoint/training_state.json')
        assert training['data_sha256']==sha(run/'editor_online_data.jsonl')
        assert training['sampling_identity']['weight_digest']==checks['editor_E0']['weight_digest']
        tensor_rows=torch.load(run/'editor_online/tokenized_examples.pt',weights_only=True)
        assert tensor_rows==online
        for r in online:
            original=generators[r['id']]
            for k in original: assert original[k]==r[k]
            source=r['provenance'];assert source['kind']=='internal_generation_positive_dev_reward'
            assert sha(source['candidate_val'])==source['candidate_val_sha256']
            assert source['candidate_solved']>source['incumbent_solved']
            assert mh['results'][source['candidate']]['solved_examples']==source['candidate_solved']
            assert mh['results'][source['incumbent']]['solved_examples']==source['incumbent_solved']
        checks['editor_E1']=checkpoint(run/'editor_online/checkpoint')
        assert checks['editor_E1']['weight_digest']!=checks['editor_E0']['weight_digest']
        assert read(run/'editor_next_generation/complete.json')['identity']['weight_digest']==checks['editor_E1']['weight_digest']
    assert frozen['new_solver_update']
    phase=run/'round_01';state=read(phase/'checkpoint/training_state.json')
    assert state['sampling_identity']['weight_digest']==identities['weight_digest']
    train_rows=verify_rows(phase/'rollouts/trajectories.jsonl','train',identities['weight_digest'],frozen['harness_sha256'],2,91)
    accepted=[c for r in train_rows if r['reward'] for c in r['policy_calls']]
    batch=torch.load(phase/'accepted_batch.pt',map_location='cpu',weights_only=True)
    assert len(accepted)==len(batch['responses'])==state['accepted_policy_calls']
    assert sum(r['reward'] for r in train_rows)==state['accepted_trajectories']
    pmax=batch['input_ids'].shape[1]-batch['responses'].shape[1]
    for i,c in enumerate(accepted):
        assert batch['input_ids'][i,:pmax][batch['attention_mask'][i,:pmax].bool()].tolist()==c['prompt_ids']
        assert batch['responses'][i][batch['response_mask'][i].bool()].tolist()==c['response_ids']
    checks['solver_W2']=checkpoint(phase/'checkpoint')
    assert checks['solver_W2']['weight_digest']==frozen['solver_identity']['weight_digest']
    before=read(reference/'round_01/checkpoint/training_state.json')['optimizer_steps_after']
    assert state['optimizer_steps_before']==before
    assert state['optimizer_steps_after']==[max(before)+int(state['metrics']['actor/sft_updates'])]
    planned_updates=(len(accepted)+3)//4
    assert state['metrics']['actor/sft_updates']==planned_updates
    assert all(np.isfinite(np.asarray(v)).all() for v in state['metrics'].values())
    progress=read(phase/'training_progress.json')
    assert progress['completed_optimizer_updates']==progress['planned_minibatches']==planned_updates
    numeric={'completed_updates':planned_updates,'all_reported_training_metrics_finite':True}
    if config.get('gradient_norm_fix'):
        fix=state['gradient_norm_precision_fix']
        assert fix['name']==config['gradient_norm_fix']
        assert fix['source_sha256']==sha(Path(__file__).with_name('safe_gradient_norm.py'))
        events=read(run/'gradient_norm_events.json') if (run/'gradient_norm_events.json').exists() else []
        assert events==state['gradient_norm_precision_events']
        for event in events:
            assert event['action']=='clip_finite_gradients_using_fp64_norm'
            assert not event['nonfinite_gradient_tensor_indices']
            assert 0<=event['minibatch_index']<planned_updates
            assert np.isfinite(event['norm_fp64']) and event['norm_fp64']>0
            assert 0<event['clip_coefficient']<=1
        failure=read(run/'FAILED_WEIGHT_UPDATE.json')
        assert failure['checkpoint_saved'] is False and failure['test_opened'] is False
        assert sha(phase/'rollouts/trajectories.jsonl')==read(run/'NUMERICAL_AMENDMENT_01.json')['data_unchanged_sha256']
        numeric.update(failed_attempt=failure,precision_guard=fix,fp64_fallback_events=events,
            retry_used_identical_sampled_trajectories=True,
            retry_finite_gradient_norm_overflow_observed=bool(events),
            original_nonfinite_norm_cause_established=False)
    initial=read(reference/'initial_identity.json')
    from pathlib import Path as P
    h0=sha(P(__file__).resolve().parents[1]/'WHALE/domains/math_reasoning/environments/retool/base_harness.py')
    cell_weights={'W0_H0':initial['weight_digest'],'W0_Hf':initial['weight_digest'],
        'W1_Hf':identities['weight_digest'],'Wf_H0':checks['solver_W2']['weight_digest'],'Wf_Hf':checks['solver_W2']['weight_digest']}
    scores={};outcomes={}
    for name,weight in cell_weights.items():
        path=run/'test'/name;harness=h0 if name.endswith('H0') else frozen['harness_sha256']
        verified=verify_rows(path/'trajectories.jsonl','test',weight,harness,4,42)
        val=read(path/'val.json');assert val['metadata']['temperature']==1.0
        assert val['solved_examples']==sum(r['reward'] for r in verified)
        outcomes[name]=np.array([r['reward'] for r in verified]).reshape(4,32).mean(axis=0)
        scores[name]={'successful_draws':val['solved_examples'],'draws':128,'mean_at_4_percent':100*outcomes[name].mean()}
    contrasts={label:paired_change(outcomes[a],outcomes[b]) for label,a,b in [
        ('harness_at_final_weights','Wf_H0','Wf_Hf'),('new_solver_update','W1_Hf','Wf_Hf'),
        ('all_solver_updates','W0_Hf','Wf_Hf'),('joint_change','W0_H0','Wf_Hf')]}
    for file in run.rglob('services.json'):
        stopped=read(file.with_name('services_stopped.json'));started=read(file)
        assert stopped['worker_pid']==started['worker_pid'] and stopped['sandbox_pid']==started['sandbox_pid']
        assert stopped['worker_returncode'] in (0,-15) and stopped['sandbox_returncode'] in (0,-15)
    for file in run.rglob('remote_transport.json'):
        assert file.with_name('services_stopped.json').exists()
    assert used_calls==set(receipts),'Unaccounted physical policy calls'
    usage={'new_task_policy_calls':len(used_calls),
        'new_policy_generated_tokens':sum(len(read(p)['response_ids']) for p in receipts.values()),
        'editor_search_proposals':4,'editor_search_generated_tokens':sum(len(r['response_ids']) for r in generators.values()),
        'cached_solver_prefix_included_in_logical_training_budget':True}
    report={'passed_audit':True,'scores':scores,'contrasts':contrasts,'checkpoints':checks,'usage':usage,
        'positive_editor_training_proposals':mh['positive_proposals'],'fresh_test_questions':32,'test_draws_per_question':4,
        'native_reward_rechecked':True,'exact_sft_tokens_verified':True,'internal_editor_receipts_verified':True,
        'context_rejections':contexts,'teacher_runtime_calls':0,'solver_numerical_validation':numeric,
        'limits':['bootstrap wording checks are development only','small pilot, no sustained self-improvement claim',
            'editor update has no independent randomized frozen-editor control','not an equal-budget comparison with external-proposer WHALE']}
    write_json(run/'verification.json',report)
    write_json(run/'STATUS.json',{'stage':'complete_audited','updated_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'verification_sha256':sha(run/'verification.json')})
    print(json.dumps(report,indent=2))

if __name__=='__main__':main()

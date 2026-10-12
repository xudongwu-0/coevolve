"""Audit fresh SearchQA WHALE receipts, training tensors, and stage attribution."""
import argparse
import json
import math
from pathlib import Path

import pandas as pd
from safetensors import safe_open
import torch

from pipeline import sha, write_json
from verify_search_run import native_em_scope, recheck_rewards


def read(p):return json.loads(Path(p).read_text())
def rows(p):return [json.loads(s) for s in Path(p).read_text().splitlines()]


def contrast(before,after):
    left={r['task_id']:r['reward'] for r in before};right={r['task_id']:r['reward'] for r in after}
    assert set(left)==set(right)
    wins=sum(right[k]>left[k] for k in left);losses=sum(right[k]<left[k] for k in left)
    n=wins+losses
    p=min(1.,2*sum(math.comb(n,k) for k in range(min(wins,losses)+1))/2**n) if n else 1.
    return {'difference_pp':100*sum(right[k]-left[k] for k in left)/len(left),
        'wins':wins,'losses':losses,'unchanged':len(left)-n,'mcnemar_exact_two_sided_p':p}


def main():
    parser=argparse.ArgumentParser();parser.add_argument('run',type=Path);args=parser.parse_args()
    run=args.run.resolve();config=read(run/'config.json');scope=native_em_scope()
    assert read(run/'STATUS.json')['stage']=='complete'
    assert sha(run/'config.json')==read(run/'PROTOCOL.json')['config_sha256']
    for item in read(run/'source_snapshot/manifest.json'):assert sha(item['path'])==item['sha256']
    for split,entry in config['data']['splits'].items():assert sha(entry['path'])==entry['sha256']
    assert sha(config['data']['corpora']['path'])==config['data']['corpora']['sha256']
    assert not (run/'test').exists()
    # Only train/development answers are opened by this audit; reserved test stays unopened.
    ids={name:{r['task_id'] for r in pd.read_parquet(config['data']['splits'][name]['path'])['extra_info']}
         for name in ['train','mh_val','monitor']}
    assert len(set.union(*ids.values()))==sum(len(v) for v in ids.values())
    all_records=[];calls_seen=set();cost={}
    def audit_output(output,split):
        value=read(output/'val.json');records=rows(output/'trajectories.jsonl')
        assert len(records)==value['num_examples']
        assert sum(r['reward'] for r in records)==value['solved_examples']
        assert value['success_rate']==sum(r['reward'] for r in records)/len(records)
        assert value['metadata']['seed']==42 and value['metadata']['temperature']==1.0
        assert value['metadata']['dataset_sha256']==config['data']['splits'][split]['sha256']
        assert len({(r['draw'],r['example_index']) for r in records})==len(records)
        recheck_rewards(records,config['data']['splits'][split],scope)
        for record in records:
            assert record['task_id'] in ids[split]
            assert record['rollout_seed']==42*100003+record['draw']*1009+record['example_index']
            assert record['harness_sha256']==value['metadata']['harness_sha256']
            assert record['sampling_identity']['weight_digest']==value['metadata']['weight_digest']
            if split!='train':assert record['draw']==0
            assert record['assistant_tokens']==sum(len(c['response_ids']) for c in record['policy_calls'])<=2048
            physical=Path(record['physical_worker_run'])
            if record.get('context_rejection_id'):
                assert (physical/'context_rejections'/(record['context_rejection_id']+'.json')).exists()
            for call in record['policy_calls']:
                assert call==read(physical/'calls'/(call['call_id']+'.json'))
                assert call['call_id'] not in calls_seen;calls_seen.add(call['call_id'])
                assert len(call['response_logprobs'])==len(call['response_ids'])
                assert len(call['response_ids'])<=call['requested_max_tokens']<=768
                assert len(call['prompt_ids'])+call['requested_max_tokens']<=4096
                assert not any(t in (248044,248046) for t in call['response_ids'][:-1])
        all_records.extend(records)
        cost.setdefault(split,{'trajectories':0,'policy_calls':0,'assistant_tokens':0})
        cost[split]['trajectories']+=len(records)
        cost[split]['policy_calls']+=sum(len(r['policy_calls']) for r in records)
        cost[split]['assistant_tokens']+=sum(r['assistant_tokens'] for r in records)
        return records
    initial=read(run/'initial_identity.json');previous=initial;harness=Path(config['initial_harness'])
    training=[]
    for n in [1,2]:
        cycle=run/f'cycle_{n:02d}';update=cycle/'solver_update';state=read(update/'round_01/checkpoint/training_state.json')
        rollout=audit_output(cycle/'train_rollouts','train')
        assert len(rollout)==128*4
        assert state['sampling_identity']['weight_digest']==previous['weight_digest']
        assert state['optimizer_steps_before']==previous['optimizer_steps']
        assert state['harness_sha256']==sha(harness)
        assert state['padding_fix']['name']=='qwen35-singleton-padding-mask-v1'
        assert state['parameter_dtype']=='float32' and state['optimizer_state_dtypes']==['torch.float32']
        assert state['changed_parameter_tensors']
        assert max(state['optimizer_steps_after'])==max(previous['optimizer_steps'] or [0])+int(state['metrics']['actor/sft_updates'])
        complete=read(update/'complete.json');assert complete['reload_verified']
        assert complete['identity']['weight_digest']==state['weight_digest']
        accepted=[c for r in rollout if r['reward']==1 for c in r['policy_calls']]
        assert sum(r['reward'] for r in rollout)==state['accepted_trajectories']
        assert len(accepted)==state['accepted_policy_calls']
        batch=torch.load(update/'round_01/accepted_batch.pt',map_location='cpu',weights_only=True)
        assert len(batch['responses'])==len(accepted)
        pmax=batch['input_ids'].shape[1]-batch['responses'].shape[1]
        for i,c in enumerate(accepted):
            assert batch['input_ids'][i,:pmax][batch['attention_mask'][i,:pmax].bool()].tolist()==c['prompt_ids']
            assert batch['responses'][i][batch['response_mask'][i].bool()].tolist()==c['response_ids']
            assert c==read(update/'calls'/(c['call_id']+'.json'))
        assert int(batch['response_mask'].sum())==state['response_tokens']
        dtype=set()
        for p in (update/'round_01/checkpoint').glob('*.safetensors'):
            with safe_open(p,framework='pt',device='cpu') as saved:
                dtype.update(saved.get_slice(k).get_dtype() for k in saved.keys() if 'visual' not in k)
        assert dtype=={'F32'}
        optimizer=torch.load(update/'round_01/checkpoint/optimizer.pt',map_location='cpu',weights_only=True)
        assert {v.dtype for st in optimizer['state'].values() for k,v in st.items() if k!='step' and torch.is_tensor(v)}=={torch.float32}
        del optimizer,batch
        evaluations=sorted((run/'mh'/cycle.name/'logs/mh_val').glob('*/*/val.json'))
        assert len(evaluations)==5
        for path in evaluations:
            vals=audit_output(path.parent,'mh_val');assert len(vals)==48
            assert all(r['sampling_identity']['weight_digest']==state['weight_digest'] for r in vals)
            source=run/'mh'/cycle.name/'harnesses'/path.parents[1].name/'harness.py'
            assert sha(source)==read(path)['metadata']['harness_sha256']
        selected=read(cycle/'mh_result.json');harness=Path(selected['harness'])
        assert sha(harness)==selected['sha256']
        previous=complete['identity']
        training.append({'cycle':n,'accepted_trajectories':state['accepted_trajectories'],
            'updates':state['metrics']['actor/sft_updates'],'response_tokens':state['response_tokens'],
            'selected_harness':selected['selected'],'elapsed_seconds':read(update/'TIMING.json')['elapsed_seconds']})
    states=read(run/'CURVE_STATES.json')
    records={p.parent.name:audit_output(p.parent,'monitor') for p in sorted((run/'monitor').glob('*/val.json'))}
    assert len(records)==7 and all(len(v)==64 for v in records.values())
    for state in states:
        value=read(run/'monitor'/state['label']/'val.json')
        assert value['metadata']['weight_digest']==state['identity']['weight_digest']
        assert value['metadata']['harness_sha256']==sha(state['harness'])
    contrasts={'weight_at_final_harness':contrast(records['W0H2'],records['W2H2']),
        'harness_at_final_weights':contrast(records['W2H0'],records['W2H2']),
        'joint':contrast(records['W0H0'],records['W2H2'])}
    phase_changes=[{'from':a['label'],'to':b['label'],**contrast(records[a['label']],records[b['label']])}
        for a,b in zip(states,states[1:])]
    qualified=all(contrasts[k]['difference_pp']>=5 for k in ['weight_at_final_harness','harness_at_final_weights'])
    assert qualified==read(run/'QUALIFICATION.json')['qualified']
    result={'passed_audit':True,'qualified_on_development':qualified,
        'test_opened':False,'training':training,'scores':{k:{'solved':int(sum(r['reward'] for r in v)),'total':len(v)} for k,v in records.items()},
        'contrasts':contrasts,'phase_changes':phase_changes,'cost':cost,
        'raw_receipts_verified':len(calls_seen),'padding_fix_verified':True,
        'limits':config['limits']}
    write_json(run/'verification.json',result);print(json.dumps(result,indent=2))


if __name__=='__main__':main()

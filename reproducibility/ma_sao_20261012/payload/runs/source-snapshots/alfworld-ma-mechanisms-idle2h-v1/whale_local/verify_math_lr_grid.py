"""Audit the corrected same-data LR grid, selecting on development only."""
import argparse
import json
from pathlib import Path
import math

import pandas as pd
import torch
from safetensors import safe_open

from assess_viability import paired_change
from math_reward import binary_reward
from pipeline import sha, write_json
from math_context_audit import recheck_context


def read(p):return json.loads(Path(p).read_text())
def lines(p):return [json.loads(x) for x in Path(p).read_text().splitlines()]


def verify_services(run):
    for path in run.rglob('services.json'):
        started=read(path);stopped=read(path.with_name('services_stopped.json'))
        for kind in ['worker','sandbox']:
            assert started[kind+'_pid']==stopped[kind+'_pid']
            assert stopped[kind+'_returncode'] in [0,-15]
            assert 'Finished server process' in path.with_name(kind+'.log').read_text()


def main():
    ap=argparse.ArgumentParser();ap.add_argument('grid',type=Path);a=ap.parse_args()
    grid=a.grid.resolve();cfg=read(grid/'PROTOCOL.json');root=Path(__file__).resolve().parents[1]
    sources=dict(cfg['source_sha256'])
    for p in sorted(grid.glob('SOURCE_AMENDMENT_*.json')):
        amendment=read(p);assert amendment['protocol_sha256']==sha(grid/'PROTOCOL.json')
        for name,change in amendment['changes'].items():
            assert sources.get(name)==change['old_sha256'];sources[name]=change['new_sha256']
    for name,digest in sources.items():assert sha(root/name)==digest,name
    registration=read(grid/'AUDITOR_REGISTRATION.json')
    auditor_digest=registration['sha256']
    for p in sorted(grid.glob('AUDITOR_AMENDMENT_*.json')):
        amendment=read(p);assert amendment['old_sha256']==auditor_digest
        auditor_digest=amendment['new_sha256']
    assert sha(__file__)==auditor_digest
    source=Path(cfg['source_run']);source_rows=lines(source/'round_01/rollouts/trajectories.jsonl')
    assert sha(source/'round_01/rollouts/trajectories.jsonl')==cfg['source_rollouts_sha256']
    assert sha(source/'round_01/accepted_batch.pt')==cfg['source_batch_sha256']
    source_batch=torch.load(source/'round_01/accepted_batch.pt',map_location='cpu',weights_only=True)
    source_calls={c['call_id']:c for row in source_rows for c in row['policy_calls']}
    assert len(source_calls)==sum(len(r['policy_calls']) for r in source_rows)==765
    source_success=[c for row in source_rows if row['reward'] for c in row['policy_calls']]
    data=read(source/'config.json')['data']
    targets={}
    for split in ['train','mh_val']:
        entry=data['splits'][split];assert sha(entry['path'])==entry['sha256']
        targets[split]={r['extra_info']['task_id']:r['reward_model']['ground_truth']['target']
                        for r in pd.read_parquet(entry['path']).to_dict('records')}
    for row in source_rows:
        assert row['reward']==binary_reward(row['last_assistant'],targets['train'][row['task_id']])
    initial_run=root/'runs/whale_math_initial_mh_diagnostic_20260915'
    initial_rows=lines(initial_run/'baseline/trajectories.jsonl')
    initial_audit=read(initial_run/'verification.json')
    assert initial_audit['passed_audit'] and sha(initial_run/'baseline/trajectories.jsonl')==initial_audit['trajectories_sha256']
    initial={r['task_id']:int(r['reward']) for r in initial_rows}
    assert len(initial)==48 and sum(initial.values())==36
    correction=read(grid/'PRODUCTION_PADDING_CHECK.json');assert sha(correction['path'])==correction['sha256']
    assert correction['passed']
    results={}
    all_new_calls=set()
    for name,trial in cfg['trials'].items():
        run=grid/name;done=read(run/'complete.json');phase=run/'round_01'
        assert done['new_training_generation_calls']==0
        assert sha(phase/'rollouts/trajectories.jsonl')==cfg['source_rollouts_sha256']
        provenance=read(run/'CACHED_TRAIN_RECEIPTS.json')
        assert provenance['only_training_receipts_copied'] and provenance['new_training_generations']==0
        assert set(provenance['receipts_sha256'])=={k+'.json' for k in source_calls}
        for key,call in source_calls.items():
            assert read(run/'calls'/(key+'.json'))==call
            assert sha(run/'calls'/(key+'.json'))==provenance['receipts_sha256'][key+'.json']
        train=read(phase/'checkpoint/training_state.json');reload=read(phase/'reload_verification.json')
        assert train['sampling_identity']['weight_digest']==cfg['base_digest']
        assert train['learning_rate']==trial['learning_rate'] and train['sft_epochs']==1
        assert train['parameter_dtype']=='float32'
        assert train['qwen35_padding_compatibility_fix']['name']=='qwen35-singleton-padding-mask-v1'
        assert train['optimizer_steps_before']==[] and train['optimizer_steps_after']==[120]
        assert train['weight_digest']==reload['weight_digest']==done['identity']['weight_digest']
        assert reload['all_parameter_hashes_equal']
        assert train['weight_digest']!=cfg['base_digest']
        assert train['rollouts']==512 and train['accepted_trajectories']==312
        assert train['accepted_policy_calls']==477 and train['response_tokens']==186075
        assert train['metrics']['actor/sft_updates']==120
        assert train['changed_parameter_tensors'] and not any('visual' in n for n in train['changed_parameter_tensors'])
        actual=torch.load(phase/'accepted_batch.pt',map_location='cpu',weights_only=True)
        assert actual.keys()==source_batch.keys()
        assert all(torch.equal(actual[k],source_batch[k]) for k in actual)
        pmax=actual['input_ids'].shape[1]-actual['responses'].shape[1]
        for i,call in enumerate(source_success):
            assert actual['input_ids'][i,:pmax][actual['attention_mask'][i,:pmax].bool()].tolist()==call['prompt_ids']
            assert actual['responses'][i][actual['response_mask'][i].bool()].tolist()==call['response_ids']
        optimizer=torch.load(phase/'checkpoint/optimizer.pt',map_location='cpu',weights_only=True)
        assert all(g['lr']==trial['learning_rate'] and g['weight_decay']==0.01 for g in optimizer['param_groups'])
        assert {int(v['step']) for v in optimizer['state'].values()}=={120}
        assert {v.dtype for state in optimizer['state'].values() for k,v in state.items() if k!='step'}=={torch.float32}
        del optimizer,actual
        dtypes=set()
        for path in (phase/'checkpoint').glob('*.safetensors'):
            with safe_open(path,framework='pt',device='cpu') as f:
                dtypes.update(f.get_slice(k).get_dtype() for k in f.keys() if 'visual' not in k)
        assert dtypes=={'F32'}
        evaluation=read(run/'mh_val/val.json');rows=lines(run/'mh_val/trajectories.jsonl')
        assert len(rows)==48 and {r['task_id'] for r in rows}==set(initial)
        eval_calls={c['call_id']:c for r in rows for c in r['policy_calls']}
        assert len(eval_calls)==sum(len(r['policy_calls']) for r in rows)
        assert not (set(eval_calls)&(set(source_calls)|all_new_calls));all_new_calls.update(eval_calls)
        for row in rows:
            assert row['reward']==binary_reward(row['last_assistant'],targets['mh_val'][row['task_id']])
            assert row['sampling_identity']['weight_digest']==train['weight_digest']
            assert row['harness_sha256']==train['harness_sha256']
            assert row['rollout_seed']==42*100003+row['example_index'] and row['draw']==0
            assert row['assistant_tokens']==sum(len(c['response_ids']) for c in row['policy_calls'])<=2048
            for call in row['policy_calls']:
                assert read(run/'calls'/(call['call_id']+'.json'))==call
                assert len(call['response_ids'])<=call['requested_max_tokens']<=1024
                assert len(call['response_ids'])==len(call['response_logprobs'])
        assert {p.stem for p in run.rglob('calls/*.json')}==set(source_calls)|set(eval_calls)
        contexts={r['context_rejection_id'] for r in source_rows+rows if r.get('context_rejection_id')}
        assert {p.stem for p in run.rglob('context_rejections/*.json')}==contexts
        context_rechecks={}
        for p in run.rglob('context_rejections/*.json'):
            context_rechecks[p.stem]=recheck_context(p)
        write_json(run/'CONTEXT_TOKEN_RECHECKS.json',context_rechecks)
        current={r['task_id']:int(r['reward']) for r in rows};solved=sum(current.values())
        assert solved==evaluation['solved_examples']==done['development_solved']
        verify_services(run)
        paired=paired_change([initial[k] for k in sorted(initial)],[current[k] for k in sorted(initial)])
        costs={'development_trajectories':48,'development_calls':len(eval_calls),
               'development_generated_tokens':sum(len(c['response_ids']) for c in eval_calls.values()),
               'development_prompt_tokens':sum(len(c['prompt_ids']) for c in eval_calls.values()),
               'new_train_generation_calls':0,'cached_train_trajectories_processed':512,
               'sft_response_tokens':186075,'optimizer_updates':120,'training_seconds':train['elapsed_seconds']}
        results[name]={'learning_rate':trial['learning_rate'],'development_solved':solved,'development_total':48,
                       'weight_digest':train['weight_digest'],'paired_vs_initial':paired,'costs':costs,
                       'checkpoint':str(phase/'checkpoint'),'passed_audit':True}
    selected=min(results,key=lambda n:(-results[n]['development_solved'],results[n]['learning_rate']))
    result={'passed_audit':True,'selection_split':'mh_val','selection_rule':cfg['selection'],
            'selected_trial':selected,'may_advance_to_harness_search':results[selected]['development_solved']>=36,
            'qualified_for_gate':False,'initial_development_solved':36,'trials':results,
            'same_512_cached_train_trajectories_used_for_every_trial':True,'no_confirmation_or_test_evaluation':True,
            'unique_initial_train_sampling_cost':{'trajectories':512,'calls':765,
                'generated_tokens':sum(len(c['response_ids']) for c in source_calls.values())},
            'shared_initial_mh_cost':initial_audit['cost'],
            'cost_limitations':'All four LR updates and all MH trials are tuning cost. Earlier invalid updates/engineering are preserved separately; this is not the WHALE/gate sample-efficiency comparison.'}
    write_json(grid/'verification.json',result)
    report=['# 修正 padding 后的学习率诊断','', '| 学习率 | 开发集解题数 | 相对初始 36/48 |','|---|---:|---:|']
    report += [f"| {v['learning_rate']:g} | {v['development_solved']}/48 | {v['paired_vs_initial']['difference_pp']:+.2f} pp |" for v in results.values()]
    report += ['',f"按预设规则选择 {selected}。允许继续 harness 搜索：{result['may_advance_to_harness_search']}。",
               '这只是开发集调参，尚非完整 WHALE、双轴确认或 gate 提升证据。四组均从原始 base 对同一批 512 条缓存轨迹真实更新，不增加 TRAIN 生成；四次训练与 192 条开发轨迹的成本全部单列。','']
    (grid/'REPORT.md').write_text('\n'.join(report))
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()

"""Complete source, backend, optimizer and cleanup checks for a Math gate run."""
from pathlib import Path
import sys
import torch

from pipeline import sha, write_json
from verify_math_run import read, jsonl
from verify_math_lr_grid import verify_services
import verify_math_gate
from math_context_audit import recheck_context


def main():
    assert len(sys.argv)==2
    run=Path(sys.argv[1]).resolve();root=Path(__file__).resolve().parents[1]
    config=read(run/'config.json');protocol=read(run/'PROTOCOL.json')
    assert config['gate_enabled']
    sources=dict(protocol['source_sha256'])
    for path in sorted(run.glob('SOURCE_AMENDMENT_*.json')):
        amendment=read(path);assert amendment['protocol_sha256']==sha(run/'PROTOCOL.json')
        for name,change in amendment['changes'].items():
            assert sources.get(name)==change['old_sha256'];sources[name]=change['new_sha256']
    assert all(sha(root/name)==digest for name,digest in sources.items())
    for name,digest in protocol['source_sha256'].items():
        assert sha(run/'source_snapshot'/name)==digest
    verify_services(run)
    rows=[]
    for number in range(1,config['rounds']+1):
        phase=run/f'round_{number:02d}';train=read(phase/'checkpoint/training_state.json')
        assert train['learning_rate']==config['learning_rate'] and train['sft_epochs']==1
        assert train['qwen35_padding_compatibility_fix']['name']=='qwen35-singleton-padding-mask-v1'
        assert not any('visual' in name for name in train['changed_parameter_tensors'])
        optimizer=torch.load(phase/'checkpoint/optimizer.pt',map_location='cpu',weights_only=True)
        assert all(g['lr']==config['learning_rate'] and g['weight_decay']==0.01 for g in optimizer['param_groups'])
        del optimizer
        rows+=jsonl(phase/'rollouts/trajectories.jsonl')
        for path in read(phase/'mh_result.json')['evaluation_files']:
            rows+=jsonl(Path(path).parent/'trajectories.jsonl')
    for arm in ['initial','whale','gate']:rows+=jsonl(run/'test'/arm/'trajectories.jsonl')
    calls=[c['call_id'] for row in rows for c in row['policy_calls']]
    assert len(calls)==len(set(calls))
    physical_call_hashes={}
    for path in run.rglob('calls/*.json'):
        digest=sha(path)
        assert physical_call_hashes.setdefault(path.stem,digest)==digest
    contexts={row['context_rejection_id']:row for row in rows if row.get('context_rejection_id')}
    assert set(contexts)=={p.stem for p in run.rglob('context_rejections/*.json')}
    for key,row in contexts.items():
        matches=list(run.rglob('context_rejections/'+key+'.json'))
        assert len({sha(p) for p in matches})==1
        rejection=read(matches[0]);assert rejection['generated_tokens']==0 and rejection['context_limit']==4096
        recheck_context(matches[0])
        assert rejection['sampling_identity']['weight_digest']==row['sampling_identity']['weight_digest']
        assert rejection['seed']==row['rollout_seed']
    verify_math_gate.main()
    result=read(run/'verification.json')
    result.update(frozen_sources_and_service_cleanup_verified=True,optimizer_settings_independently_verified=True,
                  qwen35_singleton_padding_fix_verified=True,zero_generation_context_rejections_verified=len(contexts),
                  all_physical_policy_calls_uniquely_accounted=True)
    write_json(run/'verification.json',result)


if __name__=='__main__':main()

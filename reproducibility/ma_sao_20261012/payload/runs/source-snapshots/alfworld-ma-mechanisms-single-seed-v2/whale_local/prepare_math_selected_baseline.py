"""Import an audited, development-selected first update into a fresh WHALE run.

Copies immutable evidence; never generates samples or evaluates held-out data.
The selected training stage and h0 MH evaluation count in the baseline budget.
"""
import argparse
import datetime
import json
from pathlib import Path
import shutil

from pipeline import sha, write_json


def read(p):return json.loads(Path(p).read_text())


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--grid',type=Path,required=True)
    ap.add_argument('--run-dir',type=Path,required=True);a=ap.parse_args()
    grid=a.grid.resolve();run=a.run_dir.resolve();root=Path(__file__).resolve().parents[1]
    assert not run.exists(), 'Use a fresh baseline directory'
    audit=read(grid/'verification.json')
    assert audit['passed_audit'] and audit['may_advance_to_harness_search']
    assert audit['selection_split']=='mh_val'
    selected=audit['selected_trial'];parent=grid/selected
    grid_cfg=read(grid/'PROTOCOL.json');source=Path(grid_cfg['source_run'])
    trial=grid_cfg['trials'][selected];lr=trial['learning_rate']
    phase=run/'round_01';phase.mkdir(parents=True)
    files={}
    def copy(src,dest):
        dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,dest)
        assert sha(src)==sha(dest)
        files[str(dest.relative_to(run))]={'source':str(src),'sha256':sha(src)}
    for sub in ['checkpoint','rollouts']:
        for src in sorted((parent/'round_01'/sub).rglob('*')):
            if src.is_file():copy(src,phase/sub/src.relative_to(parent/'round_01'/sub))
    for name in ['accepted_batch.pt','training_result.json']:
        copy(parent/'round_01'/name,phase/name)
    model=read(parent/'trained_identity.json')['model']
    mh_output=run/'mh/round_01/logs/mh_val/h0'/(model.lower()+'_seed42')
    for src in sorted((parent/'mh_val').rglob('*')):
        if src.is_file():copy(src,mh_output/src.relative_to(parent/'mh_val'))
    rows=[json.loads(x) for x in (phase/'rollouts/trajectories.jsonl').read_text().splitlines()]
    rows += [json.loads(x) for x in (mh_output/'trajectories.jsonl').read_text().splitlines()]
    for row in rows:
        for call in row['policy_calls']:
            name=call['call_id']+'.json';copy(parent/'calls'/name,run/'calls'/name)
        if row.get('context_rejection_id'):
            name=row['context_rejection_id']+'.json';copy(parent/'context_rejections'/name,run/'context_rejections'/name)
    hardware=read(source/'eval_hardware.json')
    if (grid/'baseline_hardware_override.json').exists():
        hardware=read(grid/'baseline_hardware_override.json')
    write_json(run/'eval_hardware.json',hardware)
    old_protocol=read(source/'PROTOCOL.json')
    sources=list(old_protocol['source_sha256'])+['whale_local/qwen35_padding.py',
        'whale_local/prepare_math_selected_baseline.py','whale_local/verify_math_lr_grid.py',
        'whale_local/verify_math_selected_run.py','whale_local/math_context_audit.py']
    protocol={k:v for k,v in old_protocol.items() if k not in ['created_utc','status','source_sha256','formal_start_weights','engineering_trials_excluded_from_formal_budget_but_preserved']}
    protocol.update(created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        status='registered_after_MH_only_LR_selection_before_harness_search_and_confirmation',
        learning_rate=lr,source_sha256={name:sha(root/name) for name in sources},
        formal_start_weights='original base; selected first RSFT update is reused with its exact same-data audit and counted in the formal budget',
        backend_correction='qwen35-singleton-padding-mask-v1',
        learning_rate_selection={'grid':str(grid),'verification_sha256':sha(grid/'verification.json'),
            'selected_trial':selected,'selection_split':'mh_val','same_cached_train_data_in_every_trial':True},
        additional_tuning_cost={'nonselected_complete_trials':[name for name in audit['trials'] if name!=selected],
            'selected_trial_training_and_h0_MH_overlap_baseline_budget':True,
            'initial_W0H0_development_reference':audit['shared_initial_mh_cost'],
            'all_invalid_prior_attempts_preserved_separately':True},
        scope_limitations=[s.replace('14 frozen inference replicas','Available frozen inference replicas')
            for s in old_protocol['scope_limitations'] if 'LR=1e-5' not in s])
    protocol['hardware']=hardware
    write_json(run/'PROTOCOL.json',protocol)
    reuse={'kind':'development-selected first training stage plus identical h0 MH evaluation',
        'grid':str(grid),'grid_audit_sha256':sha(grid/'verification.json'),'selected_trial':selected,
        'parent_run':str(parent),'reused_training_rounds':1,'reused_training_trajectories':512,
        'reused_h0_mh_trajectories':48,'new_generations_in_import':0,'files':files,
        'no_confirmation_or_test_from_parent':True,'counts_in_baseline_training_and_MH_budget':True}
    write_json(run/'TRAINING_STAGE_REUSE.json',reuse)
    for name,digest in protocol['source_sha256'].items():
        src=root/name;dest=run/'source_snapshot'/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,dest)
    (run/'PROTOCOL.md').write_text('# 修正后的两轮原生 WHALE 基线\n\n从原始 base 开始；开发集选择的首轮 RSFT checkpoint、512 条训练轨迹与原始 harness 的 48 条 MH 轨迹逐文件复用，并计入本次预算。其余学习率尝试的训练与 MH 成本单列。\n\n随后完成原生两轮交替流程：每轮总计 512 条训练轨迹、4 个新 harness 候选各 48 道 MH 题；权重冻结后首次评测 128 道独立确认题的四种组合。harness 与权重的条件增益都至少 5 个百分点才启动 gate。未查看确认或测试结果来选择学习率。\n')
    command=[str(root/'.venv-whale-search/bin/python'),'-u',str(root/'whale_local/math_pipeline.py'),
        '--run-dir',str(run),'--manifest',str(root/'runs/whale_math_setup_20260915/manifest.json'),
        '--gpu','8','--policy-port','30041','--proposer-port','30042','--rounds','2',
        '--rollouts','4','--learning-rate',str(lr),'--parameter-dtype','float32','--resume']
    write_json(run/'launch_command.json',{'argv':command,'PYTHONPATH':str(root/'whale_local')+':'+str(root/'WHALE/domains/math_reasoning')})
    print(json.dumps({'run':str(run),'selected_trial':selected,'learning_rate':lr,'launch_argv':command},indent=2))


if __name__=='__main__':main()

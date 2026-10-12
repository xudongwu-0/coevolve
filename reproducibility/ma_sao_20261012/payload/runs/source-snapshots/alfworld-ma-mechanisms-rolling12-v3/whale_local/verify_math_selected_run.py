"""Audit the selected first-stage reuse before the complete Math WHALE audit."""
import json
from pathlib import Path
import sys

from pipeline import sha, write_json
import verify_math_run


def read(p):return json.loads(Path(p).read_text())


def main():
    assert len(sys.argv)==2
    run=Path(sys.argv[1]).resolve();reuse=read(run/'TRAINING_STAGE_REUSE.json')
    grid=Path(reuse['grid']);audit=read(grid/'verification.json')
    assert sha(grid/'verification.json')==reuse['grid_audit_sha256']
    assert audit['passed_audit'] and audit['may_advance_to_harness_search']
    assert audit['selection_split']=='mh_val' and audit['selected_trial']==reuse['selected_trial']
    assert reuse['reused_training_rounds']==1 and reuse['new_generations_in_import']==0
    assert reuse['reused_training_trajectories']==512 and reuse['reused_h0_mh_trajectories']==48
    assert reuse['no_confirmation_or_test_from_parent'] and reuse['counts_in_baseline_training_and_MH_budget']
    for relative,entry in reuse['files'].items():
        assert sha(run/relative)==sha(entry['source'])==entry['sha256'],relative
    config=read(run/'config.json');selected=audit['trials'][reuse['selected_trial']]
    assert selected['learning_rate']==config['learning_rate']
    assert read(run/'round_01/checkpoint/training_state.json')['weight_digest']==selected['weight_digest']
    verify_math_run.main()
    result=read(run/'verification.json')
    result['development_selected_training_stage_reuse_verified']=True
    result['training_stage_reuse']=reuse
    result['additional_nonselected_LR_tuning_costs']={n:v['costs'] for n,v in audit['trials'].items() if n!=reuse['selected_trial']}
    result['selected_LR_stage_overlap_with_baseline']={'training_trajectories':512,'h0_mh_trajectories':48,
        'already_in_baseline_stage_costs':True,'do_not_double_count':True}
    result['additional_shared_initial_mh_cost']=audit['shared_initial_mh_cost']
    result['known_invalid_prior_training_costs_preserved_in']=[str(grid.parent/'whale_math_native_20260915'),
        str(grid.parent/'whale_math_lr_grid_20260915')]
    write_json(run/'verification.json',result)
    with (run/'REPORT.md').open('a') as f:
        f.write('\n首轮训练和原始 harness 开发评测逐文件复用已审计的学习率选择结果，包含在正式预算中；不是新增的免费采样。\n')
        f.write('其他三个学习率的完整训练和开发评测属于额外调参成本：'+json.dumps(result['additional_nonselected_LR_tuning_costs'],ensure_ascii=False)+'。\n')
        f.write('另计初始模型开发基准、修正前的无效训练及工程调试；详见对应目录。本实验没有将修复 padding 缺陷带来的变化称作 gate 的算法收益。\n')
    print(json.dumps({'passed_audit':result['passed_audit'],'qualified_for_gate':result['qualified_for_gate'],
                      'training_stage_reuse_verified':True},indent=2))


if __name__=='__main__':main()

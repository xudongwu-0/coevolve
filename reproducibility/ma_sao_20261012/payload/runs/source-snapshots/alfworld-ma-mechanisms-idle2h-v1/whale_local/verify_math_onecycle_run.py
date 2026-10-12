"""Audit the preregistered first-cycle fallback and account for shared history."""
from pathlib import Path
import sys
from pipeline import sha, write_json
from verify_math_run import read
import verify_math_run


def main():
    assert len(sys.argv) == 2
    run = Path(sys.argv[1]).resolve()
    prefix = read(run / 'prefix_reuse.json')
    parent = Path(prefix['parent_run'])
    parent_audit = read(parent / 'verification.json')
    assert parent_audit['passed_audit'] and not parent_audit['qualified_for_gate']
    selection = read(run / 'development_selection.json')
    assert sha(selection['conditional_protocol']) == selection['conditional_protocol_sha256']
    assert sha(selection['registration_timing']) == selection['registration_timing_sha256']
    timing = read(selection['registration_timing'])
    assert timing['registered_timestamp'] < timing['final_MH_comparison_timestamp']
    assert timing['conditional_protocol_sha256'] == selection['conditional_protocol_sha256']
    assert sha(timing['final_MH_comparison_path']) == timing['final_MH_comparison_sha256']
    conditional = read(selection['conditional_protocol'])
    assert conditional['candidate_weights_digest'] == read(run / 'round_01/checkpoint/training_state.json')['weight_digest']
    assert conditional['candidate_harness_sha256'] == read(run / 'round_01/mh_result.json')['harness_sha256']
    assert read(run / 'config.json')['rounds'] == 1 and prefix['completed_rounds'] == 1
    relocation = read(run / 'MH_PATH_RELOCATION.json')
    assert relocation['original'] == read(parent / 'round_01/mh_result.json')
    assert relocation['relocated'] == read(run / 'round_01/mh_result.json')
    expected = dict(relocation['original'])
    expected['harness_path'] = str(run / Path(expected['harness_path']).relative_to(parent))
    expected['evaluation_files'] = [str(run / Path(p).relative_to(parent)) for p in expected['evaluation_files']]
    assert expected == relocation['relocated']
    verify_math_run.main()
    result = read(run / 'verification.json')
    additional = {}
    for stage in ['training', 'harness_search']:
        additional[stage] = {key: value - result['costs'][stage][key]
                             for key, value in parent_audit['costs'][stage].items()}
        assert all(value >= 0 for value in additional[stage].values())
    additional['primary_confirmation'] = parent_audit['costs']['confirmation']
    result.update(preregistered_conditional_first_cycle_verified=True,
        prefix_overlap_with_parent={'training_trajectories': 512, 'MH_trajectories': 240,
                                    'counted_in_both_logical_budgets': True, 'generated_once_physically': True},
        additional_primary_search_costs=additional,
        additional_nonselected_LR_tuning_costs=parent_audit.get('additional_nonselected_LR_tuning_costs', {}),
        additional_shared_initial_mh_cost=parent_audit.get('additional_shared_initial_mh_cost', {}),
        parent_development_selected_training_stage_reuse_verified=parent_audit.get(
            'development_selected_training_stage_reuse_verified', False))
    write_json(run / 'verification.json', result)
    with (run / 'REPORT.md').open('a') as f:
        f.write('\n这是一轮较小的原生 WHALE 停止点，按主实验确认前登记的条件启动。'
                '第一轮训练和全部 MH 逐文件复用；缓存不算算法节省。'
                '两轮主实验的第二轮训练、MH 和原确认集属于额外场景筛选成本，见 verification.json。\n')
    print({'passed_audit': result['passed_audit'], 'qualified_for_gate': result['qualified_for_gate']})


if __name__ == '__main__':
    main()

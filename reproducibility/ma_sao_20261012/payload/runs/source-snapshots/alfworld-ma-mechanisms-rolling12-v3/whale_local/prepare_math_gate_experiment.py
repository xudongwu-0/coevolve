"""Prepare a matched-budget gate only after independent mean@8 qualification."""
import argparse
import datetime
from pathlib import Path
import shutil

from pipeline import ROOT, sha, write_json
from verify_math_run import read
from math_common_probe import VERSION


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--run-dir', type=Path, required=True)
    args = parser.parse_args()
    reference, run = args.reference.resolve(), args.run_dir.resolve()
    audit, config, summary = (read(reference / name) for name in
                              ['verification.json', 'config.json', 'summary.json'])
    assert audit['passed_audit'] and audit['qualified_for_gate']
    assert audit['native_prefix_exact_reuse_verified'] and audit['fp32_master_and_moments']
    assert summary['status'] == 'complete' and not summary['gate_enabled']
    assert config['rounds'] == 1 and config['rollouts_per_task'] == 4
    assert config['evaluation_rollouts'] == 8 and config['evaluation_temperature'] == 1.0
    assert config['parameter_dtype'] == 'float32'
    assert config['data']['splits']['train']['count'] == 128
    assert config['data']['splits']['mh_val']['count'] == 48
    assert config['data']['splits']['test']['count'] == 128
    assert config['mh_iterations'] == 2 and config['mh_proposals'] == 2
    assert audit['independent_gate_test_not_evaluated']
    for entry in config['data']['splits'].values():
        assert sha(entry['path']) == entry['sha256']
    training = read(reference / 'round_01/checkpoint/training_state.json')
    assert training['weight_digest'] == summary['final_identity']['weight_digest']
    assert training['metrics']['actor/sft_updates'] == (training['accepted_policy_calls'] + 3) // 4
    assert not run.exists(), 'Use a fresh gate run directory'
    checks = ROOT / 'audit/whale/MATH_GATE_INTERFACE_CHECKS_20260915.json'
    tested = read(checks)
    assert tested['passed']
    for name, digest in tested['source_sha256'].items():
        assert sha(ROOT / 'whale_local' / name) == digest
    metric_checks = ROOT / 'audit/whale/MATH_MEAN8_AGGREGATION_CHECK_20260915.json'
    metric_tested = read(metric_checks)
    assert metric_tested['passed']
    for name, digest in metric_tested['source_sha256'].items():
        assert sha(ROOT / 'whale_local' / name) == digest
    launcher_checks = ROOT / 'audit/whale/MATH_GATE_LAUNCHER_CHECK_20260915.json'
    launcher_tested = read(launcher_checks)
    assert launcher_tested['passed']
    for name, digest in launcher_tested['source_sha256'].items():
        assert sha(ROOT / 'whale_local' / name) == digest
    run.mkdir(parents=True)
    manifest = run / 'data_manifest.json'
    write_json(manifest, config['data'])
    hardware = {'sampler_gpus': [6, 7, 9, 'node1:4', 'node1:5', 'node1:6', 'node1:7', 'node1:8', 'node1:9'],
        'sampler_port_base': 21000, 'mh_replica_port_base': 23000,
        'confirmation_gpus': [6, 7], 'confirmation_port_base': 25000,
        'confirmation_replica_port_base': 27000,
        'confirmation_replica_groups': [['node1:4', 'node1:5'], ['node1:6', 'node1:7'],
                                        [9, 'node1:8', 'node1:9']]}
    write_json(run / 'eval_hardware.json', hardware)
    names = set(read(reference / 'PROTOCOL.json')['source_sha256'])
    names.update('whale_local/' + name for name in [
        'math_pipeline.py', 'math_evaluation.py', 'math_gate.py', 'math_common_probe.py',
        'run_math_mh.py', 'verify_math_gate.py', 'verify_math_gate_complete.py',
        'prepare_math_gate_experiment.py', 'check_math_gate_interfaces.py',
        'check_math_mean8_aggregation.py', 'run_math_gate_experiment.py'])
    protocol = {'created_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'scope': 'Single-cycle sampling-allocation gate with matched native RSFT and harness proposal pool',
        'reference': str(reference), 'qualification_sha256': sha(reference / 'verification.json'),
        'reference_protocol_sha256': sha(reference / 'PROTOCOL.json'),
        'interface_checks_sha256': sha(checks), 'metric_checks_sha256': sha(metric_checks),
        'launcher_checks_sha256': sha(launcher_checks),
        'data_manifest_sha256': sha(manifest),
        'source_sha256': {name: sha(ROOT / name) for name in sorted(names)},
        'hardware': hardware, 'training_trajectories_per_arm': 512, 'MH_trajectories_per_arm': 240,
        'first_probe_mode': VERSION, 'shared_initial_training_trajectories': 128,
        'shared_probe_counts_in_each_logical_budget': True,
        'shared_probe_is_not_algorithmic_compute_saving': True,
        'followup_gate_training_trajectories': 384,
        'teacher': 'active Codex through the local FastAPI queue; evidence-only routing hypotheses',
        'teacher_token_usage_known': False,
        'routing_mass': {'weight_or_both': 2.0, 'success_or_uncertain': 1.0,
                         'harness_scoring_or_environment': .25},
        'confidence_threshold': .7, 'max_draws_per_task': 8,
        'retain_all_verified_successes': True, 'harness_proposal_mode': 'reference_pool',
        'final_test_questions': 128, 'final_test_draws_per_question': 8,
        'final_test_temperature': 1.0, 'final_test_arms': ['initial', 'whale', 'gate'],
        'planned_test_trajectories': 3072, 'bootstrap_unit': 'question',
        'no_test_based_tuning': True, 'prior_scenario_screening_costs_retained': True}
    write_json(run / 'PROTOCOL.json', protocol)
    for name, digest in protocol['source_sha256'].items():
        target = run / 'source_snapshot' / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / name, target)
        assert sha(target) == digest
    command = [str(ROOT / '.venv-whale-search/bin/python'), '-u', str(ROOT / 'whale_local/math_pipeline.py'),
        '--run-dir', str(run), '--manifest', str(manifest), '--gpu', '8',
        '--policy-port', '30061', '--proposer-port', '30062', '--rounds', '1', '--rollouts', '4',
        '--mh-iterations', '2', '--mh-proposals', '2', '--learning-rate', str(config['learning_rate']),
        '--parameter-dtype', 'float32', '--gate-reference', str(reference), '--share-first-probe',
        '--evaluation-temperature', '1.0', '--evaluation-rollouts', '8', '--resume']
    write_json(run / 'launch_command.json', {'argv': command,
        'PYTHONPATH': str(ROOT / 'whale_local') + ':' + str(ROOT / 'WHALE/domains/math_reasoning')})
    write_json(run / 'STATUS.json', {'stage': 'prepared_qualified_gate', 'teacher_labels_supplied': False,
        'training_started': False, 'test_used': False})
    (run / 'PROTOCOL.md').write_text(
        '# 同样本预算 Math gate 对照\n\n独立 mean@8 双轴确认通过后启动。'
        'WHALE/gate 各 512 条训练轨迹、240 条 MH 轨迹；全部成功样本进入同一原生 RSFT。'
        '首批 128 条显式共享并计入两臂预算，缓存不算算法节省。外部 Codex 只给失败类型标签，'
        '不提供正确答案或训练目标；其余 384 条按标签进行软分配。\n\n'
        '复用相同 harness 候选池，按原生规则重新评分选择。最终 128 题对三臂分别采样 8 次，'
        '报告 mean@8、按题配对区间及全部 token/更新开销。测试结果不用于后续调参。\n')
    print({'prepared': str(run), 'qualification_verified': True, 'new_generations': 0})


if __name__ == '__main__':
    main()

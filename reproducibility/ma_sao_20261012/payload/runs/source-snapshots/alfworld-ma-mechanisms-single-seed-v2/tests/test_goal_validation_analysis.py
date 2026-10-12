from pathlib import Path
import sys
import copy
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'whale_local'))
from sao_bridge.goal_validation_analysis import analyze_verification, VERIFICATION_SEEDS
from sao_bridge.evaluation import stable_id_seed_offset


def report(seed, rewards):
    rows = [dict(id=f'alfworld-{i}', task_id=i, sample_index=0,
                 rollout_seed=seed + stable_id_seed_offset(f'alfworld-{i}'),
                 reward=v, tokens=10) for i, v in enumerate(rewards)]
    return dict(results=rows, accuracy=sum(rewards)/len(rewards),
                question_count=len(rewards), samples_per_question=1, metric='mean@1')


def test_bootstrap_keeps_both_seeds_of_each_question_together():
    a, b = VERIFICATION_SEEDS
    control = {s: report(s, [0, 0]) for s in VERIFICATION_SEEDS}
    candidate = {a: report(a, [1, 0]), b: report(b, [0, 1])}
    result = analyze_verification(control, candidate, [0, 1])
    assert result['pooled']['paired_question_bootstrap_95pct'] == [.5, .5]
    assert result['pooled']['candidate_accuracy'] == .5
    assert not result['inference_robustness_gate_passed']


def test_inference_pass_does_not_claim_overall_goal_complete():
    control = {s: report(s, [0, 0]) for s in VERIFICATION_SEEDS}
    candidate = {s: report(s, [1, 1]) for s in VERIFICATION_SEEDS}
    result = analyze_verification(control, candidate, [0, 1])
    assert result['inference_robustness_gate_passed']
    assert not result['goal_complete']
    assert len(result['per_seed']) == 2


def test_rejects_missing_seed_and_mismatched_native_seed():
    reports = {s: report(s, [1, 0]) for s in VERIFICATION_SEEDS}
    with pytest.raises(ValueError, match='prespecified'):
        analyze_verification({VERIFICATION_SEEDS[0]: reports[VERIFICATION_SEEDS[0]]}, reports, [0, 1])
    bad = copy.deepcopy(reports)
    bad[VERIFICATION_SEEDS[1]]['results'][0]['rollout_seed'] += 1
    with pytest.raises(ValueError, match='seeds'):
        analyze_verification(reports, bad, [0, 1])


def test_rejects_duplicate_questions_and_missing_coverage():
    reports = {s: report(s, [1, 0]) for s in VERIFICATION_SEEDS}
    with pytest.raises(ValueError, match='unique'):
        analyze_verification(reports, reports, [0, 0])
    with pytest.raises(ValueError, match='coverage'):
        analyze_verification(reports, reports, [0, 1, 2])

"""Analyze fixed verification seeds with questions as bootstrap clusters."""
from .alfworld_strong_phase import checked_native_report
from .analysis import paired_difference


VERIFICATION_SEEDS = (20261005, 20261006)


def analyze_verification(control, candidate, task_ids, *, solver_tokens=4096,
                         bootstrap_seed=20261080, draws=10000):
    """Inference gate only; never substitute this for training robustness."""
    if not task_ids or len(set(task_ids)) != len(task_ids):
        raise ValueError('Verification requires unique question IDs')
    if set(control) != set(VERIFICATION_SEEDS) or set(candidate) != set(VERIFICATION_SEEDS):
        raise ValueError('Both prespecified verification seeds are required')
    per_seed = {}
    for i, seed in enumerate(VERIFICATION_SEEDS):
        evaluation = dict(samples=1, seed=seed, solver_tokens=solver_tokens)
        for reports in (control, candidate):
            checked_native_report(reports[seed], task_ids, evaluation)
        per_seed[str(seed)] = paired_difference(
            control[seed], candidate[seed], seed=bootstrap_seed + i + 1, draws=draws)

    def pool(reports):
        # Preserve each native identity. paired_difference averages within question
        # before bootstrap, so seeds of the same question are never independent draws.
        return dict(question_count=len(task_ids), samples_per_question=2,
                    metric='mean_over_verification_seeds',
                    results=[row for seed in VERIFICATION_SEEDS
                             for row in reports[seed]['results']])

    pooled = paired_difference(pool(control), pool(candidate),
                               seed=bootstrap_seed, draws=draws)
    inference_pass = (pooled['candidate_accuracy'] > .92
                      and pooled['paired_question_bootstrap_95pct'][0] > 0)
    return dict(per_seed=per_seed, pooled=pooled,
                inference_robustness_gate_passed=inference_pass,
                goal_complete=False,
                remaining_requirements=['Frozen primary endpoint must exceed92%.',
                                        'Successful method must replicate at a new training seed.',
                                        'Model/harness provenance, runtime fairness and costs require audit.'],
                limitations=['Reused development questions; not unseen-task generalization.',
                             'Question-cluster bootstrap conditional on two fixed inference seeds; '
                             'not uncertainty over arbitrary inference or training seeds.'])

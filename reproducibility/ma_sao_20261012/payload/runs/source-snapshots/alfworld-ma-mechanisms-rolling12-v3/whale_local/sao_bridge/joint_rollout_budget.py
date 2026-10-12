"""Bound native asynchronous submission to a finite, paired episode ledger."""
from __future__ import annotations


class JointRolloutBudget:
    """Keep native queue/staleness controls, but never wrap the joint ledger.

    Joint episodes cannot be rejected/replaced: each input identifies one shared
    environment round. This adapter is installed on the training dispatcher only.
    """

    def __init__(self, dispatcher, total: int):
        if type(total) is not int or total <= 0:
            raise ValueError("Joint rollout budget must be a positive integer")
        self.dispatcher = dispatcher
        self.total = total
        self.submitted = 0
        self.delivered = 0

    def active_submit_and_wait(self, input_generator, batch_size, dynamic_bs=False):
        if dynamic_bs:
            raise ValueError("Finite joint rollouts do not support rejection/replacement")
        if batch_size <= 0 or self.delivered + batch_size > self.total:
            raise ValueError("Requested batch exceeds the joint rollout budget")
        dispatcher = self.dispatcher
        if dispatcher.runner.max_queue_size < batch_size:
            raise ValueError("Inference queue is smaller than the joint batch")
        results = []
        while len(results) < batch_size:
            with dispatcher._input_cv:
                pending = len(dispatcher._pending_inputs)
            capacity = min(
                dispatcher.staleness_manager.get_pending_limit() - pending,
                dispatcher.runner.max_queue_size - (
                    dispatcher.runner.get_input_queue_size() + batch_size),
                self.total - self.submitted,
                batch_size,
            )
            for _ in range(max(0, capacity)):
                try:
                    task = next(input_generator)
                except StopIteration:
                    raise RuntimeError("Joint input ledger ended before its declared budget") from None
                dispatcher.submit_task_input(task)
                self.submitted += 1
            try:
                arrived = dispatcher.wait_results(count=batch_size - len(results), timeout=1)
            except TimeoutError:
                continue
            if any(result is None for result in arrived):
                raise RuntimeError("Joint episode rejected; replacement would violate paired accounting")
            results.extend(arrived)
        self.delivered += len(results)
        return results


def install_joint_rollout_budget(rollout, total: int) -> JointRolloutBudget:
    # Single-controller uses RolloutController; the local engine owns an executor.
    owner = getattr(rollout, "workflow_executor", rollout)
    dispatcher = owner.dispatcher
    if isinstance(getattr(dispatcher.active_submit_and_wait, "__self__", None), JointRolloutBudget):
        raise ValueError("Joint rollout budget is already installed")
    budget = JointRolloutBudget(dispatcher, total)
    dispatcher.active_submit_and_wait = budget.active_submit_and_wait
    return budget

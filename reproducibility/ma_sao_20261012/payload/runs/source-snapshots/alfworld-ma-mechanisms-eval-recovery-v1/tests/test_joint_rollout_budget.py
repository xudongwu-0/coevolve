from contextlib import nullcontext
from itertools import cycle
from pathlib import Path
from types import SimpleNamespace
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "whale_local"))
from sao_bridge.joint_rollout_budget import JointRolloutBudget, install_joint_rollout_budget


class Dispatcher:
    def __init__(self, *, reject=False):
        self._input_cv = nullcontext()
        self._pending_inputs = []
        self.runner = SimpleNamespace(max_queue_size=32, get_input_queue_size=lambda: 0)
        self.staleness_manager = SimpleNamespace(get_pending_limit=lambda: 16)
        self.submissions = []
        self.ready = []
        self.waits = 0
        self.reject = reject
        self.active_submit_and_wait = None

    def submit_task_input(self, task):
        self.submissions.append(task)
        self.ready.append(task)

    def wait_results(self, count, timeout):
        self.waits += 1
        if self.waits == 1:
            # Force overlap: more than one batch is submitted before delivery.
            raise TimeoutError
        assert len(self.ready) >= count
        result, self.ready = self.ready[:count], self.ready[count:]
        return [None] * count if self.reject else result


def test_final_batch_drains_without_wrapping_cyclic_data():
    dispatcher = Dispatcher()
    budget = JointRolloutBudget(dispatcher, 24)
    data = cycle(range(24))
    delivered = [x for _ in range(3) for x in budget.active_submit_and_wait(data, 8)]
    assert delivered == list(range(24))
    assert dispatcher.submissions == list(range(24))
    assert budget.submitted == budget.delivered == 24
    assert not dispatcher.ready
    with pytest.raises(ValueError, match="exceeds"):
        budget.active_submit_and_wait(data, 8)


def test_rejected_or_short_ledger_cannot_be_replaced():
    with pytest.raises(RuntimeError, match="rejected"):
        JointRolloutBudget(Dispatcher(reject=True), 8).active_submit_and_wait(cycle(range(8)), 8)
    with pytest.raises(RuntimeError, match="ended"):
        JointRolloutBudget(Dispatcher(), 8).active_submit_and_wait(iter(range(4)), 8)
    with pytest.raises(ValueError, match="rejection"):
        JointRolloutBudget(Dispatcher(), 8).active_submit_and_wait(cycle(range(8)), 8, True)


def test_install_on_controller_and_local_engine_without_global_patch():
    for local in (False, True):
        dispatcher = Dispatcher()
        controller = SimpleNamespace(dispatcher=dispatcher)
        rollout = SimpleNamespace(workflow_executor=controller) if local else controller
        budget = install_joint_rollout_budget(rollout, 8)
        assert dispatcher.active_submit_and_wait.__self__ is budget
        with pytest.raises(ValueError, match="already installed"):
            install_joint_rollout_budget(rollout, 8)

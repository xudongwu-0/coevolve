"""Count save frequency per optimizer step, not per actor/critic save call."""
from __future__ import annotations


class StepFrequency:
    def __init__(self, delegate):
        self.delegate = delegate
        self.step = None
        self.checked_step = None
        self.decision = False
        self.final = False

    def begin(self, epoch: int, epoch_step: int, global_step: int, *, final=False) -> None:
        self.step = (epoch, epoch_step, global_step)
        self.final = final

    def check(self, **increments) -> bool:
        if self.step is None:
            raise RuntimeError("Saving requires an explicit optimizer-step identity")
        if self.checked_step != self.step:
            self.decision = self.delegate.check(**increments) or self.final
            self.checked_step = self.step
        return self.decision

    def state_dict(self):
        return self.delegate.state_dict()

    def load_state_dict(self, state):
        self.delegate.load_state_dict(state)
        self.step = self.checked_step = None

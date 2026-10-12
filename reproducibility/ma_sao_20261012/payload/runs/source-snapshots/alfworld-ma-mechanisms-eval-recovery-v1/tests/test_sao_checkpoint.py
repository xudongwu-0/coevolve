from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "whale_local"))
from sao_bridge.checkpoint import StepFrequency
from areal.utils.timeutil import EpochStepTimeFreqCtl


class CheckpointTests(unittest.TestCase):
    def test_even_frequency_saves_actor_and_critic_at_the_same_steps(self):
        frequency = StepFrequency(EpochStepTimeFreqCtl(freq_step=4, freq_epoch=None, freq_sec=None))
        actor, critic = [], []
        for step in range(8):
            frequency.begin(0, step, step)
            if frequency.check(epochs=int(step == 7), steps=1):
                actor.append(step)
            if frequency.check(epochs=int(step == 7), steps=1):
                critic.append(step)
        self.assertEqual(actor, [3, 7])
        self.assertEqual(critic, [3, 7])

    def test_restored_frequency_counts_optimizer_steps(self):
        frequency = StepFrequency(EpochStepTimeFreqCtl(freq_step=2, freq_epoch=None, freq_sec=None))
        frequency.begin(0, 0, 0)
        self.assertFalse(frequency.check(epochs=0, steps=1))
        restored = StepFrequency(EpochStepTimeFreqCtl(freq_step=2, freq_epoch=None, freq_sec=None))
        restored.load_state_dict(frequency.state_dict())
        restored.begin(0, 1, 1)
        self.assertTrue(restored.check(epochs=0, steps=1))
        self.assertTrue(restored.check(epochs=0, steps=1))

    def test_final_step_is_saved_even_between_periodic_saves(self):
        frequency = StepFrequency(EpochStepTimeFreqCtl(freq_step=4, freq_epoch=None, freq_sec=None))
        saved = []
        for step in range(5):
            frequency.begin(0, step, step, final=step == 4)
            if frequency.check(epochs=0, steps=1):
                saved.append(step)
        self.assertEqual(saved, [3, 4])


if __name__ == "__main__":
    unittest.main()

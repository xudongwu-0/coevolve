"""Use WHALE's unchanged strict boxed-answer scorer without GPU dependencies."""
import importlib.util
from pathlib import Path

SOURCE = Path(__file__).resolve().parents[1] / "WHALE/domains/math_reasoning/verl/utils/reward_score/math_dapo.py"
_spec = importlib.util.spec_from_file_location("whale_native_math_dapo", SOURCE)
native = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(native)


def binary_reward(response, target):
    # Arrow list columns deserialize as NumPy arrays in pandas, while the
    # environment passes the already-unwrapped string. Preserve one target.
    if not isinstance(target, str):
        assert len(target) == 1
        target = target[0]
    assert isinstance(target, str)
    result = native.compute_score(response, str(target), strict_box_verify=True)
    return float(float(result.get("score", 0)) > 0)


def last_assistant(messages):
    for message in reversed(messages):
        role = message.get("role") if isinstance(message, dict) else getattr(message, "role", None)
        if role == "assistant":
            content = message.get("content") if isinstance(message, dict) else getattr(message, "content", "")
            return content if isinstance(content, str) else ""
    return ""

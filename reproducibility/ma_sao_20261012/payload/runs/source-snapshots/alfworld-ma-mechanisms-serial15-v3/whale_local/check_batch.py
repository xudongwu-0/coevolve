"""Check exact prompt/response alignment used by the local RSFT transport."""
from pathlib import Path
from types import SimpleNamespace
import json
import os

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("WHALE_RUN_ROOT", str(ROOT / "runs/whale_setup"))
os.environ.setdefault("WHALE_MODEL_ROOT", str(ROOT / "local_assets/weights/W0"))
import worker


def main():
    worker.state["tokenizer"] = SimpleNamespace(pad_token_id=0)
    calls = [{"prompt_ids": [11, 12, 13], "response_ids": [31, 32]},
             {"prompt_ids": [14], "response_ids": [33]}]
    batch = worker.make_batch(calls).batch
    assert batch["input_ids"].tolist() == [[11, 12, 13, 31, 32], [0, 0, 14, 33, 0]]
    assert batch["responses"].tolist() == [[31, 32], [33, 0]]
    assert batch["response_mask"].tolist() == [[1, 1], [1, 0]]
    assert batch["attention_mask"].tolist() == [[1, 1, 1, 1, 1], [0, 0, 1, 1, 0]]
    assert batch["position_ids"].tolist() == [[0, 1, 2, 3, 4], [0, 0, 0, 1, 1]]
    try:
        worker.make_batch([])
        raise AssertionError("Empty success set incorrectly accepted")
    except ValueError:
        pass
    result = {"passed": True, "exact_sampled_tokens": True, "prompt_loss_excluded": True,
              "response_padding_loss_excluded": True, "variable_context_positions": True,
              "empty_success_set_rejected": True}
    (ROOT / "runs/whale_setup/batch_checks.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()

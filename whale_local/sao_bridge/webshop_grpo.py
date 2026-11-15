"""SAO Solver-stage WebShop GRPO backend; Engineer remains frozen and no teacher is scored."""

import argparse
import json
from pathlib import Path

from whale_local.webshop.core import Protocol, training_rows


def validate_solver_protocol(data):
    protocol = Protocol.from_dict(data)
    if protocol.grpo_profile != "reference_step" or protocol.distill_coefficient != 0:
        raise ValueError("SAO WebShop GRPO requires reference_step and distill_coefficient=0")
    if protocol.group_size != 8:
        raise ValueError("The initial SAO GRPO comparison requires eight rollouts of each exact task")
    if protocol.updates > 16 or len(protocol.eval_ids) > 64 or len(protocol.eval_seeds) > 2:
        raise ValueError("This backend is gated to subset pilots; no full training/evaluation")
    return protocol


def solver_training_rows(episodes, codec, protocol):
    validate_solver_protocol(protocol.as_dict())
    rows, statistics = training_rows(episodes, codec, protocol, use_distillation=False)
    if any(row.teacher_prompt_ids is not None for row in rows):
        raise ValueError("Privileged teacher context cannot enter SAO GRPO")
    return rows, {**statistics, "sao_stage": "solver", "engineer_updated": False}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("train", "evaluate", "environment-check"))
    parser.add_argument("--config", type=Path, required=True)
    args, rest = parser.parse_known_args(argv)
    validate_solver_protocol(json.loads(args.config.read_text()))
    if "--method" in rest:
        raise ValueError("This SAO backend has no distillation method switch")
    from whale_local.webshop.__main__ import main as run_webshop
    forwarded = [args.command, "--config", str(args.config), *rest]
    if args.command == "train":
        forwarded += ["--method", "grpo"]
    return run_webshop(forwarded)


if __name__ == "__main__":
    raise SystemExit(main())

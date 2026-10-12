"""Audited small-scale Harness-R1 optimizer execution with strict commit verification."""
import argparse
from pathlib import Path
import sys

from sao_bridge.common import write_json
from sao_bridge.launch import Launcher
from .pipeline import verify_grpo_commit


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--reward-config", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--updates", type=int, default=1)
    parser.add_argument("--rollout-gpus", default="4,5,6,7",
                        help="physical rollout GPU ids matching the config's rollout_tp")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    launch = Launcher(args.output / "processes")
    try:
        write_json(args.output / "STATUS.json", {"stage": "real_online_grpo", "updates": args.updates})
        process = launch.start([
            sys.executable, "-B", "-m", "torch.distributed.run", "--standalone",
            "--local-addr=127.0.0.1", "--nproc-per-node=4", "--max-restarts=0",
            "-m", "harness_r1_searchqa.grpo_train", "--model", str(args.model),
            "--data", str(args.data), "--reward-config", str(args.reward_config),
            "--config", str(args.config), "--output", str(args.output / "engine"),
            "--updates", str(args.updates), "--rollout-gpus", args.rollout_gpus,
        ], gpus="0,1,2,3", name="grpo")
        launch.wait(process, "Official-objective Harness-R1 4B pilot")
        receipt = verify_grpo_commit(args.output / "engine", args.updates)
        write_json(args.output / "STATUS.json", {
            "stage": "optimizer_updates_verified", "optimizer_steps": receipt["optimizer_steps"],
            "actor_sha256": receipt["actor_sha256"], "dev_gain_claimed": False,
        })
    except (OSError, RuntimeError, ValueError) as exc:
        write_json(args.output / "STATUS.json", {"stage": "blocked", "error": str(exc)})
        raise
    finally:
        launch.close()


if __name__ == "__main__":
    main()

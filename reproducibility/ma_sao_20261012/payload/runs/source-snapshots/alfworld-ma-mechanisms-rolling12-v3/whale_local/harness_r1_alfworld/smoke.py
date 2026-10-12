"""Solve one official game with the CPU hand-coded expert; no LLM/GPU needed."""
import argparse
import importlib.util
import json
import os

import yaml

from .driver import ASSETS, UPSTREAM, cpu_env, digest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-id", type=int, default=0)
    args = parser.parse_args()
    os.environ.update(cpu_env())
    source = UPSTREAM / "code/life-harness/AgentBench/src/server/tasks/alfworld"
    spec = importlib.util.spec_from_file_location("official_alfworld_environment", source / "environment.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    config = yaml.safe_load((source / "configs/base_config.yaml").read_text())
    config["general"]["use_cuda"] = False
    config["env"]["regen_game_files"] = False
    config["env"]["task_types"] = [1, 2, 3, 4, 5, 6]
    config["dagger"]["training"]["max_nb_steps_per_episode"] = 150
    cohort = json.loads((ASSETS / "cohort32.json").read_text())
    position = cohort["task_ids"].index(args.task_id)
    game = ASSETS / "data" / cohort["task_paths"][position]
    wrapper = module.AlfworldEnvWrapper(config)
    # "train" requests the CPU expert plan; game identity and dynamics stay unchanged.
    env = module.SingleAlfredTWEnv(config, str(game), train_eval="train").init_env(batch_size=1)
    steps = []
    try:
        observations, infos = wrapper.reset_env(env)
        initial = observations[0]
        won = False
        for step in range(150):
            plan = infos.get("extra.expert_plan", [[]])[0]
            if not plan:
                raise RuntimeError(f"expert plan absent at step {step}; keys={list(infos)}")
            action = plan[0]
            observations, scores, dones, infos = wrapper.step_env(env, action)
            steps.append({"action": action, "observation": observations[0]})
            won = bool(infos.get("won", [False])[0])
            if dones[0] or won:
                break
        report = {"kind": "CPU_handcoded_expert_environment_smoke_not_model_evaluation",
                  "task_id": args.task_id, "game": str(game),
                  "game_sha256": digest(game), "initial_observation": initial,
                  "won": won, "steps": steps, "cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"]}
        report["expert_smoke_step_limit"] = 150
        (ASSETS / "cpu_smoke.json").write_text(json.dumps(report, indent=2) + "\n")
        if not won:
            raise RuntimeError("CPU expert failed to solve the game")
        print(f"CPU smoke solved official task {report['task_id']} in {len(steps)} steps")
    finally:
        wrapper.close_env(env)


if __name__ == "__main__":
    main()

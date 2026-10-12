"""Transfer actual active-agent proposals through a private Modal Dict."""
from pathlib import Path
import argparse
import json
import modal


def main():
    p = argparse.ArgumentParser()
    p.add_argument("command", choices=["fetch", "submit"])
    p.add_argument("--directory", type=Path, default=Path("runs/modal_proposer_jobs"))
    p.add_argument("--job-id")
    args = p.parse_args()
    queue = modal.Dict.from_name("whale-gate-active-agent-jobs", create_if_missing=False)
    if args.command == "fetch":
        jobs = []
        for job_id, value in queue.items():
            if value.get("status") != "awaiting_active_agent":
                continue
            directory = args.directory / job_id
            directory.mkdir(parents=True, exist_ok=True)
            (directory / "request.json").write_text(json.dumps(value["request"], indent=2) + "\n")
            jobs.append({"job_id": job_id, "request": str(directory / "request.json"),
                         "run_dir": value["request"]["run_dir"]})
        print(json.dumps(jobs, indent=2))
    else:
        if not args.job_id:
            p.error("submit requires --job-id")
        value = queue[args.job_id]
        if value.get("status") != "awaiting_active_agent":
            raise RuntimeError("Job no longer waiting; response not replaced")
        response = json.loads((args.directory / args.job_id / "response.json").read_text())
        if not response.get("candidates"):
            raise ValueError("Missing reviewed candidates")
        queue[args.job_id] = value | {"response": response, "status": "responded"}
        print("Submitted reviewed proposal", args.job_id)


if __name__ == "__main__":
    main()

"""Reject incomplete AgentBench rollouts before reporting model task gains."""
import argparse
import json
import math
from pathlib import Path


def require_complete_task_report(report: dict, expected_tasks: int) -> None:
    if expected_tasks <= 0 or report.get("total") != expected_tasks:
        raise ValueError("AgentBench task count does not match the paired cohort")
    validation = report.get("validation")
    if not isinstance(validation, dict):
        raise ValueError("Missing AgentBench task validation report")
    for name in ("running", "unknown", "task error"):
        value = validation.get(name)
        if not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError(f"Missing or invalid AgentBench validation: {name}")
        if value != 0:
            raise ValueError(f"Incomplete/infrastructure-failed AgentBench tasks: {name}={value}")
    counts = report.get("custom", {}).get("overall", {})
    if counts.get("total") != expected_tasks:
        raise ValueError("Missing or mismatched AgentBench outcome counts")
    passed, wrong = counts.get("pass"), counts.get("wrong")
    if (type(passed) is not int or type(wrong) is not int
            or passed < 0 or wrong < 0 or passed + wrong != expected_tasks):
        raise ValueError("Incomplete AgentBench outcome counts")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    parser.add_argument("--expected-tasks", type=int, required=True)
    args = parser.parse_args()
    report = json.loads(args.report.read_text())
    require_complete_task_report(report, args.expected_tasks)
    print(json.dumps({
        "task_execution_complete": True,
        "outcome": report["custom"]["overall"],
        "validation": report["validation"],
    }, indent=2))


if __name__ == "__main__":
    main()

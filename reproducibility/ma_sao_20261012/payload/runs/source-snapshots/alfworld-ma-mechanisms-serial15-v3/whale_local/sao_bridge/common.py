"""CPU-only data, restricted harness, and provenance helpers."""
from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path
import string
import subprocess

from harness_code_contract import validate_source
from math_reward import binary_reward, native

ROOT = Path(__file__).resolve().parents[2]
SAO = ROOT / "third_party/sao"
AREAL = SAO / "vendor/AReaL"
SAO_REVISION = "ac2728149041b5f15ceb75413f467ce0659179cf"
AREAL_REVISION = "3cf0dfbd2b0fbeabd6977184980e189d1567747a"
INITIAL_SOURCE = '''from environments.retool.base_harness import ReToolEnv

class CandidateEnv(ReToolEnv):
    MAX_TURNS = 2
    SYSTEM_PROMPT = "Solve the mathematics problem carefully and concisely."
    USER_PROMPT_TEMPLATE = "{question}"
'''
EDITOR_SYSTEM = """Edit the literal settings in this reusable math-solver harness.
Your entire response must be one Python module with exactly five nonempty lines.
Copy this structure, keeping the import, class name and three attribute names:
from environments.retool.base_harness import ReToolEnv
class CandidateEnv(ReToolEnv):
    MAX_TURNS = 2
    SYSTEM_PROMPT = "Solve the mathematics problem carefully and concisely."
    USER_PROMPT_TEMPLATE = "{question}"
Change only the literal values. MAX_TURNS must be 1, 2 or 3. Write a better
general math-solving instruction in SYSTEM_PROMPT, as a single quoted string.
USER_PROMPT_TEMPLATE must contain {question} exactly once and no other fields.
Do not implement any functions or methods. Do not add imports, attributes,
markdown fences, explanations, examples, problem-specific answers or grading
changes. Start directly with the import. The solver must finish with \\boxed{...}.
"""
FINAL_FORMAT = "\nGive the final answer in \\boxed{...}."
RETRY_MESSAGE = (
    "Continue the unfinished solution without restarting it. "
    "Finish with the requested answer in \\boxed{...}."
)


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)


def read_json(path: Path):
    return json.loads(path.read_text())


def file_sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(2**20), b""):
            digest.update(block)
    return digest.hexdigest()


def ensure_model_runtime_metadata(checkpoint: Path, source: Path) -> dict:
    if not checkpoint.is_dir() or not source.is_dir():
        raise FileNotFoundError("Model checkpoint and metadata source must be directories")
    files = {}
    for name in ("preprocessor_config.json", "processor_config.json"):
        origin = source / name
        destination = checkpoint / name
        if not origin.is_file():
            raise FileNotFoundError(f"Missing required runtime metadata: {origin}")
        origin_sha = file_sha(origin)
        if destination.is_file():
            status = "already_present"
        else:
            temporary = destination.with_name(destination.name + ".tmp")
            temporary.write_bytes(origin.read_bytes())
            temporary.replace(destination)
            status = "copied"
        files[name] = {
            "source_sha256": origin_sha,
            "checkpoint_sha256": file_sha(destination),
            "status": status,
        }
    receipt = {
        "source": str(source.resolve()),
        "weights_unchanged": True,
        "files": files,
    }
    write_json(checkpoint / "runtime_supplements.json", receipt)
    return receipt


def verify_upstreams() -> dict[str, str]:
    result = {}
    sao, areal = SAO.resolve(), AREAL.resolve()
    for name, path, revision in (
        ("sao", sao, SAO_REVISION), ("areal", areal, AREAL_REVISION)
    ):
        actual = subprocess.check_output(
            ["git", "-c", f"safe.directory={path}", "-C", str(path),
             "rev-parse", "HEAD"], text=True,
        ).strip()
        if actual != revision:
            raise RuntimeError(f"{name} revision is {actual}, expected {revision}")
        result[name] = actual
    subprocess.run(
        ["git", "-c", f"safe.directory={areal}", "-C", str(areal),
         "apply", "--reverse", "--check",
         str(SAO / "patches/areal-sao.patch")],
        check=True, capture_output=True,
    )
    result["sao_patch_sha256"] = file_sha(SAO / "patches/areal-sao.patch")
    for name in (
        "areal-local-network.patch",
        "areal-packed-sdpa.patch",
        "areal-assistant-prefill.patch",
        "areal-qwen35-critic.patch",
    ):
        runtime_patch = Path(__file__).with_name(name)
        subprocess.run(
            ["git", "-c", f"safe.directory={areal}", "-C", str(areal),
             "apply", "--reverse", "--check", str(runtime_patch)],
            check=True, capture_output=True,
        )
        result[name.removesuffix(".patch").replace("-", "_") + "_sha256"] = file_sha(runtime_patch)
    return result


def harness_spec(content: str) -> dict:
    source = validate_source(content)
    tree = ast.parse(source)
    if len(tree.body) != 2 or not isinstance(tree.body[0], ast.ImportFrom):
        raise ValueError("This baseline permits one import and one harness class")
    imported, cls = tree.body
    if (
        imported.module != "environments.retool.base_harness"
        or imported.level != 0
        or len(imported.names) != 1
        or imported.names[0].name != "ReToolEnv"
        or imported.names[0].asname is not None
        or not isinstance(cls, ast.ClassDef)
        or cls.decorator_list
        or cls.keywords
    ):
        raise ValueError("Unexpected harness declaration")
    attrs = {}
    for node in cls.body:
        if (
            not isinstance(node, ast.Assign)
            or len(node.targets) != 1
            or not isinstance(node.targets[0], ast.Name)
        ):
            raise ValueError("Only literal class attributes are supported in this baseline")
        name = node.targets[0].id
        if name in attrs:
            raise ValueError(f"Duplicate harness attribute: {name}")
        attrs[name] = ast.literal_eval(node.value)
    if set(attrs) != {"MAX_TURNS", "SYSTEM_PROMPT", "USER_PROMPT_TEMPLATE"}:
        raise ValueError("All three harness attributes must be present")
    if type(attrs["MAX_TURNS"]) is not int or not 1 <= attrs["MAX_TURNS"] <= 3:
        raise ValueError("MAX_TURNS must be an integer between 1 and 3")
    for name in ("SYSTEM_PROMPT", "USER_PROMPT_TEMPLATE"):
        if not isinstance(attrs[name], str) or len(attrs[name]) > 2000:
            raise ValueError(f"Invalid {name}")
    fields = list(string.Formatter().parse(attrs["USER_PROMPT_TEMPLATE"]))
    if [(name, fmt, conv) for _, name, fmt, conv in fields if name is not None] != [
        ("question", "", None)
    ]:
        raise ValueError("Template may interpolate only one plain {question}")
    return {"source": source, **attrs}


def solver_messages(spec: dict, question: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": spec["SYSTEM_PROMPT"]},
        {"role": "user", "content": (
            spec["USER_PROMPT_TEMPLATE"].format(question=question) + FINAL_FORMAT
        )},
    ]


def prepare_data(source: Path, out: Path, train_n: int, dev_n: int, test_n: int) -> dict:
    import pandas as pd
    import re

    if out.exists():
        raise FileExistsError(out)
    if min(train_n, dev_n, test_n) < 1:
        raise ValueError("All data splits must be nonempty")
    pools = {}
    all_train = set()
    for split in ("train", "test"):
        rows = {}
        for row in pd.read_parquet(source / f"{split}.parquet").to_dict("records"):
            question = row["problem"].strip()
            key = hashlib.sha256(question.encode()).hexdigest()
            if split == "train":
                all_train.add(key)
            elif key in all_train:
                continue
            if row["level"] not in {"Level 2", "Level 3", "Level 4"}:
                continue
            boxed = native.last_boxed_only_string(row["solution"])
            answer = native.remove_boxed(boxed).strip() if boxed else ""
            if not re.fullmatch(r"-?(0|[1-9]\d*)", answer):
                continue
            rows[key] = {"id": key, "question": question, "answer": answer,
                         "level": row["level"]}
        pools[split] = [rows[key] for key in sorted(rows)]
    if len(pools["train"]) < train_n + dev_n or len(pools["test"]) < test_n:
        raise ValueError("Not enough filtered source examples")
    selected = {
        "train": pools["train"][:train_n],
        "dev": pools["train"][train_n:train_n + dev_n],
        "test": pools["test"][:test_n],
    }
    out.mkdir(parents=True)
    manifest = {
        "dataset": "MATH number_theory",
        "filter": "Level 2-4, canonical integer answers; no policy-based filtering",
        "source_files": {s: file_sha(source / f"{s}.parquet") for s in pools},
        "splits": {},
    }
    for name, rows in selected.items():
        write_json(out / f"{name}.json", rows)
        manifest["splits"][name] = {
            "count": len(rows), "ids": [r["id"] for r in rows],
            "sha256": file_sha(out / f"{name}.json"),
        }
    ids = [set(s["ids"]) for s in manifest["splits"].values()]
    if any(a & b for i, a in enumerate(ids) for b in ids[i + 1:]):
        raise RuntimeError("Data split overlap")
    write_json(out / "manifest.json", manifest)
    return manifest

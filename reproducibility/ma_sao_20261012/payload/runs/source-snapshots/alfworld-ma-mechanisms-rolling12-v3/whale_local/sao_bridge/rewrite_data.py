"""Validated teacher edits and a localhost, active-agent FastAPI queue."""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import secrets
import stat

from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field

from .common import read_json, write_json


class EditExample(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str = Field(pattern=r"^[a-z0-9_-]{1,80}$")
    family: str = Field(min_length=1, max_length=100)
    split: str = Field(pattern=r"^(train|dev|test)$")
    format: str = Field(pattern=r"^(python|json)$")
    instruction: str = Field(min_length=10, max_length=1800)
    source: str = Field(min_length=2, max_length=6000)
    target: str = Field(min_length=2, max_length=6000)
    changes: dict


def parse_document(text: str, format: str) -> tuple[dict, str]:
    if format == "json":
        def unique(pairs):
            result = {}
            for key, value in pairs:
                if key in result:
                    raise ValueError("Duplicate JSON field")
                result[key] = value
            return result
        values = json.loads(text, object_pairs_hook=unique)
        if not isinstance(values, dict):
            raise ValueError("Expected a JSON object")
        structure = "json"
    elif format == "python":
        tree = ast.parse(text)
        if len(tree.body) != 2 or not isinstance(tree.body[0], ast.ImportFrom):
            raise ValueError("Expected one from-import followed by one class")
        imported, cls = tree.body
        if not isinstance(cls, ast.ClassDef) or cls.decorator_list or cls.keywords:
            raise ValueError("Expected an undecorated configuration class")
        if len(cls.bases) != 1 or not isinstance(cls.bases[0], ast.Name):
            raise ValueError("Expected one named base class")
        if imported.level or len(imported.names) != 1 or imported.names[0].asname:
            raise ValueError("Expected one unaliased absolute import")
        if imported.names[0].name != cls.bases[0].id:
            raise ValueError("Base class must match the import")
        values = {}
        for node in cls.body:
            if not isinstance(node, ast.Assign) or len(node.targets) != 1:
                raise ValueError("Only literal configuration assignments are permitted")
            key = node.targets[0]
            if not isinstance(key, ast.Name) or key.id in values:
                raise ValueError("Duplicate or invalid configuration field")
            values[key.id] = ast.literal_eval(node.value)
        structure = ast.dump(imported) + "|" + cls.name + "|" + cls.bases[0].id
    else:
        raise ValueError("Unknown document format")
    if not values or len(values) > 16:
        raise ValueError("Expected 1-16 settings")
    if any(type(value) not in (str, int, bool) for value in values.values()):
        raise ValueError("Only string, integer and boolean settings are supported")
    return values, structure


def equal_value(left, right):
    return type(left) is type(right) and left == right


def validate_example(example: EditExample) -> None:
    before, structure = parse_document(example.source, example.format)
    after, target_structure = parse_document(example.target, example.format)
    if structure != target_structure or before.keys() != after.keys():
        raise ValueError("An edit changed the declaration or available fields")
    actual = {key: after[key] for key in before if not equal_value(before[key], after[key])}
    if not actual or actual.keys() != example.changes.keys():
        raise ValueError("Declared changes do not match the target edit")
    if any(not equal_value(actual[key], example.changes[key]) for key in actual):
        raise ValueError("Declared change has wrong type or value")
    if any(type(before[key]) is not type(after[key]) for key in before):
        raise ValueError("Setting types must be preserved")
    for key, value in before.items():
        if isinstance(value, str):
            for marker in ("{question}", "{query}", "{document}", "{task}", "{input}"):
                if value.count(marker) != str(after[key]).count(marker):
                    raise ValueError("An edit changed an interpolation placeholder")
    if "\n" not in example.target and example.format == "python":
        raise ValueError("Invalid Python document")


def grade_output(example: dict, output: str) -> dict:
    item = EditExample.model_validate(example)
    try:
        predicted, structure = parse_document(output, item.format)
    except (ValueError, SyntaxError, TypeError) as exc:
        return {"legal": False, "instruction_following": False, "error": str(exc)}
    expected, expected_structure = parse_document(item.target, item.format)
    legal = (structure == expected_structure and predicted.keys() == expected.keys()
             and all(type(predicted[k]) is type(expected[k]) for k in expected))
    correct = legal and all(equal_value(predicted[k], expected[k]) for k in expected)
    return {"legal": legal, "instruction_following": correct, "error": None}


class JobSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")
    shard: str = Field(pattern=r"^[a-z0-9_-]{1,50}$")
    instruction: str
    counts: dict[str, int]


class TeacherResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")
    request_sha256: str
    teacher: str
    authorship: str
    examples: list[EditExample]


def digest(value: dict) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def teacher_token(path: Path, *, resume: bool = False) -> str:
    if resume:
        fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
        with os.fdopen(fd) as stream:
            info = os.fstat(stream.fileno())
            if (not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid()
                    or info.st_mode & 0o077):
                raise PermissionError("Resuming requires a private token file owned by this user")
            token = stream.read().strip()
        if len(token) < 32:
            raise ValueError("Existing teacher token is invalid")
        return token
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    token = secrets.token_urlsafe(32)
    with os.fdopen(fd, "w") as stream:
        stream.write(token)
    return token


def create_app(root: Path, token: str) -> FastAPI:
    root.mkdir(parents=True, exist_ok=True, mode=0o700)
    if len(token) < 32:
        raise ValueError("Teacher queue requires a strong local bearer token")

    def authorize(authorization: str = Header(default="")):
        if not secrets.compare_digest(authorization, "Bearer " + token):
            raise HTTPException(401, "Unauthorized teacher queue request")

    app = FastAPI(dependencies=[Depends(authorize)])

    @app.get("/health")
    def health():
        return {"ready": True, "mode": "active_agent_queue_not_unattended_inference"}

    def job_dir(shard: str) -> Path:
        if not shard or any(c not in "abcdefghijklmnopqrstuvwxyz0123456789_-" for c in shard):
            raise HTTPException(400, "Invalid shard")
        return root / shard

    @app.post("/jobs")
    def submit(spec: JobSpec):
        if (set(spec.counts) != {"train", "dev", "test"}
                or any(n < 0 for n in spec.counts.values()) or sum(spec.counts.values()) < 1):
            raise HTTPException(400, "Declare nonnegative counts for all splits and at least one example")
        directory = job_dir(spec.shard)
        try:
            directory.mkdir(mode=0o700)
        except FileExistsError:
            raise HTTPException(409, "Teacher shard already exists")
        request = spec.model_dump()
        write_json(directory / "request.json", request)
        return {"shard": spec.shard, "request_sha256": digest(request)}

    @app.get("/jobs/{shard}/request")
    def request(shard: str):
        path = job_dir(shard) / "request.json"
        if not path.exists():
            raise HTTPException(404, "Unknown teacher shard")
        value = read_json(path)
        return {"request": value, "request_sha256": digest(value)}

    @app.post("/jobs/{shard}/response")
    def respond(shard: str, response: TeacherResponse):
        directory = job_dir(shard)
        if not (directory / "request.json").exists():
            raise HTTPException(404, "Unknown teacher shard")
        request = read_json(directory / "request.json")
        if response.request_sha256 != digest(request):
            raise HTTPException(409, "Stale teacher response")
        if response.authorship != "active_agent_authored":
            raise HTTPException(400, "This queue requires actual active-agent authorship")
        counts = {split: sum(e.split == split for e in response.examples) for split in request["counts"]}
        if counts != request["counts"] or len({e.id for e in response.examples}) != len(response.examples):
            raise HTTPException(400, "Teacher example counts or identifiers do not match")
        try:
            for example in response.examples:
                validate_example(example)
        except (ValueError, SyntaxError, TypeError) as exc:
            raise HTTPException(400, str(exc))
        path = directory / "response.json"
        try:
            with path.open("x") as stream:
                json.dump(response.model_dump(), stream, indent=2)
                stream.write("\n")
        except FileExistsError:
            raise HTTPException(409, "Teacher response is immutable")
        return {"accepted": len(response.examples), "counts": counts}

    @app.get("/jobs/{shard}/response")
    def result(shard: str):
        directory = job_dir(shard)
        if not (directory / "request.json").exists():
            raise HTTPException(404, "Unknown teacher shard")
        if not (directory / "response.json").exists():
            return JSONResponse(status_code=202, content={"status": "awaiting_active_agent"})
        return read_json(directory / "response.json")

    return app


def assemble(root: Path, output: Path, shards: list[str] | None = None) -> dict:
    if shards is None:
        paths = sorted(root.glob("*/response.json"))
        if not paths or len(paths) != len(list(root.glob("*/request.json"))):
            raise ValueError("Every teacher shard must have a completed response")
    else:
        if not shards or len(set(shards)) != len(shards):
            raise ValueError("Selected teacher shards must be nonempty and unique")
        if any(not name or any(c not in "abcdefghijklmnopqrstuvwxyz0123456789_-" for c in name)
               for name in shards):
            raise ValueError("Invalid selected teacher shard identifier")
        paths = [root / name / "response.json" for name in sorted(shards)]
        if any(not path.is_file() or not (path.parent / "request.json").is_file() for path in paths):
            raise ValueError("Every selected teacher shard must have a completed response")
    examples, fingerprints, ids, families, provenance = [], set(), set(), {}, {}
    for path in paths:
        response = TeacherResponse.model_validate(read_json(path))
        request = read_json(path.parent / "request.json")
        if digest(request) != response.request_sha256:
            raise ValueError("Teacher request digest mismatch")
        actual_counts = {
            split: sum(e.split == split for e in response.examples)
            for split in ("train", "dev", "test")
        }
        if actual_counts != request["counts"]:
            raise ValueError("Persisted teacher response does not match requested counts")
        provenance[path.parent.name] = {
            "teacher": response.teacher, "authorship": response.authorship,
            "response_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
        for item in response.examples:
            validate_example(item)
            fingerprint = digest({"instruction": item.instruction, "source": item.source})
            if item.id in ids or fingerprint in fingerprints:
                raise ValueError("Duplicate teacher sample")
            if item.family in families and families[item.family] != item.split:
                raise ValueError("An edit family leaked across train/dev/test")
            ids.add(item.id)
            fingerprints.add(fingerprint)
            families[item.family] = item.split
            examples.append(item.model_dump())
    output.mkdir(mode=0o700)
    write_json(output / "examples.json", examples)
    result = {
        "counts": {split: sum(e["split"] == split for e in examples) for split in ("train", "dev", "test")},
        "selected_shards": [path.parent.name for path in paths],
        "families": families, "provenance": provenance,
        "examples_sha256": hashlib.sha256((output / "examples.json").read_bytes()).hexdigest(),
        "scope": "synthetic configuration editing; not mathematical capability distillation",
    }
    write_json(output / "manifest.json", result)
    return result


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    serve = sub.add_parser("serve")
    serve.add_argument("--root", type=Path, required=True)
    serve.add_argument("--token-file", type=Path, required=True)
    serve.add_argument("--port", type=int, default=18543)
    serve.add_argument("--resume", action="store_true",
                       help="Restart an existing queue using its private token without replacing data")
    pack = sub.add_parser("assemble")
    pack.add_argument("--root", type=Path, required=True)
    pack.add_argument("--output", type=Path, required=True)
    pack.add_argument("--shards", nargs="+",
                      help="Freeze only these completed shards while later authoring continues")
    args = parser.parse_args()
    if args.command == "assemble":
        print(json.dumps(assemble(args.root, args.output, args.shards), indent=2))
        return
    import uvicorn
    if args.resume and not args.root.is_dir():
        raise FileNotFoundError("Resuming requires an existing teacher queue")
    app = create_app(args.root, teacher_token(args.token_file, resume=args.resume))
    uvicorn.run(app, host="127.0.0.1", port=args.port)


if __name__ == "__main__":
    main()

"""Run multiple disjoint model roles under one audited GPU supervisor."""
from __future__ import annotations

import argparse
from contextlib import ExitStack
import json
import os
from pathlib import Path
import signal
import threading
import time

from sao_bridge.common import file_sha, write_json
from sao_bridge.launch import Launcher
from sao_bridge.resources import require_research_node


def validate_roles(roles: list[dict]) -> None:
    used = set()
    names = set()
    ports = set()
    for role in roles:
        name = role["name"]
        devices = role["gpus"].split(",")
        if name not in {"target", "engineer"} or name in names:
            raise ValueError("Require unique target/engineer roles")
        if not devices or any(device not in {str(i) for i in range(8)} for device in devices):
            raise ValueError("Invalid research-node GPU allocation")
        if len(set(devices)) != len(devices) or used.intersection(devices):
            raise ValueError("Model roles must use disjoint physical GPUs")
        for index in range(len(devices)):
            port = role["port"] + index * 10
            if port in ports or not 1024 <= port <= 65535:
                raise ValueError("Overlapping or invalid role endpoints")
            ports.add(port)
        used.update(devices)
        names.add(name)
    if names != {"target", "engineer"}:
        raise ValueError("Both frozen target and Engineer roles are required")


def verify_role(role: dict) -> None:
    model = Path(role["model"])
    if "model_file_sha256" in role:
        if file_sha(model / "model.safetensors") != role["model_file_sha256"]:
            raise ValueError("Frozen target weight identity changed")
    else:
        receipt = Path(role["model_receipt"])
        if file_sha(receipt) != role["model_receipt_sha256"]:
            raise ValueError("Engineer download receipt identity changed")
        provenance = json.loads(receipt.read_text())
        for item in provenance["files"]:
            path = model / item["path"]
            if not path.resolve().is_relative_to(model.resolve()) or file_sha(path) != item["sha256"]:
                raise ValueError("Engineer protected checkpoint file identity changed")


def role_server_args(role: dict) -> list[str]:
    extra = []
    parser = role.get("tool_call_parser")
    if parser is not None:
        if parser not in {"hermes", "qwen3_xml"}:
            raise ValueError("Unsupported role tool-call parser")
        extra.extend(["--tool-call-parser", parser])
    if role.get("language_model_only", False):
        extra.append("--language-model-only")
    backend = role.get("gdn_prefill_backend")
    if backend is not None:
        if backend != "triton":
            raise ValueError("Unsupported role GDN prefill backend")
        extra.extend(["--gdn-prefill-backend", backend])
    if role.get("prefix_caching"):
        extra.append("--enable-prefix-caching")
    return extra


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--roles", type=Path, required=True)
    parser.add_argument("--run", type=Path, required=True)
    args = parser.parse_args()
    require_research_node()
    roles = json.loads(args.roles.read_text())["roles"]
    validate_roles(roles)
    for role in roles:
        verify_role(role)
    args.run.mkdir(parents=True, exist_ok=False)
    done = threading.Event()
    signal.signal(signal.SIGTERM, lambda *args: done.set())
    signal.signal(signal.SIGINT, lambda *args: done.set())
    launcher = Launcher(args.run)
    try:
        with ExitStack() as stack:
            endpoints = {}
            for role in roles:
                extra = role_server_args(role)
                if role.get("engineer_think_prefill"):
                    from transformers import AutoTokenizer
                    from .sft_train import configure_engineer_template
                    tokenizer = AutoTokenizer.from_pretrained(role["model"], local_files_only=True)
                    configure_engineer_template(tokenizer, True)
                    template = args.run / "engineer_template.jinja"
                    template.write_text(tokenizer.chat_template)
                    extra.extend(["--chat-template", str(template)])
                endpoints[role["name"]] = stack.enter_context(launcher.vllm(
                    role["model"], port=role["port"], gpus=role["gpus"],
                    name=role["name"], data_parallel=len(role["gpus"].split(",")),
                    max_model_len=role["context"], server_args=extra,
                ))
            write_json(args.run / "READY.json", {
                "pid": os.getpid(), "endpoints": endpoints,
                "roles_sha256": file_sha(args.roles),
                "started_at_unix": time.time(), "audited_single_owner": True,
            })
            while not done.wait(5):
                launcher.check()
                if any(process.poll() is not None for process in launcher.children):
                    raise RuntimeError("A role replica exited unexpectedly")
    finally:
        launcher.close()
        (args.run / "READY.json").unlink(missing_ok=True)
        write_json(args.run / "STOPPED.json", {"stopped_at_unix": time.time()})


if __name__ == "__main__":
    main()

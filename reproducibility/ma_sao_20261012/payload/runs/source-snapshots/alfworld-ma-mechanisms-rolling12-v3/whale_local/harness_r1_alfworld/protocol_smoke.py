"""Bounded local HTTP stub checks official controller/worker/tool plumbing, not a model."""
import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import threading
import uuid

from .driver import (ASSETS, PYTHON, checked_rows, freeze_baseline, official_paths,
                     preserve_paired_ids, run)


class Stub(BaseHTTPRequestHandler):
    def do_POST(self):
        request = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        if request["model"] not in {"cpu_protocol_stub", "cpu_engineer_stub"}:
            self.send_error(400)
            return
        payload = {"id": "chatcmpl-cpu", "object": "chat.completion", "created": 0,
            "model": "cpu_protocol_stub", "choices": [{"index": 0,
            "message": {"role": "assistant", "content": None, "tool_calls": [
                {"id": "call_" + uuid.uuid4().hex, "type": "function",
                 "function": {"name": "take_action", "arguments": '{"action":"look"}'}}]},
            "finish_reason": "tool_calls"}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}}
        if request["model"] == "cpu_engineer_stub":
            patch = {"benchmark": "alfworld", "description": "CPU protocol hook plumbing",
                "actions": [{"type": "add_code_hook", "hook": "make_pre_hint",
                "code": "def hook(ctx, nb):\n    return {'message': 'Use admissible actions.'}\n"}]}
            payload["model"] = "cpu_engineer_stub"
            payload["choices"][0]["message"] = {"role": "assistant", "content":
                "A reusable admissible-action reminder.</think>\n<patch>" +
                json.dumps(patch) + "</patch>"}
            payload["choices"][0]["finish_reason"] = "stop"
        encoded = json.dumps(payload).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def log_message(self, *args):
        pass


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--size", type=int, choices=[1, 32], default=1)
    args = parser.parse_args()
    identifier = "cpu_protocol_" + uuid.uuid4().hex[:8]
    output = ASSETS / identifier
    output.mkdir()
    ids = output / "task_ids.json"
    task_ids = [0] if args.size == 1 else json.loads((ASSETS / "cohort32.json").read_text())["task_ids"]
    ids.write_text(json.dumps(task_ids) + "\n")
    server = ThreadingHTTPServer(("127.0.0.1", 0), Stub)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    scripts, examples = official_paths()
    try:
        run([str(PYTHON), str(scripts / "harness_r1_batch_debug.py"),
             "--bench", "alfworld", "--run-id", identifier,
             "--agentbench-dir", str(ASSETS / "AgentBench"),
             "--agentbench-python", str(PYTHON), "--alfworld-worker-python", str(PYTHON),
             "--task-ids-file", str(ids), "--batch-size", str(args.size), "--num-batches", "1",
             "--alfworld-split", "test_set", "--alfworld-max-step", "2",
             "--rollout-concurrency", "4", "--skip-adb",
             "--rollout-base-url", f"http://127.0.0.1:{server.server_port}/v1",
             "--rollout-model", "cpu_protocol_stub", "--agent-name", "cpu_protocol_stub",
             "--rollout-tool-choice", "auto", "--rollout-temperature", "0.0",
             "--rollout-chat-template-kwargs", '{"enable_thinking":false}',
             "--controller-port", "15440", "--worker-port-base", "15540",
             "--startup-timeout", "60", "--rollout-timeout", "300",
             "--output-root", str(output / "official")], output / "pipeline.log")
        rows = checked_rows(output / "official", task_ids)
        frozen = freeze_baseline(output / "official")
        packets = output / "failure_packets.jsonl"
        run([str(PYTHON), str(examples / "build_alfworld_dataset.py"),
             "--source-root", str(output / "official"), "--output", str(packets),
             "--alfworld-split", "test_set", "--target-model", "cpu_protocol_stub",
             "--target-agent-name", "cpu_protocol_stub",
             "--response-protocol", "prefill_think_patch",
             "--schema-style", "alfworld_life_multihook_v1",
             "--include-incomplete"], output / "packet.log")
        records = [json.loads(line) for line in packets.read_text().splitlines() if line.strip()]
        preserve_paired_ids(records, task_ids, "cpu_protocol_stub_identity")
        packets.write_text("".join(json.dumps(row) + "\n" for row in records))
        run([str(PYTHON), str(examples / "eval_alfworld_patches.py"),
             "--input", str(packets), "--output-root", str(output / "patched"),
             "--run-id", identifier + "_patched",
             "--engineer-base-url", f"http://127.0.0.1:{server.server_port}/v1",
             "--engineer-model", "cpu_engineer_stub",
             "--engineer-assistant-prefill", "<think>\n",
             "--engineer-extra-body", '{"add_generation_prompt":false}',
             "--response-protocol", "prefill_think_patch",
             "--target-base-url", f"http://127.0.0.1:{server.server_port}/v1",
             "--target-model", "cpu_protocol_stub", "--target-agent-name", "cpu_protocol_stub",
             "--agentbench-dir", str(ASSETS / "AgentBench"),
             "--agentbench-python", str(PYTHON), "--alfworld-worker-python", str(PYTHON),
             "--alfworld-split", "test_set", "--alfworld-max-step", "2",
             "--rollout-tool-choice", "auto", "--rollout-concurrency", "4",
             "--rollout-chat-template-kwargs", '{"enable_thinking":false}',
             "--controller-port-base", "15441", "--worker-port-base", "15560",
             "--startup-timeout", "60", "--rollout-timeout", "300", "--eval-timeout", "420"],
            output / "engineer_eval.log")
        results = json.loads((output / "patched/results.json").read_text())
        if len(results) != 1 or not results[0].get("valid_patch") or results[0].get("eval_status") != "ok":
            raise RuntimeError(f"official engineer/patch smoke failed: {results}")
        checked_rows(output / "patched" / results[0]["batch_tag"] / "eval", task_ids)
        if frozen != freeze_baseline(output / "official"):
            raise RuntimeError("CPU protocol smoke baseline mutated")
        report = {"kind": "CPU_protocol_stub_not_model_evaluation", "rows": rows,
                  "output": str(output), "cuda_visible_devices": "", "task_ids": task_ids,
                  "official_engineer_patch_results": results}
        (ASSETS / "cpu_protocol_smoke.json").write_text(json.dumps(report, indent=2))
        (ASSETS / f"cpu_protocol_smoke{args.size}.json").write_text(json.dumps(report, indent=2))
        print(f"Official CPU controller/worker/tool protocol passed: {output}")
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


if __name__ == "__main__":
    main()

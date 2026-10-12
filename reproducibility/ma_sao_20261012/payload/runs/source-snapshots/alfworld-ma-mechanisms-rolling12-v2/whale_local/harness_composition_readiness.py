"""Test composition execution and diversity before outcome optimization."""
import argparse
import ast
import json
import os
from pathlib import Path
import subprocess
import time

from pipeline import ROOT, sha, write_json


def structure(source):
    cls = next(n for n in ast.parse(source).body if isinstance(n, ast.ClassDef))
    hooks = {n.name: ast.dump(n, include_attributes=False) for n in cls.body
        if isinstance(n, (ast.AsyncFunctionDef, ast.FunctionDef))}
    attributes = {}
    for n in cls.body:
        if isinstance(n, ast.Assign):
            for target in n.targets:
                if isinstance(target, ast.Name): attributes[target.id] = ast.literal_eval(n.value)
    return hooks, attributes


def main():
    p = argparse.ArgumentParser(); p.add_argument('--run', type=Path, required=True)
    args = p.parse_args(); run = args.run.resolve()
    deadline = time.monotonic() + 1800
    while not (run / 'editor_sft/complete.json').exists():
        if time.monotonic() > deadline: raise TimeoutError('Initial composition SFT did not finish')
        time.sleep(2)
    expected_hooks, expected_attributes = structure((run / 'development_incumbent.py').read_text())
    checkpoint = run / 'editor_sft/checkpoint'
    for attempt in (1, 2):
        out = run / f'readiness_{attempt:02d}'; out.mkdir(exist_ok=True)
        data = out / 'requests.jsonl'; data.write_bytes((run / 'development_requests.jsonl').read_bytes())
        env = dict(os.environ, CUDA_VISIBLE_DEVICES='6', WHALE_RUN_ROOT=str(out), WHALE_MODEL_ROOT=str(checkpoint),
            WHALE_PARAMETER_DTYPE='float32', WHALE_LEARNING_RATE='1e-5', HF_HUB_OFFLINE='1', TOKENIZERS_PARALLELISM='false',
            PYTHONPATH=':'.join(str(ROOT / p) for p in ['whale_local', 'WHALE/domains/math_reasoning', 'WHALE/domains/chess_puzzles']))
        with (out / 'generate.log').open('a') as log:
            subprocess.run([str(ROOT / '.venv-whale-portable/bin/python'), '-u', str(ROOT / 'whale_local/harness_grpo.py'),
                'generate', '--data', str(data), '--temperature', '1.0'], env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
        subprocess.run([str(ROOT / '.venv-whale-search/bin/python'), str(ROOT / 'whale_local/harness_probe.py'),
            '--input', str(out / 'generations.jsonl'), '--data', str(ROOT / 'runs/whale_math_setup_20260915/mh_val.parquet'),
            '--output', str(out / 'runtime_checks.json')], check=True)
        checks = {r['id']: r for r in json.loads((out / 'runtime_checks.json').read_text())['results']}
        scored = []; instructions = set()
        for row in map(json.loads, (out / 'generations.jsonl').read_text().splitlines()):
            result = {'id': row['id'], 'ready': False}
            try:
                hooks, attributes = structure(row['source'])
                instruction = attributes.get('SYSTEM_PROMPT')
                check = checks[row['id']]
                result['ready'] = bool(check['passed'] and check['recovers_partial_response'] and check['tool_error_nudge']
                    and hooks == expected_hooks and attributes['MAX_TURNS'] == expected_attributes['MAX_TURNS']
                    and isinstance(instruction, str) and instruction and instruction != expected_attributes.get('SYSTEM_PROMPT'))
                if result['ready']: instructions.add(instruction)
            except Exception as exc: result['error'] = repr(exc)
            scored.append(result)
        passed = sum(r['ready'] for r in scored) >= 7 and len(instructions) >= 2
        proof = {'passed_readiness': passed, 'attempt': attempt, 'checkpoint': str(checkpoint),
            'weight_digest': json.loads((checkpoint / 'training_state.json').read_text())['weight_digest'],
            'executable_preserving_changed_guidance': sum(r['ready'] for r in scored),
            'distinct_system_prompts': len(instructions), 'results': scored,
            'protocol_sha256': sha(run / 'PROTOCOL.json'), 'uses_task_rewards': False}
        write_json(out / 'READINESS.json', proof); print(json.dumps(proof), flush=True)
        if passed:
            write_json(run / 'READY.json', proof); return
        if attempt == 1:
            continuation = run / 'editor_sft_extension'
            env.update(WHALE_RUN_ROOT=str(continuation), WHALE_MODEL_ROOT=str(checkpoint))
            with (run / 'extension_train.log').open('a') as log:
                subprocess.run([str(ROOT / '.venv-whale-portable/bin/python'), '-u', str(ROOT / 'whale_local/harness_learner.py'),
                    'train', '--data', str(run / 'train.jsonl'), '--epochs', '2', '--learning-rate', '1e-5'],
                    env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
            checkpoint = continuation / 'checkpoint'
    write_json(run / 'NOT_READY.json', proof)
    raise RuntimeError('Registered composition SFT budget exhausted without code readiness')


if __name__ == '__main__': main()

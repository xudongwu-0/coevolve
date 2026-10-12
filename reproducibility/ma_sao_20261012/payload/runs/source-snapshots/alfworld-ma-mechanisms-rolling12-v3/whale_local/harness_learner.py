"""Separate 2B harness-editor checkpoint: SFT and recorded local generation.

Uses the same native WHALE response-token SFT actor as the task solver, but
never loads solver rollouts into editor SFT. Set WHALE_* worker environment.
"""
import argparse
import asyncio
import ast
import hashlib
import inspect
import json
from pathlib import Path
import random
import time

import torch
import worker as w
from qwen35_padding import install

def read_rows(path):
    return [json.loads(x) for x in Path(path).read_text().splitlines()]

from harness_code_contract import validate_source


def tokenize_row(row):
    if 'prompt_ids' in row:
        assert row['provenance']['kind'] == 'internal_generation_positive_dev_reward'
        return row
    tokenizer = w.state['tokenizer']
    text = tokenizer.apply_chat_template(row['messages'], tokenize=False,
        add_generation_prompt=True, enable_thinking=False)
    prompt = tokenizer(text, add_special_tokens=False)['input_ids']
    completion = tokenizer.encode(row['completion'] + '<|im_end|>', add_special_tokens=False)
    assert len(prompt) + len(completion) <= 4096, 'No silent truncation of harness targets'
    return dict(row, prompt_ids=prompt, response_ids=completion)

def train(args):
    started = time.monotonic()
    rows = read_rows(args.data)
    assert Path(inspect.getfile(w.DataParallelPPOActor)).resolve() == w.ROOT / 'WHALE/domains/math_reasoning/verl/workers/actor/dp_actor.py'
    calls = [tokenize_row(r) for r in rows]
    assert all(r['response_ids'] for r in calls)
    # Each item is a complete code proposal, not a task-solving response.
    before = w.state['parameter_hashes']
    initial = w.identity()
    steps_before = w.optimizer_steps(w.state['optimizer'])
    for group in w.state['optimizer'].param_groups: group['lr'] = args.learning_rate
    total = args.epochs * ((len(calls) + 3) // 4)
    updates = 0
    def progress(optimizer, _args, _kwargs):
        nonlocal updates
        updates += 1
        if updates == 1 or updates % 4 == 0:
            status = {'completed_updates': updates, 'planned_updates': total,
                'elapsed_seconds': time.monotonic() - started}
            w.atomic_json(w.RUN / 'progress.json', status)
            print(json.dumps(status), flush=True)
    hook = w.state['optimizer'].register_step_post_hook(progress)
    metrics = []
    torch.cuda.reset_peak_memory_stats()
    try:
        for epoch in range(args.epochs):
            ordered = list(calls)
            random.Random(1067 + epoch).shuffle(ordered)
            batch = w.make_batch(ordered)
            metrics.append(w.state['actor'].update_sft_policy(batch))
    finally:
        hook.remove()
    after = w.parameter_hashes(w.state['model'])
    changed = [key for key in before if before[key] != after[key]]
    assert changed and updates == total
    assert all(torch.isfinite(p).all() for p in w.state['model'].parameters())
    info = {'role': 'harness_editor', 'method': 'native WHALE SFT actor on code proposals',
        'sampling_identity': initial, 'data': str(args.data.resolve()),
        'data_sha256': hashlib.sha256(args.data.read_bytes()).hexdigest(),
        'examples': len(calls), 'epochs': args.epochs, 'learning_rate': args.learning_rate,
        'response_tokens_per_epoch': sum(len(r['response_ids']) for r in calls),
        'parameter_dtype': w.PARAMETER_DTYPE_NAME, 'parameter_hashes': after,
        'weight_digest': w.digest(after), 'optimizer_steps_before': steps_before,
        'optimizer_steps_after': w.optimizer_steps(w.state['optimizer']),
        'changed_parameter_tensors': changed, 'metrics': metrics,
        'qwen35_padding_compatibility_fix': install(),
        'optimizer_state_dtypes': sorted({str(v.dtype) for s in w.state['optimizer'].state.values()
            for key, v in s.items() if key != 'step' and torch.is_tensor(v)}),
        'peak_gpu_reserved_bytes': torch.cuda.max_memory_reserved(),
        'elapsed_seconds': time.monotonic() - started}
    staging = w.RUN / 'checkpoint.staging'
    staging.mkdir()
    w.state['model'].save_pretrained(staging, max_shard_size='5GB')
    w.state['tokenizer'].save_pretrained(staging)
    torch.save(w.state['optimizer'].state_dict(), staging / 'optimizer.pt')
    torch.save({'cpu': torch.get_rng_state(), 'cuda': torch.cuda.get_rng_state_all()}, staging / 'rng.pt')
    w.atomic_json(staging / 'training_state.json', info)
    staging.rename(w.RUN / 'checkpoint')
    torch.save(calls, w.RUN / 'tokenized_examples.pt')
    # Mandatory same-process reload independently hashes saved tensors and
    # verifies optimizer continuity, just like the solver worker.
    identity = w.load_model(w.RUN / 'checkpoint')
    assert identity['weight_digest'] == info['weight_digest']
    w.atomic_json(w.RUN / 'reload_verification.json', identity)
    w.atomic_json(w.RUN / 'complete.json', {**info, 'reload_verified': True})
    w.atomic_json(w.RUN / 'progress.json', {'completed_updates': updates, 'planned_updates': total,
        'status': 'complete_reload_verified'})
    print(json.dumps({k: info[k] for k in ['examples', 'weight_digest', 'elapsed_seconds', 'peak_gpu_reserved_bytes']}), flush=True)

def generate(args):
    requests = read_rows(args.data)
    output = w.RUN / 'generations.jsonl'
    with output.open('x') as stream:
        for index, row in enumerate(requests):
            torch.manual_seed(args.seed + index)
            torch.cuda.manual_seed_all(args.seed + index)
            started = time.monotonic()
            content, usage, record = w.generate(row['messages'], max_tokens=args.max_tokens,
                temperature=args.temperature, top_p=1.0, stop_at_chat_end=True)
            result = dict(record, id=row['id'], usage=usage, seed=args.seed + index,
                sampling_identity=w.identity(), elapsed_seconds=time.monotonic() - started,
                requested_max_tokens=args.max_tokens)
            try:
                result['source'] = validate_source(content)
                result['valid_source'] = True
                path = w.RUN / (row['id'] + '.py')
                path.write_text(result['source'])
            except Exception as exc:
                result.update(valid_source=False, validation_error=repr(exc))
            stream.write(json.dumps(result) + '\n'); stream.flush()
            print(json.dumps({k: result[k] for k in ['id', 'valid_source', 'usage', 'elapsed_seconds']}), flush=True)
    w.atomic_json(w.RUN / 'complete.json', {'identity': w.identity(), 'requests': len(requests),
        'valid_sources': sum(r['valid_source'] for r in read_rows(output)),
        'generations_sha256': hashlib.sha256(output.read_bytes()).hexdigest()})

async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['train', 'generate'])
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--epochs', type=int, default=2)
    parser.add_argument('--learning-rate', type=float, default=1e-5)
    parser.add_argument('--temperature', type=float, default=0.0)
    parser.add_argument('--max-tokens', type=int, default=1024)
    parser.add_argument('--seed', type=int, default=131)
    args = parser.parse_args()
    install()
    async with w.lifespan(None):
        (train if args.mode == 'train' else generate)(args)

if __name__ == '__main__':
    asyncio.run(main())

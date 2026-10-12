"""Generate short edits, warm up their format, or train on current-solver gains."""
import argparse
import asyncio
import json
import time
from pathlib import Path

import numpy as np
import torch

import worker as w
from harness_delta import compile_edit, anchored_advantages, gain_rank_advantages
from harness_grpo import forward, clipped_loss, read_rows
from pipeline import sha
from qwen35_padding import install
import safe_gradient_norm


def generate(args):
    requests = read_rows(args.data); path = w.RUN / 'generations.jsonl'
    with path.open('w') as stream:
        for i, row in enumerate(requests):
            torch.manual_seed(row['seed']); torch.cuda.manual_seed_all(row['seed'])
            started = time.monotonic()
            content, usage, record = w.generate(row['messages'], max_tokens=384, temperature=args.temperature,
                top_p=1.0, top_k=0, stop_at_chat_end=True, return_token_logprobs=True)
            result = {**record, 'id': row['id'], 'group': row['group'], 'seed': row['seed'],
                'sampling_identity': w.identity(), 'temperature': args.temperature, 'usage': usage,
                'elapsed_seconds': time.monotonic() - started}
            try:
                source, patch, changed = compile_edit(content, Path(row['harness']).read_text())
                result.update(source=source, patch=patch, valid_source=True, edit_changed=changed)
                (w.RUN / (row['id'] + '.py')).write_text(source)
            except Exception as exc:
                result.update(valid_source=False, edit_changed=False, validation_error=repr(exc))
            stream.write(json.dumps(result) + '\n'); stream.flush()
            w.atomic_json(w.RUN / 'progress.json', {'generated': i + 1, 'planned': len(requests)})
            print(json.dumps({k: result[k] for k in ['id', 'valid_source', 'edit_changed', 'usage']}), flush=True)
    w.atomic_json(w.RUN / 'complete.json', {'identity': w.identity(), 'requests': len(requests), 'generations_sha256': sha(path)})


def train(args):
    started = time.monotonic(); calls = read_rows(args.data); initial = w.identity()
    sft = args.mode == 'sft'; method = 'delta-format SFT' if sft else 'current-solver paired-gain delta PPO'
    if not sft and args.advantage_mode == 'rank': method = 'current-solver gain-ranked delta PPO'
    if sft:
        tokenizer = w.state['tokenizer']; end = tokenizer.encode('<|im_end|>', add_special_tokens=False)
        calls = [{**c, 'prompt_ids': tokenizer.apply_chat_template(c['messages'], tokenize=True, return_dict=False, add_generation_prompt=True, enable_thinking=False),
            'response_ids': tokenizer.encode(c['completion'], add_special_tokens=False) + end, 'temperature': 1.0} for c in calls]
        assert all(isinstance(c['prompt_ids'], list) and all(isinstance(t, int) for t in c['prompt_ids']) for c in calls)
        adv = [1.0] * len(calls)
    else:
        assert len(calls) == 8 and len({r['group'] for r in calls}) == 1
        assert all(r['sampling_identity']['weight_digest'] == initial['weight_digest'] for r in calls)
        adv = (gain_rank_advantages if args.advantage_mode == 'rank' else anchored_advantages)(calls)
        w.atomic_json(w.RUN / 'advantages.json', {'advantages': adv, 'reward_deltas': [r['reward_delta'] for r in calls],
            'advantage_mode': args.advantage_mode, 'scale': .125 if args.advantage_mode == 'paired' else 'max(valid std, 1/12)',
            'no_group_mean_subtraction': args.advantage_mode == 'paired'})
        if not any(adv):
            w.atomic_json(w.RUN / 'complete.json', {'updated': False, 'reason': 'no measured nonzero task gain', 'identity': initial}); return
    before = w.parameter_hashes(w.state['model']); inherited = w.optimizer_steps(w.state['optimizer'])
    prior_method = (w.state['training_state'] or {}).get('method')
    reset = prior_method != method
    if not sft and prior_method in {'current-solver paired-gain delta PPO', 'current-solver gain-ranked delta PPO'}:
        reset = False  # The two advantages use the same optimizer; preserve moments for the paired comparison.
    if reset:
        w.state['optimizer'] = torch.optim.AdamW([p for p in w.state['model'].parameters() if p.requires_grad],
            lr=args.learning_rate, betas=(.9, .98), weight_decay=.01)
        w.state['actor'].actor_optimizer = w.state['optimizer']
    optimizer = w.state['optimizer']
    for g in optimizer.param_groups: g['lr'] = args.learning_rate
    steps_before = w.optimizer_steps(optimizer)
    safe_gradient_norm.install(lambda event: w.atomic_json(w.RUN / 'gradient_norm_events.json', safe_gradient_norm.EVENTS))
    old = []
    w.state['model'].eval()
    if not sft:
        with torch.no_grad():
            old = [forward(c).detach().float().cpu() for c in calls]
    batches = [list(range(len(calls)))] if not sft else [list(range(i, min(i + 4, len(calls)))) for i in range(0, len(calls), 4)]
    metrics = []; w.state['model'].train()
    for epoch in range(args.epochs):
        for indices in batches:
            optimizer.zero_grad(set_to_none=True); value = 0.
            for i in indices:
                if adv[i] == 0: continue
                lp = forward(calls[i]).float()
                if sft: loss = -lp.mean()
                else:
                    loss, ratio, correction = clipped_loss(lp, old[i].to('cuda'),
                        torch.tensor(calls[i]['response_logprobs'], device='cuda'), adv[i])
                    # Reference is frozen at the start of this editor phase.
                    difference = lp - old[i].to('cuda')
                    loss = loss + .02 * (torch.exp(difference) - 1 - difference).mean()
                assert torch.isfinite(loss)
                (loss / len(indices)).backward(); value += float(loss.detach()) / len(indices)
            norm = torch.nn.utils.clip_grad_norm_([p for p in w.state['model'].parameters() if p.requires_grad], 1.)
            assert torch.isfinite(norm); optimizer.step()
            metrics.append({'step': len(metrics) + 1, 'loss': value, 'grad_norm': float(norm)})
            w.atomic_json(w.RUN / 'progress.json', {'completed_updates': len(metrics), 'planned_updates': args.epochs * len(batches), 'latest': metrics[-1]})
    optimizer.zero_grad(set_to_none=True)
    assert all(torch.isfinite(p).all() for p in w.state['model'].parameters())
    after = w.parameter_hashes(w.state['model']); changed = [k for k in before if before[k] != after[k]]
    assert changed and w.optimizer_steps(optimizer) == [(max(steps_before) if steps_before else 0) + len(metrics)]
    info = {'role': 'harness_editor', 'method': method, 'sampling_identity': initial, 'data': str(args.data),
        'data_sha256': sha(args.data), 'examples': len(calls), 'new_updates': len(metrics), 'learning_rate': args.learning_rate,
        'optimizer_reset': reset, 'inherited_optimizer_steps': inherited, 'optimizer_steps_before': steps_before,
        'optimizer_steps_after': w.optimizer_steps(optimizer), 'weight_digest': w.digest(after), 'parameter_hashes': after,
        'changed_parameter_tensors': changed, 'parameter_dtype': 'float32', 'metrics': metrics,
        'elapsed_seconds': time.monotonic() - started, 'response_tokens': sum(len(r['response_ids']) for r in calls),
        'objective': {'gain_baseline': 'same current solver, harness, task IDs and solver seed', 'advantage_scale': .125,
            'advantage_clip': 2., 'invalid_and_noop_advantage': 0., 'reference_kl_coefficient': .02, 'sft': sft}}
    info['objective']['advantage_mode'] = args.advantage_mode
    if args.advantage_mode == 'rank':
        info['objective']['advantage_scale'] = 'max(valid-edit reward std, 1/12)'
        info['objective']['ranking_baseline'] = 'max(valid-edit mean gain, 0); zero for a constant-gain group'
    staging = w.RUN / 'checkpoint.staging'; staging.mkdir()
    w.state['model'].save_pretrained(staging, max_shard_size='5GB'); w.state['tokenizer'].save_pretrained(staging)
    torch.save(optimizer.state_dict(), staging / 'optimizer.pt')
    torch.save({'cpu': torch.get_rng_state(), 'cuda': torch.cuda.get_rng_state_all()}, staging / 'rng.pt')
    w.atomic_json(staging / 'training_state.json', info); staging.rename(w.RUN / 'checkpoint')
    loaded = w.load_model(w.RUN / 'checkpoint'); assert loaded['weight_digest'] == info['weight_digest']
    w.atomic_json(w.RUN / 'complete.json', {'updated': True, 'reload_verified': True, 'identity': loaded, 'new_updates': len(metrics)})


async def main():
    p = argparse.ArgumentParser(); p.add_argument('mode', choices=['generate', 'sft', 'train'])
    p.add_argument('--data', type=Path, required=True); p.add_argument('--temperature', type=float, default=1.1)
    p.add_argument('--learning-rate', type=float, default=3e-6); p.add_argument('--epochs', type=int, default=8)
    p.add_argument('--advantage-mode', choices=['paired', 'rank'], default='paired')
    args = p.parse_args(); install()
    async with w.lifespan(None): (generate if args.mode == 'generate' else train)(args)


if __name__ == '__main__': asyncio.run(main())

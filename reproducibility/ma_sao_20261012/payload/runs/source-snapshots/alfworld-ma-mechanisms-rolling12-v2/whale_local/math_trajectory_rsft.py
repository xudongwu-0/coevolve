"""Native RSFT with trajectory minibatches and exact original call contexts.

This diagnostic trainer is separate from all frozen historical experiments.
It preserves every verifier-accepted call and accumulates its token-weighted
gradient in groups of 64 successful trajectories. No new targets are generated.
"""
import argparse
import atexit
import json
import math
import os
from pathlib import Path
import shutil
import time
import uuid


def read(path):
    return json.loads(Path(path).read_text())


def accepted_groups(rows, batch_size=64):
    assert batch_size > 0
    accepted = [row for row in rows if row['reward']]
    assert accepted and all(row['policy_calls'] for row in accepted)
    return [accepted[start:start + batch_size]
            for start in range(0, len(accepted), batch_size)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-run', type=Path, required=True)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--base-model', type=Path, required=True)
    parser.add_argument('--gpu', required=True)
    parser.add_argument('--learning-rate', type=float, default=1e-7)
    parser.add_argument('--trajectory-minibatch', type=int, default=64)
    parser.add_argument('--check-only', action='store_true')
    args = parser.parse_args()
    source, run, base = args.source_run.resolve(), args.run_dir.resolve(), args.base_model.resolve()
    assert args.learning_rate > 0 and args.trajectory_minibatch > 0
    assert not run.exists(), 'Use a fresh diagnostic run directory'
    os.environ.update(CUDA_VISIBLE_DEVICES=str(args.gpu), WHALE_RUN_ROOT=str(run),
        WHALE_MODEL_ROOT=str(base), WHALE_PARAMETER_DTYPE='float32',
        WHALE_LEARNING_RATE=str(args.learning_rate), HF_HUB_OFFLINE='1',
        TRANSFORMERS_OFFLINE='1', TOKENIZERS_PARALLELISM='false')
    import torch
    import pandas as pd
    import pipeline
    from math_reward import binary_reward
    from math_worker import w, PADDING_FIX

    started = time.monotonic()
    source_audit = read(source / 'verification.json')
    assert source_audit['passed_audit'] and source_audit['fp32_master_and_moments']
    config = read(source / 'config.json')
    source_path = source / 'round_01/rollouts/trajectories.jsonl'
    rows = [json.loads(line) for line in source_path.read_text().splitlines()]
    original = read(source / 'initial_identity.json')
    training = read(source / 'round_01/checkpoint/training_state.json')
    assert Path(original['weights']).resolve() == base
    entry = config['data']['splits']['train']
    assert pipeline.sha(entry['path']) == entry['sha256']
    targets = {row['extra_info']['task_id']: row['reward_model']['ground_truth']['target']
               for row in pd.read_parquet(entry['path']).to_dict('records')}
    assert [(row['draw'], row['example_index']) for row in rows] == [
        (draw, index) for draw in range(config['rollouts_per_task']) for index in range(entry['count'])]
    receipt_hashes = {}
    for row in rows:
        assert row['harness_sha256'] == training['harness_sha256']
        assert row['sampling_identity']['weight_digest'] == original['weight_digest']
        assert binary_reward(row['last_assistant'], targets[row['task_id']]) == row['reward']
        for call in row['policy_calls']:
            path = source / 'calls' / (call['call_id'] + '.json')
            assert call == read(path)
            assert call['sampling_identity']['weight_digest'] == original['weight_digest']
            receipt_hashes[str(path)] = pipeline.sha(path)
    groups = accepted_groups(rows, args.trajectory_minibatch)
    total_calls = sum(len(row['policy_calls']) for group in groups for row in group)
    total_tokens = sum(len(call['response_ids']) for group in groups
                       for row in group for call in row['policy_calls'])
    preflight = {'source': str(source), 'source_audit_sha256': pipeline.sha(source / 'verification.json'),
        'source_trajectories_sha256': pipeline.sha(source_path), 'receipt_sha256': receipt_hashes,
        'sampled_trajectories': len(rows), 'accepted_trajectories': sum(map(len, groups)),
        'accepted_policy_calls': total_calls, 'response_tokens': total_tokens,
        'trajectory_minibatch': args.trajectory_minibatch, 'planned_updates': len(groups),
        'all_verified_successes_retained': True, 'original_contexts_preserved': True,
        'learning_rate': args.learning_rate, 'new_generation_calls': 0,
        'check_only': args.check_only, 'confirmation_used': False, 'test_used': False}
    run.mkdir(parents=True)
    pipeline.write_json(run / 'PREFLIGHT.json', preflight)
    if args.check_only:
        print({key: value for key, value in preflight.items() if key != 'receipt_sha256'}, flush=True)
        return
    assert (run / 'PROTOCOL.json').exists() is False
    source_files = [Path(__file__).resolve(), *[pipeline.ROOT / 'whale_local' / name for name in
        ['worker.py', 'math_worker.py', 'qwen35_padding.py', 'math_reward.py']],
        pipeline.ROOT / 'WHALE/domains/math_reasoning/verl/workers/actor/dp_actor.py']
    frozen_sources = {str(path): pipeline.sha(path) for path in source_files}
    for path in source_files:
        target = run / 'source_snapshot' / path.relative_to(pipeline.ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
    pipeline.write_json(run / 'PROTOCOL.json', {
        'scope': 'TRAIN-only optimizer-minibatch diagnostic, not a WHALE/gate qualification',
        'source_sha256': frozen_sources,
        'preflight_sha256': pipeline.sha(run / 'PREFLIGHT.json'), 'base_model': str(base),
        'expected_base_digest': original['weight_digest'], 'native_sft_epochs': 1,
        'trajectory_minibatch': args.trajectory_minibatch, 'microbatch_calls': 1,
        'sampling_data_reuse_is_not_algorithmic_saving': True})
    # Match the proven worker lifespan: the native actor constructor queries
    # torch.distributed even when training on a single device.
    torch.set_num_threads(8)
    torch.manual_seed(42)
    torch.cuda.set_device(0)
    torch.distributed.init_process_group('nccl',
        init_method='file://' + str(run / ('rdzv_' + uuid.uuid4().hex)), rank=0, world_size=1)
    atexit.register(torch.distributed.destroy_process_group)
    initial = w.load_model(base)
    assert initial['weight_digest'] == original['weight_digest'] and initial['optimizer_steps'] == []
    before = w.parameter_hashes(w.state['model'])
    pipeline.write_json(run / 'initial_identity.json', initial)
    pipeline.write_json(run / 'before_parameter_hashes.json', before)
    torch.cuda.reset_peak_memory_stats()
    records, tokens_seen = [], 0
    for number, group in enumerate(groups, 1):
        calls = [call for row in group for call in row['policy_calls']]
        batch = w.make_batch(calls)
        # One optimizer step for this whole trajectory group; the native actor
        # weights each microbatch by its response tokens / group response tokens.
        batch.meta_info.update(sft_mini_batch_size=len(calls), sft_micro_batch_size_per_gpu=1)
        group_dir = run / f'group_{number:03d}'
        group_dir.mkdir()
        torch.save({key: value.cpu() for key, value in batch.batch.items()}, group_dir / 'batch.pt')
        record = {'trajectory_keys': [[row['draw'], row['example_index']] for row in group],
            'call_ids': [call['call_id'] for call in calls],
            'response_tokens': int(batch.batch['response_mask'].sum()),
            'batch_sha256': pipeline.sha(group_dir / 'batch.pt')}
        pipeline.write_json(group_dir / 'input.json', record)
        metrics = w.state['actor'].update_sft_policy(batch)
        assert metrics['actor/sft_updates'] == 1.0
        assert math.isfinite(float(metrics['actor/sft_loss']))
        assert w.optimizer_steps(w.state['optimizer']) == [number]
        record['metrics'] = metrics
        tokens_seen += record['response_tokens']
        records.append(record)
        pipeline.write_json(group_dir / 'result.json', record)
        pipeline.write_json(run / 'training_progress.json', {'completed_updates': number,
            'planned_updates': len(groups), 'response_tokens_seen': tokens_seen,
            'elapsed_seconds': time.monotonic() - started})
        del batch
    assert tokens_seen == total_tokens
    after = w.parameter_hashes(w.state['model'])
    changed = [name for name in before if before[name] != after[name]]
    assert changed and all('visual' not in name for name in changed)
    info = {'method': 'native WHALE math RSFT with trajectory-group gradient accumulation',
        'sampling_identity': initial, 'rollouts': len(rows), 'accepted_trajectories': sum(map(len, groups)),
        'accepted_policy_calls': total_calls, 'response_tokens': total_tokens,
        'parameter_dtype': 'float32', 'learning_rate': args.learning_rate,
        'sft_epochs': 1, 'trajectory_minibatch': args.trajectory_minibatch,
        'metrics': {'actor/sft_updates': len(groups)}, 'groups': records,
        'optimizer_steps_before': [], 'optimizer_steps_after': w.optimizer_steps(w.state['optimizer']),
        'parameter_hashes': after, 'weight_digest': w.digest(after),
        'changed_parameter_tensors': changed, 'qwen35_padding_compatibility_fix': PADDING_FIX,
        'elapsed_seconds': time.monotonic() - started,
        'peak_gpu_allocated_bytes': torch.cuda.max_memory_allocated(),
        'peak_gpu_reserved_bytes': torch.cuda.max_memory_reserved()}
    staging, checkpoint = run / 'checkpoint.staging', run / 'checkpoint'
    staging.mkdir()
    w.state['model'].save_pretrained(staging, max_shard_size='5GB')
    w.state['tokenizer'].save_pretrained(staging)
    torch.save(w.state['optimizer'].state_dict(), staging / 'optimizer.pt')
    torch.save({'cpu': torch.get_rng_state(), 'cuda': torch.cuda.get_rng_state_all()}, staging / 'rng.pt')
    pipeline.write_json(staging / 'training_state.json', info)
    staging.rename(checkpoint)
    reloaded = w.load_model(checkpoint)
    assert reloaded['weight_digest'] == info['weight_digest']
    assert reloaded['optimizer_steps'] == [len(groups)]
    assert all(pipeline.sha(path) == digest for path, digest in frozen_sources.items())
    pipeline.write_json(run / 'reload_verification.json', reloaded)
    torch.distributed.destroy_process_group()
    atexit.unregister(torch.distributed.destroy_process_group)
    pipeline.write_json(run / 'complete.json', {'training_complete': True,
        'checkpoint_reload_verified': True, 'new_generation_calls': 0,
        'confirmation_used': False, 'test_used': False, 'gate_qualified_by_this_run': False})
    print({'checkpoint': str(checkpoint), 'weight_digest': info['weight_digest'],
           'optimizer_updates': len(groups)}, flush=True)


if __name__ == '__main__':
    main()

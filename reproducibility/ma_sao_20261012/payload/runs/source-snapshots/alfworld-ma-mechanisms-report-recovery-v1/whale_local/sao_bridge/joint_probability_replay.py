"""Bounded native fixed-weight likelihood replay; never calls optimizer updates."""
from pathlib import Path
import argparse
from dataclasses import asdict
import math
import os
import socket
import time

from .common import read_json, write_json, file_sha


def replay_row(receipt):
    import torch
    if receipt['role'] != 'editor' or receipt['version'] != 0 or len(receipt['calls']) != 1:
        raise ValueError('Replay requires a single-call first-round Engineer action')
    call = receipt['calls'][0]
    prompt, action = call['prompt_ids'], call['response_ids']
    if not prompt or not action or len(action) != len(call['response_logprobs']):
        raise ValueError('Incomplete authentic token probabilities')
    if any(v != 0 for v in call['versions']):
        raise ValueError('Warmup replay received updated policy tokens')
    ids = torch.tensor([prompt + action], dtype=torch.long)
    mask = torch.zeros_like(ids, dtype=torch.bool)
    mask[:, len(prompt):] = True
    logp = torch.tensor([[0.] * len(prompt) + call['response_logprobs']], dtype=torch.float32)
    return dict(input_ids=ids, attention_mask=torch.ones_like(mask), loss_mask=mask,
                logprobs=logp, versions=torch.where(mask, 0, -1), rewards=torch.tensor([0.]))


def summarize(row, proximal):
    import torch
    mask = torch.roll(row['loss_mask'], -1, -1)
    behavior = torch.roll(row['logprobs'], -1, -1)[mask].double()
    value = proximal.detach().cpu().reshape(row['input_ids'].shape)[mask].double()
    if not torch.isfinite(value).all():
        raise ValueError('Nonfinite recomputed log probabilities')
    delta = value - behavior
    return {'action_tokens': int(mask.sum()), 'log_importance_sum': float(delta.sum()),
            'log_importance_mean': float(delta.mean()),
            'clipped_at_bound2': abs(float(delta.sum())) > math.log(2),
            'action_logprobs': value.tolist()}


def run(output: Path):
    import torch
    import yaml
    from omegaconf import OmegaConf
    from areal.api import FinetuneSpec
    from areal.api.cli_args import PPOConfig, to_structured_cfg
    from areal.engine.fsdp_engine import FSDPPPOActor
    from areal.infra.scheduler.local import LocalScheduler
    from areal.infra.rpc.rtensor import RTensor
    from .resources import gpu_release_audit

    if socket.gethostname() != 'dev-NDH200-026':
        raise RuntimeError('This replay is pinned to idle node4')
    os.environ['AREAL_HOST_IP'] = '127.0.0.1'
    for directory in ('engine', 'name_resolve'):
        (output / directory).mkdir(parents=True, exist_ok=True)
    design = read_json(output / 'protocol.json')
    for name, digest in design['input_sha256'].items():
        if file_sha(Path(name)) != digest:
            raise ValueError(f'Pinned input changed: {name}')
    before = gpu_release_audit()
    write_json(output / 'GPU_BEFORE.json', before)
    if before['foreign_gpu_pids'] or not before['filler_unit_active']:
        raise RuntimeError('Replay requires released node4 and active filler')
    raw = yaml.safe_load(Path(design['trainer_yaml']).read_text())
    # No rollout server is created by this read-only scorer.
    raw['rollout']['admin_api_key'] = 'unused-offline-replay'
    raw['rollout']['agent']['admin_api_key'] = 'unused-offline-replay'
    raw.update(experiment_name=output.name, trial_name='likelihood', total_train_epochs=1,
               total_train_steps=1)
    raw['cluster'].update(fileroot=str(output / 'engine'), n_gpus_per_node=4)
    raw['cluster']['name_resolve']['nfs_record_root'] = str(output / 'name_resolve')
    raw['actor'].update(experiment_name=output.name, trial_name='likelihood', optimizer=None)
    env = {'NCCL_IB_DISABLE': '1', 'OMP_NUM_THREADS': '4', 'GLOO_SOCKET_IFNAME': 'lo',
           'NCCL_SOCKET_IFNAME': 'lo', 'AREAL_HOST_IP': '127.0.0.1', 'TMPDIR': '/dev/shm',
           'PYTHONDONTWRITEBYTECODE': '1', 'PYTORCH_ALLOC_CONF': 'expandable_segments:True',
           'TOKENIZERS_PARALLELISM': 'false'}
    raw['actor']['scheduling_spec'][0]['env_vars'] = env
    config = OmegaConf.to_object(to_structured_cfg(OmegaConf.create(raw), PPOConfig))
    assert config.actor.optimizer is None and config.actor.temperature == .7
    write_json(output / 'CONFIG.json', asdict(config))
    rows = [replay_row(read_json(Path(p))) for p in design['receipts']]
    assert len(rows) == 8
    scheduler = None
    started = time.time()
    try:
        scheduler = LocalScheduler(exp_config=config, gpu_devices=[0, 1, 2, 3])
        actor = FSDPPPOActor.as_controller(config.actor, scheduler)
        actor.create_process_group()
        actor.initialize(role='ref', addr=None,
                         ft_spec=FinetuneSpec(total_train_epochs=1, dataset_size=8, train_batch_size=8))
        results = []
        for order in (list(range(8)), list(range(1, 8)) + [0]):
            scored = RTensor.localize(actor.compute_logp([rows[i] for i in order]))
            if len(scored) != 8:
                raise ValueError('Native scorer omitted trajectories')
            result = {str(i): summarize(rows[i], value) for i, value in zip(order, scored)}
            results.append(result)
            write_json(output / f'PASS_{len(results)}.json', result)
        comparison = []
        for i, original_path in enumerate(design['proximal_records']):
            original = read_json(Path(original_path));a, b = results[0][str(i)], results[1][str(i)]
            assert a['action_tokens'] == original['action_tokens']
            comparison.append({'slot': i, 'action_tokens': a['action_tokens'],
                'original_log_importance_sum': original['log_importance_sum'],
                'replay_log_importance_sum': a['log_importance_sum'],
                'replay_minus_original_sum': a['log_importance_sum'] - original['log_importance_sum'],
                'permutation_sum_change': b['log_importance_sum'] - a['log_importance_sum'],
                'permutation_max_token_difference': max(abs(x-y) for x,y in zip(a['action_logprobs'],b['action_logprobs'])),
                'original_clipped': abs(original['log_importance_sum']) > math.log(2),
                'replay_clipped': a['clipped_at_bound2']})
        write_json(output / 'RESULT.json', {'status': 'completed', 'comparison': comparison,
            'optimizer_updates': 0, 'new_environment_episodes': 0, 'new_generated_tokens': 0,
            'rescored_trajectories': 16, 'elapsed_seconds': time.time()-started,
            'limitations': ['Only eight fixed first-round Engineer trajectories at initial weights.',
                'Probability discrepancy does not by itself prove an accuracy effect.',
                'This separates repeat/batch sensitivity from stored behavior mismatch, not all backend causes.']})
    except BaseException as exc:
        write_json(output / 'FAILED.json', {'error': repr(exc), 'at_unix': time.time()})
        raise
    finally:
        if scheduler is not None:
            scheduler.delete_workers(reverse_order=True)
        for _ in range(15):
            release = gpu_release_audit()
            if not release['foreign_gpu_pids']:
                break
            time.sleep(1)
        write_json(output / 'RELEASED.json', release)
        if release['foreign_gpu_pids'] or not release['filler_unit_active']:
            raise RuntimeError('Replay resource release failed')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    run(parser.parse_args().output.resolve())

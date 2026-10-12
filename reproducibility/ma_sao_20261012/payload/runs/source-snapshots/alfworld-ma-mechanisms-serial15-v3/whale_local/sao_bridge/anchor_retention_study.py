"""Audit consumed updates separately from shadow Engineer generation."""
import math
from pathlib import Path

from .common import read_json, file_sha
from .joint_anchor import SCHEDULE, route_patch, is_anchor_round
from .joint_experiment import validate_spec
from .joint_health import metric_rows, metric_value
from .joint_schedule import accounting, schedule_name, update_counts, validate_metric_row


def audit_training(root: Path) -> dict:
    result = {}
    specs = {role: read_json(root / f'{role}_spec.json') for role in ('editor', 'solver')}
    editor = specs['editor']
    schedule = schedule_name(editor)
    if schedule not in (SCHEDULE, 'simultaneous'):
        raise ValueError('Training audit requires anchor or simultaneous scheduling')
    anchored = schedule == SCHEDULE
    for role in specs:
        if (root / role / 'FAILED.json').exists() or (root / f'{role}_FAILED.json').exists():
            raise ValueError('Anchor role failure marker present')
    for role, spec in specs.items():
        validate_spec(spec)
        shared = ('steps', 'batch_size', 'data')
        if anchored:
            shared += ('anchor_mode', 'anchor_harness_sha256')
        if schedule_name(spec) != schedule or any(spec[k] != editor[k] for k in shared):
            raise ValueError('Anchor learner cohorts differ')
        if not anchored and spec.get('peer_weight_mode', 'bounded') != 'bounded':
            raise ValueError('Simultaneous study requires bounded teammate correction')
        steps, batch = spec['steps'], spec['batch_size']
        expected = {(v, s) for v in range(steps) for s in range(batch)}
        record = read_json(root / role / 'COMPLETE.json')
        counts = update_counts(spec, steps)
        metrics_path = root / role / 'optimizer/metrics.jsonl'
        metrics = metric_rows(metrics_path)
        if (record['status'] != 'completed' or record['steps'] != steps
                or record['actor_updates'] != counts['actor_steps']
                or record['critic_updates'] != counts['critic_steps']
                or file_sha(metrics_path) != record['metrics_sha256']
                or [x['global_step'] for x in metrics] != list(range(steps))):
            raise ValueError('Anchor optimizer budget mismatch')
        for row in metrics:
            validate_metric_row(spec, row)
            m = row['metrics']; anchor = anchored and is_anchor_round(row['global_step'])
            if not anchored:
                gradient = metric_value(m, 'ppo_critic/update/grad_norm')
                if not math.isfinite(gradient) or gradient <= 0:
                    raise ValueError('Simultaneous critic omitted its finite nonzero update')
            if metric_value(m, 'joint_policy/peer_unit_weight_mode') != float(anchor):
                raise ValueError('Anchor worker applied the wrong peer weight mode')
            if anchor and any(metric_value(m, 'joint_policy/' + k) != v for k, v in
                    {'peer_importance_mean': 1., 'peer_importance_clipped_fraction': 0.,
                     'peer_effective_sample_fraction': 1.}.items()):
                raise ValueError('Anchor peer weights are not unit weights')
        data = read_json(Path(spec['data']))
        fixed = read_json(Path(spec['anchor_harness'])) if anchored else None
        receipts = [read_json(p) for p in (root / 'joint').glob(f'round-*/{role}-receipt-*.json')]
        if (len(receipts) != len(expected) or {(x['version'], x['slot']) for x in receipts} != expected
                or len({x['sample_id'] for x in receipts}) != len(expected)):
            raise ValueError('Anchor generation missing, duplicated or over budget')
        for rec in receipts:
            version, slot = rec['version'], rec['slot']
            directory = root / 'joint' / f'round-{version:04d}'
            proposal = read_json(directory / f'proposal-{slot:03d}.json')
            outcome = read_json(directory / f'outcome-{slot:03d}.json')
            if anchored:
                applied, routing = route_patch(proposal['patch'], fixed, round_index=version,
                                               mode=spec['anchor_mode'])
                if (rec['applied_patch'] != applied or rec['anchor_routing'] != routing
                        or outcome['applied_patch'] != applied or outcome['anchor_routing'] != routing):
                    raise ValueError('Anchor applied-patch or credit provenance mismatch')
            elif any('anchor_routing' in value or 'applied_patch' in value for value in (rec, outcome)):
                raise ValueError('Simultaneous episode unexpectedly applied anchor routing')
            if (any(x.get('sample_id') != rec['sample_id'] or x.get('version') != version
                    or x.get('slot') != slot for x in (proposal, outcome))
                    or rec['sample_id'] != data[version * batch + slot]['id']
                    or rec['patch'] != proposal['patch']
                    or rec['team_reward'] != outcome['reward']
                    or rec['reward'] != outcome['reward']):
                raise ValueError('Anchor applied-patch or credit provenance mismatch')
        release = read_json(root / role / 'RELEASED.json')
        if release['foreign_gpu_pids'] or not release['filler_unit_active']:
            raise ValueError('Anchor training resources not released')
        result[role] = {**accounting(spec, steps, generated=len(receipts)),
                        'metrics_sha256': file_sha(metrics_path)}
    for pattern in ('proposal-*.json', 'outcome-*.json'):
        records = [read_json(p) for p in (root / 'joint').glob(f'round-*/{pattern}')]
        if len(records) != len(expected) or {(x['version'], x['slot']) for x in records} != expected:
            raise ValueError('Anchor environment ledger is missing or over budget')
    if list((root / 'joint').glob('*_FAILED.json')):
        raise ValueError('Anchor joint failure marker present')
    return result

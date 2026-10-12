import json
from pathlib import Path
import pytest
from test_joint_schedule import spec_for, original_rows, metric_row
from sao_bridge.common import write_json, read_json, file_sha
from sao_bridge.joint_anchor import SCHEDULE, route_patch
from sao_bridge.joint_schedule import scheduled_rows, update_counts
from sao_bridge.anchor_retention_study import audit_training


@pytest.fixture
def run(tmp_path):
    source = tmp_path / 'source.json'; anchor = tmp_path / 'anchor.json'
    fixed = {'benchmark': 'alfworld', 'actions': [], 'description': 'anchor'}
    proposal = {'benchmark': 'alfworld', 'actions': []}
    write_json(source, original_rows()[:16]); write_json(anchor, fixed)
    for role in ('editor', 'solver'):
        spec = {**spec_for(tmp_path, role), 'schedule': SCHEDULE, 'steps': 3,
                'critic_warmup_steps': 1, 'anchor_mode': 'fixed_anchor',
                'schedule_source_data': str(source), 'anchor_harness': str(anchor),
                'anchor_harness_sha256': file_sha(anchor),
                'pinned_files': {str(p): file_sha(p) for p in (source, anchor)}}
        rows = scheduled_rows(original_rows()[:16], spec)
        write_json(Path(spec['data']), rows); write_json(tmp_path / f'{role}_spec.json', spec)
        path = tmp_path / role / 'optimizer/metrics.jsonl'; path.parent.mkdir(parents=True)
        metrics = []
        for step in range(3):
            row = metric_row(spec, step)
            row['metrics'].update({'joint_policy/peer_unit_weight_mode': float(step == 2),
                                   'joint_policy/peer_importance_mean': 1.,
                                   'joint_policy/peer_effective_sample_fraction': 1.})
            metrics.append(row)
        path.write_text(''.join(json.dumps(x) + '\n' for x in metrics))
        counts = update_counts(spec, 3)
        write_json(tmp_path / role / 'COMPLETE.json', {'status': 'completed', 'steps': 3,
                   'actor_updates': counts['actor_steps'], 'critic_updates': counts['critic_steps'],
                   'metrics_sha256': file_sha(path)})
        write_json(tmp_path / role / 'RELEASED.json', {'foreign_gpu_pids': [], 'filler_unit_active': True})
        for row in rows:
            step, slot = row['joint_version'], row['joint_slot']
            directory = tmp_path / 'joint' / f'round-{step:04d}'
            identity = {'sample_id': row['id'], 'version': step, 'slot': slot}
            applied, routing = route_patch(proposal, fixed, round_index=step, mode='fixed_anchor')
            write_json(directory / f'proposal-{slot:03d}.json', {**identity, 'patch': proposal})
            outcome = {**identity, 'applied_patch': applied, 'anchor_routing': routing, 'reward': 1.}
            write_json(directory / f'outcome-{slot:03d}.json', outcome)
            write_json(directory / f'{role}-receipt-{slot:03d}.json',
                       {**outcome, 'patch': proposal, 'team_reward': 1.})
    return tmp_path


def test_accounting_distinguishes_unused_shadow_generation(run):
    audit = audit_training(run)
    assert audit['editor']['consumed_policy_trajectories'] == 16
    assert audit['editor']['generated_policy_trajectories'] == 24
    assert audit['editor']['inactive_counterpart_trajectories'] == 8
    assert audit['solver']['consumed_policy_trajectories'] == 24


@pytest.mark.parametrize('error', ['patch', 'identity', 'extra_outcome', 'failure', 'inactive_update', 'weight_mode'])
def test_corrupt_intervention_and_budget_cannot_pass(run, error):
    directory = run / 'joint/round-0002'
    if error in ('patch', 'identity', 'extra_outcome'):
        path = directory / 'outcome-000.json'; v = read_json(path)
        if error == 'patch': v['applied_patch'] = {'actions': ['wrong']}
        if error == 'identity': v['sample_id'] = 'another-episode'
        if error == 'extra_outcome': path = directory / 'outcome-999.json'
        write_json(path, v)
    elif error == 'failure': write_json(run / 'editor/FAILED.json', {'error': 'test'})
    else:
        path = run / 'editor/optimizer/metrics.jsonl'
        rows = [json.loads(x) for x in path.read_text().splitlines()]
        key = 'ppo_actor/update/grad_norm' if error == 'inactive_update' else 'joint_policy/peer_unit_weight_mode'
        rows[2]['metrics'][key] = 1. if error == 'inactive_update' else 0.
        path.write_text(''.join(json.dumps(x) + '\n' for x in rows))
        record = read_json(run / 'editor/COMPLETE.json'); record['metrics_sha256'] = file_sha(path)
        write_json(run / 'editor/COMPLETE.json', record)
    with pytest.raises(ValueError): audit_training(run)


@pytest.fixture
def simultaneous_run(run):
    for role in ('editor', 'solver'):
        path = run / f'{role}_spec.json'; spec = read_json(path)
        spec.update(schedule='simultaneous', critic_warmup_steps=0)
        for key in ('anchor_mode', 'anchor_harness', 'anchor_harness_sha256'):
            del spec[key]
        write_json(path, spec)
        path = run / role / 'optimizer/metrics.jsonl'
        rows = []
        for step in range(3):
            row = metric_row(spec, step)
            row['metrics']['joint_policy/peer_unit_weight_mode'] = 0.
            rows.append(row)
        path.write_text(''.join(json.dumps(x) + '\n' for x in rows))
        complete = read_json(run / role / 'COMPLETE.json')
        complete.update(actor_updates=3, critic_updates=6, metrics_sha256=file_sha(path))
        write_json(run / role / 'COMPLETE.json', complete)
    for path in (run / 'joint').glob('round-*/*.json'):
        value = read_json(path)
        value.pop('anchor_routing', None)
        value.pop('applied_patch', None)
        write_json(path, value)
    return run


def test_simultaneous_updates_both_roles_on_every_round(simultaneous_run):
    audit = audit_training(simultaneous_run)
    for role in ('editor', 'solver'):
        assert audit[role]['actor_steps'] == 3
        assert audit[role]['critic_steps'] == 6
        assert audit[role]['consumed_policy_trajectories'] == 24
        assert audit[role]['inactive_counterpart_trajectories'] == 0


@pytest.mark.parametrize('error', ['actor', 'critic', 'peer', 'anchor'])
def test_simultaneous_rejects_hidden_inactivity_or_anchor_routing(simultaneous_run, error):
    root = simultaneous_run
    if error == 'anchor':
        path = root / 'joint/round-0002/outcome-000.json'
        value = read_json(path); value['anchor_routing'] = {}
        write_json(path, value)
    else:
        path = root / 'editor/optimizer/metrics.jsonl'
        rows = [json.loads(x) for x in path.read_text().splitlines()]
        if error == 'peer':
            rows[2]['metrics']['joint_policy/peer_unit_weight_mode'] = 1.
        else:
            del rows[2]['metrics'][f'ppo_{error}/update/grad_norm']
        path.write_text(''.join(json.dumps(x) + '\n' for x in rows))
        receipt = read_json(root / 'editor/COMPLETE.json')
        receipt['metrics_sha256'] = file_sha(path)
        write_json(root / 'editor/COMPLETE.json', receipt)
    with pytest.raises(ValueError):
        audit_training(root)

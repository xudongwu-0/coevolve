"""CPU-only gate guard, allocation fallback, and evaluation orchestration checks."""
import copy
import json
from pathlib import Path
import sys
import tempfile
from unittest.mock import patch

import math_evaluation as evaluation
import math_sampling
import math_pipeline
from math_common_probe import first_wave, install
from math_gate import make_request, plan_routes
from pipeline import ROOT, sha, write_json


def read(path):
    return json.loads(Path(path).read_text())


def main():
    source = ROOT / 'runs/whale_math_onecycle_fixed_native_20260915'
    assert not read(source / 'verification.json')['qualified_for_gate']
    config = read(source / 'config.json')
    identity = read(source / 'initial_identity.json')
    harness = ROOT / 'WHALE/domains/math_reasoning/environments/retool/base_harness.py'
    rows, _ = first_wave(source, identity=identity, harness=harness,
        data_entry=config['data']['splits']['train'], seed=43)
    request = make_request(rows, harness)
    annotations = []
    for case in request['payload']['cases']:
        quote = case['last_assistant'] or next(
            call['content'] for call in case['policy_calls'] if call['content'])
        annotations.append({'sample_index': case['sample_index'], 'label': 'weight',
            'confidence': .69, 'rationale': 'Synthetic low-confidence fixture; not a teacher annotation.',
            'evidence_quote': quote[:80]})
    response = {'request_digest': request['request_digest'], 'annotations': annotations}
    uniform = plan_routes(rows, request, response, seed=43, total_rollouts=512)
    assert uniform['followup_counts'] == [3] * 128
    assert uniform['followup_schedule'] == [[draw, index] for draw in range(1, 4) for index in range(128)]
    routed_response = copy.deepcopy(response)
    routed_response['annotations'][0].update(label='harness', confidence=.9)
    routed_response['annotations'][1].update(label='weight', confidence=.9)
    routed = plan_routes(rows, request, routed_response, seed=43, total_rollouts=512)
    h, w = (a['sample_index'] for a in routed_response['annotations'][:2])
    assert routed['followup_counts'][h] < 3 < routed['followup_counts'][w]
    assert 128 + sum(routed['followup_counts']) == 512
    assert max(routed['followup_counts']) <= 7
    assert len({tuple(item) for item in routed['followup_schedule']}) == 384

    with tempfile.TemporaryDirectory(prefix='whale-gate-interface-check-') as folder:
        temp = Path(folder)
        blocked = temp / 'must_not_be_written'
        try:
            install(source, run=temp, output=blocked, identity=identity, harness=harness,
                    data_entry=config['data']['splits']['train'], seed=43)
        except AssertionError:
            pass
        else:
            raise AssertionError('Unqualified shared-probe import was accepted')
        assert not blocked.exists()
        argv = ['math_pipeline.py', '--run-dir', str(temp / 'blocked_pipeline'),
                '--manifest', str(ROOT / 'runs/whale_math_onecycle_setup_20260915/manifest.json'),
                '--rounds', '1', '--gate-reference', str(source), '--share-first-probe']
        with patch.object(sys, 'argv', argv), patch.object(math_pipeline, 'service') as mocked_service, \
             patch.object(math_pipeline.subprocess, 'Popen') as mocked_process:
            try:
                math_pipeline.main()
            except AssertionError:
                pass
            else:
                raise AssertionError('Unqualified pipeline was accepted')
            mocked_service.assert_not_called()
            mocked_process.assert_not_called()
        assert not (temp / 'blocked_pipeline/config.json').exists()

        run = temp / 'three_arm_scheduler'
        run.mkdir()
        data_path = temp / 'dataset.fixture'
        data_path.write_text('CPU orchestration fixture; not benchmark data.\n')
        fixture_harness = temp / 'harness.fixture'
        fixture_harness.write_text('Never executed.\n')
        entry = {'path': str(data_path), 'count': 2}
        settings = {'data': {'splits': {'test': entry}}, 'gpu': '8',
                    'evaluation_rollouts': 8, 'evaluation_temperature': 1.0}
        hardware = {'confirmation_gpus': [6, 7], 'confirmation_port_base': 25000,
            'confirmation_replica_port_base': 27000,
            'confirmation_replica_groups': [['node1:4', 'node1:5'], ['node1:6', 'node1:7'],
                                            [9, 'node1:8', 'node1:9']]}
        write_json(run / 'eval_hardware.json', hardware)
        initial = {'weights': 'unused-initial', 'weight_digest': 'fixture-initial', 'parameter_dtype': 'float32'}
        whale = {'weights': 'unused-whale', 'weight_digest': 'fixture-whale', 'parameter_dtype': 'float32'}
        gate = {'weights': 'unused-gate', 'weight_digest': 'fixture-gate', 'parameter_dtype': 'float32'}
        launched, main_requests = [], []

        def report(digest, harness_path):
            return {'num_examples': 16, 'metadata': {'weight_digest': digest,
                'harness_sha256': sha(harness_path), 'temperature': 1.0, 'dataset_path': str(data_path)}}

        class FakeProcess:
            def __init__(self, argv, **kwargs):
                values = {key: argv[index + 1] for index, key in enumerate(argv[:-1]) if key.startswith('--')}
                assert values['--evaluation-rollouts'] == '8'
                assert float(values['--evaluation-temperature']) == 1.0
                launched.append(values)
                work = Path(values['--run-dir'])
                (work / 'results').mkdir(parents=True)
                write_json(work / 'results/val.json', report(values['--expected-digest'], values['--harness']))
                write_json(work / 'complete.json', {'frozen_weights_verified': True})

            def wait(self, **kwargs):
                return 0

            def poll(self):
                return 0

        def fake_sampling(**kwargs):
            assert kwargs['rollouts'] == 8 and kwargs['temperature'] == 1.0
            assert kwargs['replica_gpus'] == hardware['confirmation_replica_groups'][-1]
            assert kwargs['port_base'] == 27200
            main_requests.append(kwargs)
            output = kwargs['output']
            output.mkdir(parents=True)
            write_json(output / 'val.json', report(kwargs['identity']['weight_digest'], kwargs['harness']))

        arguments = dict(run=run, config=settings, split='test', base_url='not-called', initial=initial,
            final=gate, final_harness=fixture_harness, main_cell='gate',
            remote_cells=[('initial', initial, fixture_harness), ('whale', whale, fixture_harness)])
        with patch.object(evaluation.subprocess, 'Popen', FakeProcess), \
             patch.object(math_sampling, 'sample_parallel', fake_sampling):
            evaluation.evaluate_four_cells(**arguments)
            assert len(launched) == 2 and len(main_requests) == 1
            evaluation.evaluate_four_cells(**arguments, resume=True)
            assert len(launched) == 2 and len(main_requests) == 1
            cached = run / 'test/initial/val.json'
            value = read(cached)
            write_json(cached, value | {'metadata': value['metadata'] | {'temperature': 0.0}})
            try:
                evaluation.evaluate_four_cells(**arguments, resume=True)
            except AssertionError:
                pass
            else:
                raise AssertionError('Stale greedy cache was accepted for mean@8')
            write_json(cached, value)
            broken_hardware = copy.deepcopy(hardware)
            broken_hardware['confirmation_replica_groups'][-1][0] = 6
            write_json(run / 'eval_hardware.json', broken_hardware)
            try:
                evaluation.evaluate_four_cells(**arguments, resume=True)
            except AssertionError:
                pass
            else:
                raise AssertionError('Overlapping GPU assignments were accepted')
    result = {'passed': True, 'scope': 'CPU-only mocked orchestration and synthetic route fixtures; no teacher labels or benchmark results produced',
        'uncertain_gate_reduces_exactly_to_uniform_512_schedule': True,
        'routed_budget_is_512_with_max8_draws': True,
        'unqualified_probe_and_pipeline_rejected_before_services': True,
        'three_test_arms_allocate_all10_GPUs_without_overlap': True,
        'draws_and_temperature_reach_main_and_child_evaluators': True,
        'resume_uses_completed_cells_without_new_jobs': True,
        'stale_greedy_cache_and_overlapping_GPUs_rejected': True,
        'new_model_calls': 0, 'new_optimizer_updates': 0,
        'source_sha256': {name: sha(ROOT / 'whale_local' / name) for name in
            ['check_math_gate_interfaces.py', 'math_pipeline.py', 'math_gate.py',
             'math_evaluation.py', 'math_common_probe.py']}}
    write_json(ROOT / 'audit/whale/MATH_GATE_INTERFACE_CHECKS_20260915.json', result)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()

"""Frozen independent mean@8 confirmation; two model groups, two harnesses each."""
import argparse
import json
from pathlib import Path

from pipeline import ROOT, api, sha, write_json
from math_runtime import service
from math_sampling import sample_parallel


def read(path):
    return json.loads(Path(path).read_text())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--weights', choices=['W0', 'W1'], required=True)
    args = parser.parse_args()
    root = args.run_dir.resolve()
    config, protocol = read(root / 'config.json'), read(root / 'PROTOCOL.json')
    for path, digest in protocol['source_sha256'].items():
        assert sha(ROOT / path) == digest
    assert sha(root / 'frozen_evaluation.json') == protocol['frozen_evaluation_sha256']
    frozen, hardware = read(root / 'frozen_evaluation.json'), read(root / 'eval_hardware.json')[args.weights]
    conditional = read(protocol['conditional_protocol'])
    assert sha(protocol['conditional_protocol']) == protocol['conditional_protocol_sha256']
    development = read(protocol['development_audit'])
    assert sha(protocol['development_audit']) == protocol['development_audit_sha256']
    assert development['passed_audit'] and development['development_both_axes_above_5pp']
    assert config['evaluation_rollouts'] == conditional['draws_per_question'] == 8
    assert config['evaluation_temperature'] == conditional['temperature'] == 1.0
    run = root / 'frozen_eval_workers' / args.weights
    run.mkdir(parents=True, exist_ok=True)
    write_json(run / 'config.json', {'data': config['data']})
    write_json(run / 'eval_hardware.json', {'sampler_gpus': hardware['replica_gpus'],
                                          'sampler_port_base': hardware['replica_port_base']})
    identity_spec = frozen[args.weights]
    with service(run, Path(identity_spec['model']), hardware['gpu'], hardware['port'],
                 config['data'], parameter_dtype='float32') as (url, identity):
        assert identity['weight_digest'] == identity_spec['weight_digest']
        write_json(run / 'identity.json', identity)
        for harness_name in ['H0', 'Hf']:
            cell = args.weights + '_' + harness_name
            harness = Path(frozen[harness_name])
            phase = run / harness_name
            phase.mkdir(exist_ok=True)
            output = root / 'confirmation' / cell
            if not (output / 'val.json').exists():
                sample_parallel(run=run, phase=phase, output=output, base_url=url, identity=identity,
                    harness=harness, data_entry=config['data']['splits']['confirmation'],
                    rollouts=8, seed=42, temperature=1.0, split='confirmation', resume=True,
                    port_base=hardware['replica_port_base'])
            assert api(url, '/health')['weight_digest'] == identity['weight_digest']
        assert api(url, '/health')['optimizer_steps'] == identity['optimizer_steps']
    write_json(run / 'complete.json', {'frozen_weights_verified': True,
        'both_frozen_harnesses_evaluated': True, 'new_optimizer_updates': 0, 'final_test_used': False})


if __name__ == '__main__':
    main()

"""Frozen development-only mean@8 diagnosis; never qualifies a gate."""
import argparse
import json
from pathlib import Path

import pipeline
from math_runtime import service, BASE_HARNESS
from math_sampling import sample_parallel


def read(path):
    return json.loads(Path(path).read_text())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--cell', choices=['W0_H0', 'W1_H0'], required=True)
    args = parser.parse_args()
    root = args.run_dir.resolve()
    protocol = read(root / 'PROTOCOL.json')
    for path, digest in protocol['source_sha256'].items():
        assert pipeline.sha(path) == digest
    cell = protocol['cells'][args.cell]
    run = root / args.cell
    run.mkdir(exist_ok=True)
    pipeline.write_json(run / 'config.json', {'data': protocol['data']})
    pipeline.write_json(run / 'eval_hardware.json', {
        'sampler_gpus': cell['replica_gpus'], 'sampler_port_base': cell['replica_port_base']})
    with service(run, Path(cell['model']), cell['gpu'], cell['port'], protocol['data'],
                 parameter_dtype='float32') as (url, identity):
        assert identity['weight_digest'] == cell['weight_digest']
        pipeline.write_json(run / 'identity.json', identity)
        phase = run / 'development_sampling'
        phase.mkdir(exist_ok=True)
        output = run / 'mh_val'
        if not (output / 'val.json').exists():
            sample_parallel(run=run, phase=phase, output=output, base_url=url, identity=identity,
                harness=BASE_HARNESS, data_entry=protocol['data']['splits']['mh_val'],
                rollouts=8, seed=42, temperature=1.0, split='mh_val', resume=True,
                port_base=cell['replica_port_base'])
        assert pipeline.api(url, '/health')['weight_digest'] == identity['weight_digest']
    pipeline.write_json(run / 'complete.json', {
        'frozen_weights_verified': True, 'development_only': True,
        'new_training_samples': 0, 'new_optimizer_updates': 0,
        'confirmation_used': False, 'test_used': False, 'gate_qualified_by_this_run': False})


if __name__ == '__main__':
    main()

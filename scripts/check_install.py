"""CPU-only installation checks. Does not load a model or start training."""
import argparse
import importlib
import importlib.metadata
import inspect
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
UPSTREAM = 'fbe125eb7abea7f760c99ab9acc1a6261e708fc6'
sys.path[:0] = [str(ROOT/'whale_local'), str(ROOT/'WHALE/domains/math_reasoning'),
               str(ROOT/'WHALE/domains/chess_puzzles')]


def main(role, sandbox):
    commit = subprocess.check_output(['git', '-C', str(ROOT/'WHALE'), 'rev-parse', 'HEAD'], text=True).strip()
    if commit != UPSTREAM:
        raise RuntimeError(f'Expected WHALE {UPSTREAM}, found {commit}')
    result = {'role': role, 'upstream_commit': commit, 'gpu_model_loads': 0}
    if role == 'coordinator':
        for name in ['math_runtime', 'alternating_pool', 'three_arm_hwh']:
            importlib.import_module(name)
        from harness_code_contract import validate_source
        validate_source((ROOT/'examples/initial_harness.py').read_text())
        from math_reward import binary_reward
        assert binary_reward(r'\boxed{42}', '42') == 1.0
        assert binary_reward(r'\boxed{41}', '42') == 0.0
        result['verifiers'] = importlib.metadata.version('verifiers')
        result['native_reward_checks'] = True
    else:
        # Imports require worker paths, but lifespan (which loads CUDA models)
        # is not entered. A temporary output directory has no model assets.
        with tempfile.TemporaryDirectory(prefix='coevolve-import-') as directory:
            os.environ['WHALE_RUN_ROOT'] = directory
            os.environ['WHALE_MODEL_ROOT'] = directory
            import worker
            import math_worker
            import harness_grpo
            import harness_learner
            import numpy as np
            import torch
            expected = ROOT/'WHALE/domains/math_reasoning/verl/workers/actor/dp_actor.py'
            assert Path(inspect.getfile(worker.DataParallelPPOActor)).resolve() == expected
            advantage, _ = harness_grpo.advantages([0, 1, 2, 3], ['a']*4)
            assert np.isclose(advantage.mean(), 0) and np.isclose(advantage.std(), 1)
            zero, _ = harness_grpo.advantages([1, 1], ['a', 'a'])
            assert not np.any(zero)
            loss, _, _ = harness_grpo.clipped_loss(torch.zeros(2), torch.zeros(2), torch.zeros(2), 1.0)
            assert loss.item() == -1.0
            result.update(torch=torch.__version__, transformers=importlib.metadata.version('transformers'),
                          native_actor_import=True, editor_objective_checks=True)
            if sandbox:
                if not shutil.which('bwrap'):
                    raise RuntimeError('Install bubblewrap to run the math code tool')
                from math_sandbox import run_python
                response = run_python({'code': 'print(sum([1, 2, 3]))'})
                assert response['status'] == 'Success', response
                assert response['run_result']['stdout'].strip() == '6', response
                result['sandbox_execution'] = True
    print(json.dumps(result))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--role', choices=['coordinator', 'worker'], required=True)
    parser.add_argument('--sandbox', action='store_true')
    args = parser.parse_args()
    main(args.role, args.sandbox)

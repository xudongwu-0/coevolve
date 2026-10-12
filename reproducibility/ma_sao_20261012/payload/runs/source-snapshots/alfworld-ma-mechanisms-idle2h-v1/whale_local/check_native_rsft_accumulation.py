"""CPU gradient oracle for native RSFT accumulation over unequal call lengths."""
import copy
import json
import os
from pathlib import Path
from types import MethodType, SimpleNamespace

os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ.setdefault('WHALE_RUN_ROOT', '/tmp/whale-rsft-gradient-oracle')
os.environ.setdefault('WHALE_MODEL_ROOT', '/tmp/unused-model-no-load')

import torch
import torch.nn.functional as F
import verl.workers.actor.dp_actor as native
import worker as w
from pipeline import ROOT, sha, write_json


def features(tokens):
    x = tokens.to(torch.float64)
    return torch.stack((torch.sin(x), torch.cos(x), x / 10), dim=-1)


def main():
    calls = [{'prompt_ids': [6, 8], 'response_ids': [1]},
             {'prompt_ids': [6, 9, 7], 'response_ids': [4, 3, 2]},
             {'prompt_ids': [8], 'response_ids': [0, 1, 4, 2, 3, 1, 2]}]
    w.state['tokenizer'] = SimpleNamespace(pad_token_id=0)
    model = torch.nn.Linear(3, 5, bias=False, dtype=torch.float64)
    with torch.no_grad():
        model.weight.copy_(torch.arange(15, dtype=torch.float64).reshape(5, 3) / 20)
    expected = copy.deepcopy(model)
    optimizer = torch.optim.SGD(expected.parameters(), lr=.05)
    losses = []
    for call in calls:
        previous = torch.tensor([call['prompt_ids'][-1]] + call['response_ids'][:-1])
        losses.append(F.cross_entropy(expected(features(previous)),
                                      torch.tensor(call['response_ids']), reduction='none'))
    token_loss = torch.cat(losses).mean()
    token_loss.backward()
    optimizer.step()
    # A per-call average is a plausible but incorrect implementation. The fixture
    # must distinguish it from the native token-weighted objective.
    incorrect = copy.deepcopy(model)
    wrong_losses = [F.cross_entropy(incorrect(features(torch.tensor(
        [call['prompt_ids'][-1]] + call['response_ids'][:-1]))),
        torch.tensor(call['response_ids'])) for call in calls]
    torch.stack(wrong_losses).mean().backward()
    torch.optim.SGD(incorrect.parameters(), lr=.05).step()
    wrong_difference = float((incorrect.weight - expected.weight).abs().max().detach())
    assert wrong_difference > 1e-5
    original_device = native.get_device_id
    native.get_device_id = lambda: 'cpu'
    results = []
    try:
        for microbatch in [1, 3]:
            actor = native.DataParallelPPOActor.__new__(native.DataParallelPPOActor)
            actor.actor_module = copy.deepcopy(model)
            actor.actor_optimizer = torch.optim.SGD(actor.actor_module.parameters(), lr=.05)
            actor.config = SimpleNamespace(ppo_mini_batch_size=3,
                ppo_micro_batch_size_per_gpu=microbatch, use_dynamic_bsz=False, grad_clip=1e9)
            actor.use_prefix_grouper, actor.scaler = False, None

            def forward(self, inputs, temperature, calculate_entropy):
                responses = inputs['responses']
                previous = inputs['input_ids'][:, -responses.shape[1] - 1:-1]
                logits = self.actor_module(features(previous)) / temperature
                return {'log_probs': F.log_softmax(logits, dim=-1).gather(
                    -1, responses.unsqueeze(-1)).squeeze(-1)}

            actor._forward_micro_batch = MethodType(forward, actor)
            batch = w.make_batch(calls)
            batch.meta_info.update(sft_mini_batch_size=3,
                                   sft_micro_batch_size_per_gpu=microbatch)
            metrics = actor.update_sft_policy(batch)
            error = float((actor.actor_module.weight - expected.weight).abs().max().detach())
            # Native integer token-count division produces FP32 proportions,
            # even for this FP64 toy model. Each feature/softmax derivative is
            # bounded by 1 here, so sum of three ratio-rounding errors times
            # the SGD learning rate gives a conservative parameter bound.
            tolerance = len(calls) * torch.finfo(torch.float32).eps * .05
            assert error <= tolerance and metrics['actor/sft_updates'] == 1
            assert metrics['actor/sft_token_count'] == 11
            results.append({'microbatch': microbatch, 'max_parameter_error_vs_token_CE_oracle': error,
                            'fp32_ratio_rounding_bound': tolerance,
                            'optimizer_updates': metrics['actor/sft_updates']})
    finally:
        native.get_device_id = original_device
    result = {'passed': True,
        'scope': 'CPU toy-model gradient oracle; actual native SFT accumulation and optimizer methods, stubbed model forward; not a model accuracy test',
        'native_actor_sha256': sha(Path(native.__file__)),
        'check_source_sha256': sha(__file__), 'response_lengths': [1, 3, 7],
        'results': results, 'incorrect_per_call_average_detected': wrong_difference,
        'gpu_used': False, 'benchmark_generations': 0}
    write_json(ROOT / 'audit/whale/NATIVE_RSFT_ACCUMULATION_CHECK_20260915.json', result)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()

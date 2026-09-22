"""Bounded synthetic continuation for editing an already composed harness.

Multiple alternatives deliberately share each input. Targets teach the edit
interface, not measured task rewards; outcome GRPO decides their utility later.
"""
import argparse
import ast
import datetime
import hashlib
import json
from pathlib import Path

from harness_bootstrap import START, FAMILIES
from harness_code_contract import validate_source
from harness_focused_bootstrap import request


def guidance(source, instruction):
    tree = ast.parse(source)
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef))
    lines = source.splitlines(keepends=True)
    existing = next((n for n in cls.body if isinstance(n, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == 'SYSTEM_PROMPT' for t in n.targets)), None)
    line = '    SYSTEM_PROMPT = ' + repr(instruction) + '\n'
    if existing:
        lines[existing.lineno - 1:existing.end_lineno] = [line]
    else:
        lines.insert(cls.lineno, line)
    return validate_source(''.join(lines))


def main():
    p = argparse.ArgumentParser(); p.add_argument('--out', type=Path, required=True)
    p.add_argument('--prior', type=Path, required=True); args = p.parse_args()
    out = args.out.resolve(); out.mkdir(parents=True, exist_ok=False)
    prior = args.prior.resolve()
    h1 = (prior / 'internal_mh/harnesses/h1/harness.py').read_text()
    h3 = (prior / 'internal_mh/harnesses/h3/harness.py').read_text()
    sources = {'start': START, 'finalize': h1, 'composed': h3,
        'composed_with_guidance': guidance(h3, 'Solve the problem and provide a final boxed answer.')}
    alternatives = [('concise', 0), ('verify', 0), ('exact', 0), ('bounds', 0),
        ('print', 0), ('quantity', 0), ('concise', 1), ('verify', 1)]
    packets = [
        {'tasks': 8, 'solved': 5, 'missing_boxed': 1, 'failure_observations': [
            {'assistant_tail': 'I should check this intermediate calculation before finishing.', 'tool_feedback': []},
            {'assistant_tail': 'Let me repeat the derivation and examine the remaining cases.', 'tool_feedback': ['Execution completed.']}]},
        {'tasks': 8, 'solved': 6, 'missing_boxed': 0, 'failure_observations': [
            {'assistant_tail': 'The calculation is complete, but the requested quantity needs one more check.', 'tool_feedback': ['No visible output.']},
            {'assistant_tail': 'The enumeration uses an assumed search interval and decimal arithmetic.', 'tool_feedback': []}]}]
    rows = []
    for name, source in sources.items():
        for i, packet in enumerate(packets):
            for j, (family, wording) in enumerate(alternatives):
                target = guidance(source, FAMILIES[family][1][wording])
                rows.append({'id': f'{name}_{i}_{j}', 'family': family,
                    'messages': request(source, json.dumps(packet), 'strategy'), 'completion': target,
                    'provenance': 'Codex-authored synthetic compositional edit demonstration; not reward-verified',
                    'uses_benchmark_questions_or_answers': False})
    (out / 'train.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in rows))
    dev_packet = {'tasks': 8, 'solved': 6, 'missing_boxed': 1, 'failure_observations': [
        {'assistant_tail': 'An intermediate result was computed, but the final response did not establish the requested result.',
         'tool_feedback': ['A calculation completed without a useful consistency check.']}]}
    dev = [{'id': f'composition_{i:02d}', 'group': 'composition',
        'messages': request(h3, json.dumps(dev_packet), 'strategy'), 'seed': 90311 + i} for i in range(8)]
    (out / 'development_requests.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in dev))
    (out / 'development_incumbent.py').write_text(h3)
    manifest = {'registered_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'train_examples': len(rows), 'distinct_inputs': 8, 'alternatives_per_input': 8,
        'train_sha256': hashlib.sha256((out / 'train.jsonl').read_bytes()).hexdigest(),
        'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'initial_checkpoint': str(prior / 'editor_online/checkpoint'),
        'initial_weight_digest': json.loads((prior / 'editor_online/checkpoint/training_state.json').read_text())['weight_digest'],
        'epochs': 2, 'learning_rate': 1e-5, 'planned_updates': 32,
        'optional_extension': 'Exactly two more epochs only if code readiness fails, not based on task reward or monitor.',
        'readiness_temperature': 1.0, 'readiness_requests': 8,
        'readiness_required': {'executable_hook_preserving_changed_guidance': 7, 'distinct_system_prompts': 2},
        'benchmark_content': False, 'development_scope': 'new synthetic feedback; seen code composition, not unseen-task generalization'}
    (out / 'PROTOCOL.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps(manifest))


if __name__ == '__main__': main()

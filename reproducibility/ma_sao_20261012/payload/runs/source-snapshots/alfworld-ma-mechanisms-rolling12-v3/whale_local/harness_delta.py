"""Compact edit actions for the existing ReTool harness, without GPU imports."""
import ast
import json

from harness_code_contract import validate_source

SYSTEM = '''You improve a reusable math-agent harness for the CURRENT solver checkpoint. Read the observed failure packet and current harness settings. Output only a JSON object containing the settings you want to replace. Allowed keys: SYSTEM_PROMPT (guidance before solving), partial_response_hint (message when a tool-free response has no boxed answer), tool_error_hint (message after a Python error). Values must be short English instructions, at most 900 characters each. You may change one or several keys. Use {} only if no improvement is warranted. Preserve useful existing guidance. Do not output Python, explanations, task answers, specific numerical results, or instructions to change tools, grading, or budgets. Choose a substantive, reusable correction appropriate to the CURRENT failures. Previously helpful edits may be obsolete after solver training.'''


def settings(source):
    tree = ast.parse(source)
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef))
    result = {'SYSTEM_PROMPT': None}
    for n in cls.body:
        if isinstance(n, ast.Assign) and n.targets[0].id == 'SYSTEM_PROMPT':
            result['SYSTEM_PROMPT'] = ast.literal_eval(n.value)
    hints = [n for n in ast.walk(cls) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == 'UserMessage']
    # Source order: partial-response branch, then inherited-tool error branch.
    hints.sort(key=lambda n: n.lineno)
    for key, node in zip(['partial_response_hint', 'tool_error_hint'], hints):
        result[key] = ast.literal_eval(next(k.value for k in node.keywords if k.arg == 'content'))
    return result


def messages(source, feedback):
    return [{'role': 'system', 'content': SYSTEM}, {'role': 'user', 'content':
        'Current settings (other executable hooks are preserved automatically):\n' + json.dumps(settings(source)) +
        '\nCurrent-solver failure packet, with reference answers hidden:\n' + json.dumps(feedback, ensure_ascii=False) +
        '\nReturn a compact JSON edit. Improve reasoning or recovery, not the spelling of an existing instruction.'}]


def compile_edit(content, source):
    content = content.strip()
    if content.startswith('```json') and content.endswith('```'):
        content = content[7:-3].strip()
    patch = json.loads(content)
    assert isinstance(patch, dict)
    assert set(patch) <= {'SYSTEM_PROMPT', 'partial_response_hint', 'tool_error_hint'}
    assert all(isinstance(v, str) and 0 < len(v) <= 900 for v in patch.values())
    current = settings(source)
    if not any(current.get(k) != v for k, v in patch.items()):
        return source, patch, False
    tree = ast.parse(source); cls = next(n for n in tree.body if isinstance(n, ast.ClassDef))
    if 'SYSTEM_PROMPT' in patch:
        cls.body = [n for n in cls.body if not (isinstance(n, ast.Assign) and n.targets[0].id == 'SYSTEM_PROMPT')]
        cls.body.insert(0, ast.Assign(targets=[ast.Name(id='SYSTEM_PROMPT', ctx=ast.Store())], value=ast.Constant(patch['SYSTEM_PROMPT'])))
    hints = sorted([n for n in ast.walk(cls) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == 'UserMessage'], key=lambda n: n.lineno)
    for key in ['partial_response_hint', 'tool_error_hint']:
        if key in patch:
            index = ['partial_response_hint', 'tool_error_hint'].index(key)
            assert len(hints) > index, 'Requested recovery hook is absent'
            next(k for k in hints[index].keywords if k.arg == 'content').value = ast.Constant(patch[key])
    return validate_source(ast.unparse(ast.fix_missing_locations(tree))), patch, True


def anchored_advantages(calls, scale=0.125):
    """Paired gain over incumbent: no group centering or invalid-format bonus."""
    return [max(-2.0, min(2.0, r['reward_delta'] / scale)) if r['outcome_kind'] == 'valid' else 0.0 for r in calls]


def gain_rank_advantages(calls):
    """Rank measured valid edits; do not learn from an invalid-format gap."""
    gains = [r['reward_delta'] for r in calls if r['outcome_kind'] == 'valid']
    if not gains: return [0.] * len(calls)
    mean = sum(gains) / len(gains)
    std = (sum((r - mean) ** 2 for r in gains) / len(gains)) ** .5
    baseline = max(0., mean) if std > 1e-8 else 0.
    scale = max(std, 1 / 12)
    return [max(-2., min(2., (r['reward_delta'] - baseline) / scale))
        if r['outcome_kind'] == 'valid' else 0. for r in calls]

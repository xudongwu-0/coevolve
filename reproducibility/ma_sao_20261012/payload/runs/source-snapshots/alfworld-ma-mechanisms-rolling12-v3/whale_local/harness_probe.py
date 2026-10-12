"""Run structural and actual ReTool hook checks in a CPU-only process."""
import argparse
import asyncio
import json
from pathlib import Path
from unittest.mock import patch
from types import SimpleNamespace

from environments.retool.base_harness import ReToolEnv
from verifiers.types import AssistantMessage, UserMessage, ToolMessage
from harness_code_contract import validate_source
from math_env import load_environment

async def probe(source, path, data):
    path.write_text(validate_source(source))
    env = load_environment(harness_path=path, eval_dataset_path=data, num_eval_examples=1)
    assert env._assistant_token_budget is None
    env.set_assistant_token_budget(2048)
    assert env._get_assistant_token_budget() == 2048
    out = {'loadable': True}
    initial = [UserMessage(role='user', content='A synthetic API probe.')]
    initial_state = {'trajectory': [], 'prompt': initial}
    assert not await env.no_tools_called(initial_state)
    first = await env.get_prompt_messages(initial_state)
    assert first and any(getattr(m, 'content', '') == initial[0].content for m in first)
    tool_state = {'trajectory': [{'completion': [SimpleNamespace(content=r'Pending \boxed{7}', tool_calls=[object()])]}]}
    assert not await env.no_tools_called(tool_state), 'Must execute pending tools before finalizing'
    for name, text in [('partial', 'I have computed an intermediate result.'), ('boxed', r'The result is \boxed{7}.')]:
        state = {'trajectory': [{'prompt': [UserMessage(role='user', content='A synthetic API probe.')],
                 'completion': [AssistantMessage(role='assistant', content=text)]}]}
        out[name + '_stops'] = bool(await env.no_tools_called(state))
        msg = await env.get_prompt_messages(state)
        assert isinstance(msg, list) and len(msg) >= 2
    async def tool_messages(self, state):
        return [ToolMessage(role='tool', tool_call_id='synthetic', content='Traceback: NameError: undefined variable')]
    with patch.object(ReToolEnv, 'get_prompt_messages', tool_messages):
        msg = await env.get_prompt_messages({'trajectory': []})
    assert len(msg) >= 1 and msg[0].role == 'tool'
    out['tool_error_nudge'] = any(m.role == 'user' for m in msg[1:])
    out['preserves_final_stop'] = out['boxed_stops']
    out['recovers_partial_response'] = not out['partial_stops'] and out['boxed_stops']
    return out

async def main():
    p = argparse.ArgumentParser(); p.add_argument('--input', type=Path, required=True)
    p.add_argument('--data', type=Path, required=True); p.add_argument('--output', type=Path, required=True)
    args = p.parse_args(); results = []
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for row in map(json.loads, args.input.read_text().splitlines()):
        result = {'id': row['id']}
        try:
            source = row.get('source', row.get('completion', row.get('content')))
            result.update(await asyncio.wait_for(probe(source, args.output.parent / ('probe_' + row['id'] + '.py'), args.data), 20))
            result['passed'] = result['preserves_final_stop']
        except Exception as exc:
            result.update(passed=False, error=repr(exc))
        results.append(result)
    value = {'count': len(results), 'passed': sum(x['passed'] for x in results), 'results': results}
    args.output.write_text(json.dumps(value, indent=2) + '\n'); print(json.dumps(value))

if __name__ == '__main__': asyncio.run(main())

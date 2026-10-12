"""Small, explicitly synthetic ReTool editor curriculum; no benchmark answers.

Eight mechanism families, six training wordings, two held-out wordings each.
These are interface demonstrations, NOT 48 measured reward improvements.
"""
import argparse
import ast
import hashlib
import json
from pathlib import Path

SYSTEM = r'''You edit reusable Python harnesses for a math solver. Return only a complete Python module, without markdown or explanations. Define class CandidateEnv(ReToolEnv). Change one general mechanism suggested by development failures. Preserve other behavior. Never solve or specialize to a particular question. Inherit tools and token-budget enforcement. Do not change grading, read files, access the network, or inspect answers.
API: from environments.retool.base_harness import ReToolEnv; import verifiers as vf; from verifiers.types import UserMessage; from verifiers.utils.message_utils import concat_messages.
ReToolEnv attributes: SYSTEM_PROMPT, USER_PROMPT_TEMPLATE (must preserve {question}), MAX_TURNS (at most 3). Methods: async get_prompt_messages(self, state), @vf.stop async no_tools_called(self, state). state['trajectory'] is a list of {'prompt': messages, 'completion': messages}. Messages have role, content, tool_calls. self._message_content_text(message) returns text. super().get_prompt_messages(state) preserves tool responses. A tool-free partial response normally stops immediately. The strict final-answer format is \\boxed{...}. Never override token-budget stops, tools, or reward functions.'''

START = 'from environments.retool.base_harness import ReToolEnv\n\nclass CandidateEnv(ReToolEnv):\n    MAX_TURNS = 2\n'

def messages(source, feedback):
    return [{'role': 'system', 'content': SYSTEM}, {'role': 'user', 'content':
        'Current harness:\n' + source + '\nDevelopment feedback (no reference answers):\n' + feedback +
        '\nWrite one general improvement. Output the full Python module only.'}]

FINALIZE = r'''from environments.retool.base_harness import ReToolEnv
import verifiers as vf
from verifiers.types import UserMessage
from verifiers.utils.message_utils import concat_messages

class CandidateEnv(ReToolEnv):
    MAX_TURNS = 3

    @vf.stop
    async def no_tools_called(self, state):
        if not state["trajectory"]:
            return False
        last = state["trajectory"][-1]["completion"][-1]
        if getattr(last, "tool_calls", None):
            return False
        return "\\boxed{" in self._message_content_text(last)

    async def get_prompt_messages(self, state):
        if state["trajectory"]:
            previous = state["trajectory"][-1]
            last = previous["completion"][-1]
            if not getattr(last, "tool_calls", None) and "\\boxed{" not in self._message_content_text(last):
                return concat_messages([previous["prompt"], previous["completion"],
                    [UserMessage(role="user", content=NUDGE)]])
        return await super().get_prompt_messages(state)
'''
REPAIR = '''from environments.retool.base_harness import ReToolEnv
from verifiers.types import UserMessage
from verifiers.utils.message_utils import concat_messages

class CandidateEnv(ReToolEnv):
    MAX_TURNS = 3

    async def get_prompt_messages(self, state):
        messages = await super().get_prompt_messages(state)
        outputs = []
        for message in reversed(messages):
            if getattr(message, "role", None) != "tool":
                break
            outputs.append(self._message_content_text(message))
        if any("Traceback" in text or "[tool error]" in text or "[parse error]" in text for text in outputs):
            return concat_messages([messages, [UserMessage(role="user", content=NUDGE)]])
        return messages
'''

FAMILIES = {
 'finalize': (['Partial reasoning ended without a boxed answer.', 'A non-tool response stopped before the requested final quantity.', 'The solver used its first turn on a derivation and then stopped without final formatting.', 'An unfinished derivation is treated as complete just because there was no tool call.', 'Failed traces contain partial reasoning, no tool call, and no final boxed expression.', 'The solver stops after listing intermediate cases without stating the result.', 'The final assistant message is incomplete; unused budget remains but termination fires.', 'Useful progress is lost when the first assistant response has no final answer.'],
    ['Finish only the remaining steps and give the requested result in \\boxed{}. Do not restart the derivation.', 'Complete the calculation briefly, then state the final answer in \\boxed{}. Reuse the progress above.', 'The response lacks a final answer. Continue the unfinished work and finish with \\boxed{}.']),
 'repair': (['Interpreter errors are followed by confident unsupported answers.', 'The solver does not repair an undefined variable after Python fails.', 'A traceback is mistaken for successful verification.', 'Malformed tool arguments are not corrected on the next turn.', 'Tool failures are ignored instead of prompting one concise repair.', 'A failed calculation leads to guessing instead of fixing the code.', 'Execution failed and the model needs actionable recovery feedback.', 'The next response should repair the displayed error before drawing a conclusion.'],
    ['The interpreter failed. Make one concise corrected code_interpreter call addressing the error, then finish with \\boxed{}.', 'This tool response did not verify an answer. Correct the displayed error in a short tool call and then give the final boxed result.', 'Repair the failed calculation once, using the error message. After the repair, finish the requested answer in \\boxed{}.']),
 'concise': (['Long repetitive reasoning exhausts the token budget.', 'The solver restarts the same derivation several times.', 'Verbose explanations crowd out the final answer.', 'Many equivalent examples are listed after the pattern is clear.', 'Redundant derivations consume the remaining generation budget.', 'Repeated self-commentary delays finalization.', 'The response runs out of tokens while restating previous steps.', 'The model spends too much time rephrasing an already established argument.'],
    ['Solve with a short plan and only necessary calculations. Avoid repeating derivations. Reserve space for the final answer in \\boxed{}.', 'Keep reasoning focused on the requested quantity. Do not restart a valid derivation. Finish concisely with \\boxed{}.', 'Use compact reasoning, preserve intermediate progress, and avoid redundant examples. End with the requested value in \\boxed{}.']),
 'verify': (['Arithmetic errors survive an otherwise correct derivation.', 'The solver guesses a numerical total without checking.', 'Hand calculations are inconsistent with stated constraints.', 'An intermediate arithmetic mistake changes the final result.', 'The reasoning method is plausible but its computation is unchecked.', 'The solver confidently reports a calculation with no verification.', 'Several failed solutions could be caught by a short arithmetic check.', 'Computed values are accepted without testing them against the derived relation.'],
    ['Derive the method briefly. Use code_interpreter for useful arithmetic verification, then give the checked requested value in \\boxed{}.', 'Check error-prone calculations with a concise Python computation before reporting the final answer in \\boxed{}.', 'Verify the decisive calculation using code_interpreter when useful. Connect the tool result to the requested quantity and finish with \\boxed{}.']),
 'exact': (['Floating-point rounding changes a discrete mathematical result.', 'Approximate division hides a divisibility condition.', 'A numerical approximation is presented as an exact answer.', 'Floating-point equality is used in an integer computation.', 'Premature rounding corrupts the last step.', 'An exact arithmetic problem is handled with lossy decimals.', 'An integer constraint is checked with an unreliable decimal comparison.', 'Precision loss causes an otherwise sensible calculation to fail.'],
    ['Prefer exact integer or rational arithmetic when the problem permits it. Delay approximations and give the final answer in \\boxed{}.', 'Preserve exact expressions through the calculation. Avoid floating-point equality for divisibility or integer conditions. End with \\boxed{}.', 'Use exact arithmetic for discrete conditions and round only when explicitly required. State the requested final result in \\boxed{}.']),
 'bounds': (['A brute-force search uses arbitrary bounds and misses cases.', 'The solver searches an unexplained finite range.', 'Boundary values are silently excluded from an enumeration.', 'Enumeration starts at a guessed cutoff rather than a derived bound.', 'The search interval does not follow from the constraints.', 'An exhaustive-search claim has no justification for its endpoints.', 'A loop omits a feasible endpoint without checking it.', 'A search may overlook solutions outside the assumed interval.'],
    ['Derive finite search bounds from the problem constraints before enumeration. Check endpoint inclusivity, then finish with \\boxed{}.', 'If using a search, justify its domain and include every permitted boundary case. Report the requested answer in \\boxed{}.', 'Use constraint-derived search limits rather than guessed cutoffs. Verify both endpoints and end with the result in \\boxed{}.']),
 'print': (['The tool program assigns a result but prints nothing.', 'A Python calculation has no visible output for the solver.', 'The final statement is an assignment, so tool feedback is unusable.', 'A loop computes the quantity but the result is never displayed.', 'Successful execution produces empty stdout.', 'The tool returns no output even though a value was computed.', 'The model cannot use the calculation because its output is blank.', 'The tool program ends without printing the value needed for the answer.'],
    ['When using code_interpreter, explicitly print the requested computed quantity so its output can be used. Finish with \\boxed{}.', 'Keep tool programs self-contained and end them with an explicit print of the decisive result. Give the final answer in \\boxed{}.', 'Ensure each Python calculation displays its useful result with print. Use that output to answer the question in \\boxed{}.']),
 'quantity': (['The solver reports an intermediate value rather than the requested quantity.', 'A correct computation is mapped to the wrong final expression.', 'The final response forgets the last transformation requested by the problem.', 'The reported result does not correspond to what was asked.', 'Intermediate variables are confused at finalization.', 'The calculation is correct but the wrong value is boxed.', 'The last step answers a related question instead of the actual one.', 'The final output is a raw tool result without the required interpretation.'],
    ['Before finalizing, map intermediate results back to the exact quantity requested. Perform any remaining transformation and end with \\boxed{}.', 'Track what the question asks for. Distinguish intermediate values from the final requested result and box only that result.', 'After computation, verify that the reported quantity matches the original request. Finish with the intended value in \\boxed{}.']),
}

def build(out):
    out.mkdir(parents=True, exist_ok=False)
    train, validation = [], []
    for family, (feedbacks, instructions) in FAMILIES.items():
        for index, feedback in enumerate(feedbacks):
            instruction = instructions[index % len(instructions)]
            if family in ('finalize', 'repair'):
                source = (FINALIZE if family == 'finalize' else REPAIR).replace('NUDGE', repr(instruction))
            else:
                source = START + '    SYSTEM_PROMPT = ' + repr(instruction) + '\n'
            ast.parse(source)
            row = {'id': f'{family}-{index}', 'family': family, 'messages': messages(START, feedback),
                   'completion': source, 'provenance': 'Codex-authored synthetic interface demonstration; not reward-verified',
                   'uses_benchmark_questions_or_answers': False}
            (train if index < 6 else validation).append(row)
    for name, rows in [('train', train), ('validation', validation)]:
        (out / (name + '.jsonl')).write_text(''.join(json.dumps(x) + '\n' for x in rows))
    metadata = {'train_examples': len(train), 'validation_examples': len(validation), 'mechanism_families': len(FAMILIES),
        'validation_scope': 'held-out feedback wordings of seen mechanisms; NOT unseen-family generalization or benchmark gain',
        'teacher': 'active Codex, bootstrap only', 'no_benchmark_content': True,
        'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (out / 'manifest.json').write_text(json.dumps(metadata, indent=2) + '\n')
    print(json.dumps(metadata))

if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--out', type=Path, required=True)
    build(p.parse_args().out)

"""Teach explicit edit requests and preservation of an existing mechanism.

The initial pilot could emit valid prompt rewrites but failed to map general
feedback to recovery code. This bounded curriculum makes the edit request
explicit. At runtime a deterministic telemetry-based search chooses a focus;
the internal model must write all code. This is not open-ended diagnosis.
"""
import argparse
import ast
import hashlib
import json
from pathlib import Path
from harness_bootstrap import FINALIZE, REPAIR, START, FAMILIES, messages
from harness_code_contract import validate_source

FOCUS = {
    'strategy': 'Improve the solver guidance by adding or replacing SYSTEM_PROMPT. Choose a general reasoning or tool-use strategy from the failure observations. Preserve all existing Python hook methods and MAX_TURNS. Return the complete module with the new guidance.',
    'finalize': 'Implement recovery for incomplete non-tool responses: override no_tools_called and get_prompt_messages. Continue a partial response, stop on a boxed final answer, retain tool calls and the inherited total token budget.',
    'repair': 'Implement tool-error recovery in get_prompt_messages. Preserve tool output and append one actionable user nudge after an interpreter error. Keep any existing incomplete-response recovery logic.',
    'concise': 'Change SYSTEM_PROMPT to reduce repeated derivations and reserve room for the final answer. Preserve existing Python hook methods.',
    'verify': 'Change SYSTEM_PROMPT to check error-prone arithmetic using the Python tool. Preserve existing Python hook methods.',
    'exact': 'Change SYSTEM_PROMPT to prefer exact arithmetic over premature rounding. Preserve existing Python hook methods.',
    'bounds': 'Change SYSTEM_PROMPT to derive enumeration bounds from constraints and check endpoints. Preserve existing Python hook methods.',
    'print': 'Change SYSTEM_PROMPT to require useful explicit printed output from Python computations. Preserve existing Python hook methods.',
    'quantity': 'Change SYSTEM_PROMPT to distinguish the requested quantity from intermediate values. Preserve existing Python hook methods.',
}

def request(source, feedback, focus):
    result=messages(source, feedback)
    result[-1]['content']+='\nImplementation focus ('+focus+'): '+FOCUS[focus]
    return result

def choose_focus(trajectories, source, slot):
    failed=[r for r in trajectories if not r['reward']]
    missing=sum('\\boxed{' not in r['last_assistant'] for r in failed)
    errors=sum(any('Traceback' in str(t.get('feedback','')) or '[tool error]' in str(t.get('feedback',''))
        or '[parse error]' in str(t.get('feedback','')) for t in r.get('harness_trace',[])) for r in failed)
    choices=[]
    if missing and 'async def no_tools_called' not in source:choices.append('finalize')
    if errors and 'Traceback' not in source:choices.append('repair')
    choices.extend(['verify','concise','quantity'])
    return choices[slot % len(choices)]

def combined(final_instruction, repair_instruction):
    final=FINALIZE.replace('NUDGE',repr(final_instruction))
    repair=REPAIR.replace('NUDGE',repr(repair_instruction))
    tail=repair.split('        messages = await super().get_prompt_messages(state)',1)[1]
    return final.replace('        return await super().get_prompt_messages(state)',
        '        messages = await super().get_prompt_messages(state)'+tail)

def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);args=p.parse_args()
    out=args.out.resolve();out.mkdir(parents=True)
    train=[];dev=[]
    for family,(feedbacks,instructions) in FAMILIES.items():
        for i,feedback in enumerate(feedbacks):
            instruction=instructions[i%3]
            source=START
            if family=='finalize':target=FINALIZE.replace('NUDGE',repr(instruction))
            elif family=='repair':target=REPAIR.replace('NUDGE',repr(instruction))
            else:target=START+'    SYSTEM_PROMPT = '+repr(instruction)+'\n'
            row={'id':f'{family}-{i}','family':family,'messages':request(source,feedback,family),
                'completion':validate_source(target),'provenance':'Codex-authored synthetic explicit edit demonstration; not reward-verified'}
            (train if i<6 else dev).append(row)
            if family in ('finalize','repair'):
                fi=instruction if family=='finalize' else FAMILIES['finalize'][1][i%3]
                ri=instruction if family=='repair' else FAMILIES['repair'][1][i%3]
                source=(REPAIR.replace('NUDGE',repr(ri)) if family=='finalize' else FINALIZE.replace('NUDGE',repr(fi)))
                target=combined(fi,ri)
                row=dict(row,id=f'{family}-preserve-{i}',messages=request(source,feedback,family),completion=validate_source(target))
                (train if i<6 else dev).append(row)
    for name,data in [('train',train),('development',dev)]:
        (out/(name+'.jsonl')).write_text(''.join(json.dumps(x)+'\n' for x in data))
    (out/'manifest.json').write_text(json.dumps({'train_examples':len(train),'development_examples':len(dev),
        'mechanism_families':8,'benchmark_content':False,'explicit_implementation_focus':True,
        'scope':'bounded edit-conditioned code learner; deterministic runtime focus from telemetry, not autonomous diagnosis',
        'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},indent=2)+'\n')
    print('train',len(train),'development',len(dev))

if __name__=='__main__':main()

"""Summarize only the current harness-development feedback for the active proposer."""
import argparse
from pathlib import Path
import json

p=argparse.ArgumentParser();p.add_argument('run',type=Path);args=p.parse_args()
run=args.run.resolve();status=json.loads((run/'STATUS.json').read_text())
assert status['stage']=='awaiting_active_agent'
job=run/'proposer_jobs'/status['job_id'];request=json.loads((job/'request.json').read_text())
root=Path(request['run_dir']);accepted=(root/'logs/accepted_harness.txt').read_text().strip()
files=list((root/'logs/mh_val'/accepted).glob('*/trajectories.jsonl'));assert len(files)==1
rows=[json.loads(s) for s in files[0].read_text().splitlines()]
assert not any(any(code in t.get('env_feedback','') for code in ['502 Bad Gateway','503 Service Unavailable','All connection attempts failed']) for r in rows for t in r['harness_trace'])
def condensed(r):
    return {'task_id':r['task_id'],'reward':r['reward'],'question':r['policy_calls'][0]['messages'][-1]['content'],
        'assistant_messages':[c['parsed_message'] for c in r['policy_calls']],
        'harness_trace':r['harness_trace'],
        'full_tool_messages':[m for c in r['policy_calls'] for m in c['messages'] if m['role']=='tool']}
result={'job_id':status['job_id'],'request':request,'accepted':accepted,
    'summary':{'count':len(rows),'solved':sum(r['reward'] for r in rows),
        'missing_answer_tag':sum('<answer>' not in r['last_assistant'] for r in rows),
        'multi_tool_turns':sum(sum(bool(c['parsed_message'].get('tool_calls')) for c in r['policy_calls'])>1 for r in rows)},
    'failures':[condensed(r) for r in rows if not r['reward']][:7],
    'successes':[condensed(r) for r in rows if r['reward']][:2]}
(job/'development_context.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(result,ensure_ascii=False,indent=2))

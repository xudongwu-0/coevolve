"""One prospective paper-LR check using the same on-policy W1/H1 training receipts."""
import asyncio
import json
import os
from pathlib import Path
import shutil
import time
from pipeline import ROOT,sha,write_json
from search_baseline_v2 import read,train
from search_runtime_v2 import service,evaluate

reference=ROOT/'runs/whale_search_transportfixed_baseline_20260917'
run=ROOT/'runs/whale_search_lr1e7_probe_20260917';run.mkdir(exist_ok=False)
config=read(reference/'config.json');config['learning_rate']=1e-7
model=reference/'cycle_01/solver_update/round_01/checkpoint'
ident=read(reference/'cycle_01/solver_update/complete.json')['identity']
harness=Path(read(reference/'cycle_01/mh_result.json')['harness'])
source=reference/'cycle_02/train_rollouts/trajectories.jsonl'
(run/'train_rollouts').mkdir();shutil.copy2(source,run/'train_rollouts/trajectories.jsonl')
write_json(run/'config.json',config)
write_json(run/'PROTOCOL.json',{'reason':'W2 at LR1e-6 regressed from W1 on the preexisting 48-question harness-development set, 20/48 to15/48. Test and monitor are still unopened at registration.',
    'single_new_setting':1e-7,'reference_paper':'https://arxiv.org/html/2609.00196v1',
    'seed':42,'same_weight_start':str(model),'same_weight_digest':ident['weight_digest'],
    'same_harness':str(harness),'same_harness_sha256':sha(harness),
    'same_training_rollouts':str(source),'same_training_rollouts_sha256':sha(source),
    'new_training_samples':0,'reused_on_policy_training_samples':512,
    'comparison_split':'mh_val only; no monitor or test is used for the learning-rate check',
    'selection_rule':'Prefer paper LR1e-7 only if it improves on LR1e-6 on the same 48 dev questions; record whether it also exceeds W1.',
    'cost_policy':'Count the extra optimization and evaluation as scenario-calibration overhead, not as a free baseline improvement.'})
write_json(run/'STATUS.json',{'stage':'training'})
started=time.monotonic()
try:
    checkpoint,new_ident=train(run,run,config,model,ident,harness)
    write_json(run/'STATUS.json',{'stage':'evaluating_mh_val'})
    with service(run/'evaluation_worker',checkpoint,'6',21220,config['data'],parameter_dtype='float32') as (url,loaded):
        assert loaded['weight_digest']==new_ident['weight_digest']
        value=asyncio.run(evaluate(base_url=url,identity=loaded,harness=harness,
            data_entry=config['data']['splits']['mh_val'],output=run/'mh_val',temperature=1.0,seed=42))
    write_json(run/'RESULT.json',{'new_checkpoint':str(checkpoint),'identity':new_ident,
        'mh_val':value,'W1_same_harness_solved':20,'high_lr_W2_same_harness_solved':15,
        'better_than_high_lr':value['solved_examples']>15,'better_than_incoming_W1':value['solved_examples']>20,
        'elapsed_seconds':time.monotonic()-started,'test_opened':False,'monitor_opened':False})
    write_json(run/'STATUS.json',{'stage':'complete'})
except BaseException as exc:
    write_json(run/'STATUS.json',{'stage':'failed','error':repr(exc)});raise

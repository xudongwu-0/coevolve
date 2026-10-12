"""One predeclared LR diagnostic: same cached native RSFT targets, MH only."""
import argparse,json,shutil,time
from pathlib import Path
import pipeline
from math_runtime import service,BASE_HARNESS
from math_sampling import sample_parallel


def read(p):return json.loads(Path(p).read_text())

def main():
    p=argparse.ArgumentParser();p.add_argument('--run-dir',type=Path,required=True);p.add_argument('--grid',type=Path,required=True);p.add_argument('--trial',required=True)
    a=p.parse_args();run=a.run_dir.resolve();grid=read(a.grid);trial=grid['trials'][a.trial];source=Path(grid['source_run'])
    run.mkdir(parents=True,exist_ok=True);phase=run/'round_01';phase.mkdir(exist_ok=True)
    if not (phase/'rollouts').exists():shutil.copytree(source/'round_01/rollouts',phase/'rollouts')
    rows=[json.loads(line) for line in (phase/'rollouts/trajectories.jsonl').read_text().splitlines()]
    (run/'calls').mkdir(exist_ok=True)
    receipt_hashes={}
    for row in rows:
        for call in row['policy_calls']:
            name=call['call_id']+'.json';original=source/'calls'/name;target=run/'calls'/name
            if not target.exists():shutil.copy2(original,target)
            assert pipeline.sha(original)==pipeline.sha(target)
            receipt_hashes[name]=pipeline.sha(original)
        if row.get('context_rejection_id'):
            name=row['context_rejection_id']+'.json';(run/'context_rejections').mkdir(exist_ok=True)
            shutil.copy2(source/'context_rejections'/name,run/'context_rejections'/name)
    pipeline.write_json(run/'CACHED_TRAIN_RECEIPTS.json',{'source':str(source),'receipts_sha256':receipt_hashes,
        'only_training_receipts_copied':True,'new_training_generations':0})
    data=read(source/'config.json')['data'];base=Path(grid['base_model'])
    pipeline.write_json(run/'config.json',{'data':data,'learning_rate':trial['learning_rate'],'trial':a.trial,'source_run':str(source),'source_rollouts_sha256':pipeline.sha(source/'round_01/rollouts/trajectories.jsonl')})
    pipeline.write_json(run/'eval_hardware.json',{'sampler_gpus':trial['replica_gpus'],'sampler_port_base':trial['replica_port_base']})
    started=time.monotonic()
    with service(run,base,trial['gpu'],trial['port'],data,trial['learning_rate'],'float32') as (url,initial):
        assert initial['weight_digest']==grid['base_digest'];pipeline.write_json(run/'initial_identity.json',initial)
        checkpoint=phase/'checkpoint'
        if not (checkpoint/'training_state.json').exists():
            pipeline.api(url,'/rsft/train',{'round':1,'rollouts':str(phase/'rollouts/trajectories.jsonl'),'train_data':data['splits']['train']['path'],'harness':str(BASE_HARNESS),'expected_weight_digest':initial['weight_digest']},timeout=7200)
        current=pipeline.api(url,'/checkpoint/load',{'checkpoint':str(checkpoint)},timeout=300)
        pipeline.write_json(run/'trained_identity.json',current)
        eval_phase=run/'development_sampling';eval_phase.mkdir(exist_ok=True)
        output=run/'mh_val'
        if not (output/'val.json').exists():
            sample_parallel(run=run,phase=eval_phase,output=output,base_url=url,identity=current,harness=BASE_HARNESS,data_entry=data['splits']['mh_val'],rollouts=1,seed=42,temperature=0,split='mh_val',port_base=trial['replica_port_base'],resume=True)
        assert pipeline.api(url,'/health')['weight_digest']==current['weight_digest']
    pipeline.write_json(run/'complete.json',{'scope':'LR selection on development only; not a full WHALE run or gate qualification','trial':a.trial,'learning_rate':trial['learning_rate'],'development_solved':read(run/'mh_val/val.json')['solved_examples'],'development_total':48,'training_trajectories_reused_without_generation':512,'new_training_generation_calls':0,'identity':current,'elapsed_seconds':time.monotonic()-started})

if __name__=='__main__':main()

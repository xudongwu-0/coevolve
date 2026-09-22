"""Local Harness-R1-style outcome GRPO over exact generated code tokens.

Uses the WHALE actor's teacher-forced log-probability implementation. Same-batch
environment deltas provide rewards; no judge, format bonus, or teacher targets.
"""
import argparse
import asyncio
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch

import worker as w
from harness_code_contract import validate_source
from pipeline import sha
from qwen35_padding import install as install_padding
import safe_gradient_norm


def read_rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines()]


def advantages(rewards,groups):
    result=np.zeros(len(rewards),dtype=np.float64);summary=[]
    for group in dict.fromkeys(groups):
        indices=[i for i,g in enumerate(groups) if g==group]
        values=np.asarray([rewards[i] for i in indices],dtype=np.float64)
        mean=float(values.mean());std=float(values.std(ddof=0))
        if std>1e-8:result[indices]=(values-mean)/std
        summary.append({'group':group,'indices':indices,'rewards':values.tolist(),
            'mean':mean,'population_std':std,'zero_variance':std<=1e-8})
    return result,summary


def clipped_loss(logp,old,rollout,advantage):
    ratio=torch.exp(logp-old)
    correction=torch.exp(old-rollout).clamp(max=2).detach()
    raw=ratio*advantage
    clipped=ratio.clamp(0.8,1.28)*advantage
    return -(correction*torch.minimum(raw,clipped)).mean(),ratio,correction


def generate(args):
    requests=read_rows(args.data);path=w.RUN/'generations.jsonl'
    retained=read_rows(path) if path.exists() else []
    assert [r['id'] for r in retained]==[r['id'] for r in requests[:len(retained)]]
    with path.open('a') as stream:
        for index,row in enumerate(requests):
            if index<len(retained):continue
            seed=row.get('seed',args.seed+index)
            torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
            started=time.monotonic()
            content,usage,record=w.generate(row['messages'],max_tokens=args.max_tokens,
                temperature=args.temperature,top_p=1.0,top_k=0,stop_at_chat_end=True,return_token_logprobs=True)
            result={**record,'id':row['id'],'group':row['group'],'usage':usage,'seed':seed,
                'sampling_identity':w.identity(),'temperature':args.temperature,'top_p':1.0,'top_k':0,
                'requested_max_tokens':args.max_tokens,'elapsed_seconds':time.monotonic()-started}
            assert all(np.isfinite(record['response_logprobs']))
            try:
                result['source']=validate_source(content);result['valid_source']=True
                (w.RUN/(row['id']+'.py')).write_text(result['source'])
            except Exception as exc:result.update(valid_source=False,validation_error=repr(exc))
            stream.write(json.dumps(result)+'\n');stream.flush()
            w.atomic_json(w.RUN/'progress.json',{'generated':index+1,'planned':len(requests)})
            print(json.dumps({'id':row['id'],'valid_source':result['valid_source'],'tokens':usage['completion_tokens']}),flush=True)
    w.atomic_json(w.RUN/'complete.json',{'identity':w.identity(),'generations_sha256':sha(path),
        'requests':len(requests),'data_sha256':sha(args.data)})


def forward(call):
    batch=w.make_batch([call]).to('cuda')
    inputs={**batch.batch,'pad_token_id':w.state['tokenizer'].pad_token_id}
    return w.state['actor']._forward_micro_batch(inputs,temperature=call['temperature'],
        calculate_entropy=False)['log_probs'][0,:len(call['response_ids'])]


def train(args):
    started=time.monotonic();calls=read_rows(args.data)
    adv,groups=advantages([r['reward_delta'] for r in calls],[r['group'] for r in calls])
    w.atomic_json(w.RUN/'advantages.json',{'data_sha256':sha(args.data),'advantages':adv.tolist(),'groups':groups})
    assert all(len(g['indices'])==8 for g in groups),'Harness-R1 training group size is eight'
    if not np.any(adv):
        w.atomic_json(w.RUN/'complete.json',{'updated':False,'reason':'all reward groups have zero variance',
            'identity':w.identity(),'data_sha256':sha(args.data)})
        return
    initial=w.identity();before=w.parameter_hashes(w.state['model']);inherited=w.optimizer_steps(w.state['optimizer'])
    assert all(r['sampling_identity']['weight_digest']==initial['weight_digest'] for r in calls)
    prior=w.state['training_state'] or {}
    reset=prior.get('method')!='Harness-R1-style outcome GRPO'
    if reset:
        # SFT Adam moments have different betas/decay. Start a named RL optimizer.
        optimizer=torch.optim.AdamW([p for p in w.state['model'].parameters() if p.requires_grad],
            lr=args.learning_rate,betas=(0.9,0.98),weight_decay=0.1)
        w.state['optimizer']=optimizer;w.state['actor'].actor_optimizer=optimizer
    else:optimizer=w.state['optimizer']
    for group in optimizer.param_groups:group['lr']=args.learning_rate
    steps_before=w.optimizer_steps(optimizer)
    fix=safe_gradient_norm.install(lambda event:w.atomic_json(w.RUN/'gradient_norm_events.json',safe_gradient_norm.EVENTS))
    w.state['model'].eval();old=[]
    with torch.no_grad():
        for call in calls:
            lp=forward(call).detach().float().cpu()
            assert torch.isfinite(lp).all();old.append(lp)
    torch.save({'old_training_logprobs':old,'rollout_logprobs':[r['response_logprobs'] for r in calls],
        'advantages':torch.tensor(adv),'ids':[r['id'] for r in calls]},w.RUN/'grpo_batch.pt')
    torch.cuda.reset_peak_memory_stats();metrics=[]
    w.state['model'].train()
    for epoch in range(args.epochs):
        optimizer.zero_grad(set_to_none=True);loss_value=0;ratios=[];corrections=[]
        for i,call in enumerate(calls):
            if adv[i]==0:continue
            logp=forward(call)
            sampled=torch.tensor(call['response_logprobs'],device='cuda',dtype=torch.float32)
            loss,ratio,correction=clipped_loss(logp.float(),old[i].to('cuda'),sampled,float(adv[i]))
            assert torch.isfinite(loss)
            (loss/len(calls)).backward()
            loss_value+=float(loss.detach())/len(calls)
            ratios.append(float(ratio.detach().mean()));corrections.append(float(correction.mean()))
        norm=torch.nn.utils.clip_grad_norm_([p for p in w.state['model'].parameters() if p.requires_grad],1.0)
        assert torch.isfinite(norm);optimizer.step()
        metrics.append({'epoch':epoch,'loss':loss_value,'grad_norm':float(norm),
            'mean_ratio':float(np.mean(ratios)),'mean_truncated_importance_weight':float(np.mean(corrections))})
        w.atomic_json(w.RUN/'progress.json',{'completed_updates':epoch+1,'planned_updates':args.epochs,'metrics':metrics})
        print(json.dumps(metrics[-1]),flush=True)
    optimizer.zero_grad(set_to_none=True)
    assert all(torch.isfinite(p).all() for p in w.state['model'].parameters())
    after=w.parameter_hashes(w.state['model']);changed=[k for k in before if before[k]!=after[k]]
    assert changed and w.optimizer_steps(optimizer)==[(max(steps_before) if steps_before else 0)+args.epochs]
    info={'role':'harness_editor','method':'Harness-R1-style outcome GRPO','sampling_identity':initial,
        'data':str(args.data.resolve()),'data_sha256':sha(args.data),'examples':len(calls),'groups':len(groups),
        'epochs':args.epochs,'learning_rate':args.learning_rate,'weight_decay':0.1,'adam_betas':[0.9,0.98],
        'optimizer_reset_for_grpo':reset,'inherited_optimizer_steps':inherited,
        'optimizer_steps_before':steps_before,'optimizer_steps_after':w.optimizer_steps(optimizer),
        'parameter_dtype':'float32','parameter_hashes':after,'weight_digest':w.digest(after),
        'changed_parameter_tensors':changed,'qwen35_padding_compatibility_fix':install_padding(),
        'gradient_norm_precision_fix':fix,'gradient_norm_precision_events':safe_gradient_norm.EVENTS,
        'metrics':metrics,'peak_gpu_reserved_bytes':torch.cuda.max_memory_reserved(),
        'elapsed_seconds':time.monotonic()-started,'response_tokens':sum(len(r['response_ids']) for r in calls),
        'objective':{'clip_low':0.2,'clip_high':0.28,'importance_weight_max':2.0,
            'advantage_std':'population','kl_coefficient':0,'format_bonus':0,
            'sequence_mean_then_batch_mean':True,'same_batch_reward_delta':True}}
    staging=w.RUN/'checkpoint.staging';staging.mkdir()
    w.state['model'].save_pretrained(staging,max_shard_size='5GB');w.state['tokenizer'].save_pretrained(staging)
    torch.save(optimizer.state_dict(),staging/'optimizer.pt')
    torch.save({'cpu':torch.get_rng_state(),'cuda':torch.cuda.get_rng_state_all()},staging/'rng.pt')
    w.atomic_json(staging/'training_state.json',info);staging.rename(w.RUN/'checkpoint')
    identity=w.load_model(w.RUN/'checkpoint');assert identity['weight_digest']==info['weight_digest']
    w.atomic_json(w.RUN/'complete.json',{'updated':True,'reload_verified':True,'identity':identity,
        'training_state_sha256':sha(w.RUN/'checkpoint/training_state.json')})


async def main():
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['generate','train'])
    p.add_argument('--data',type=Path,required=True);p.add_argument('--seed',type=int,default=1337)
    p.add_argument('--temperature',type=float,default=0.7);p.add_argument('--max-tokens',type=int,default=1024)
    p.add_argument('--learning-rate',type=float,default=1e-6);p.add_argument('--epochs',type=int,default=4)
    args=p.parse_args();install_padding()
    async with w.lifespan(None):(generate if args.mode=='generate' else train)(args)


if __name__=='__main__':asyncio.run(main())

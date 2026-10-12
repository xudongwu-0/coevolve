"""Read-only gradients on cached TRAIN minibatches; never step the optimizer."""
import argparse
import asyncio
import json
from pathlib import Path
import types
import torch
import math_worker as mw

w=mw.w

async def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True)
    p.add_argument('--partition',type=int,required=True);p.add_argument('--partitions',type=int,default=4)
    args=p.parse_args();source=args.run.resolve()
    tensors=torch.load(source/'round_01/accepted_batch.pt',map_location='cpu',weights_only=True)
    rows=[json.loads(x) for x in (source/'round_01/rollouts/trajectories.jsonl').read_text().splitlines()]
    accepted=[c for r in rows if r['reward'] for c in r['policy_calls']]
    async with w.lifespan(None):
        initial=w.identity();results=[];current=None
        def inspect_gradient(self):
            pairs=[(n,p.grad) for n,p in self.actor_module.named_parameters() if p.grad is not None]
            norms=torch._foreach_norm([g for _,g in pairs],2)
            norm32=torch.linalg.vector_norm(torch.stack(norms),2)
            record={'minibatch':current,'norm_fp32':float(norm32),'optimizer_updated':False}
            if not torch.isfinite(norm32):
                bad=[];large=[];norms64=[]
                for (name,g),n in zip(pairs,norms):
                    if not torch.isfinite(g).all():bad.append(name)
                    if not torch.isfinite(n) or n>1e10:
                        large.append({'parameter':name,'max_abs':float(g.abs().max()),'norm32':float(n)})
                    norms64.append(torch.linalg.vector_norm(g,2,dtype=torch.float64))
                record.update(nonfinite_gradient_tensors=bad,norm_fp64=float(torch.linalg.vector_norm(torch.stack(norms64),2)),large=large)
            results.append(record)
            print(json.dumps(record),flush=True)
            w.atomic_json(w.RUN/'results.json',results)
            self.actor_optimizer.zero_grad()
            return torch.tensor(0.,device='cuda')
        w.state['actor']._optimizer_step=types.MethodType(inspect_gradient,w.state['actor'])
        for start in range(0,len(accepted),4):
            current=start//4
            if current%args.partitions!=args.partition:continue
            batch=w.DataProto.from_dict(tensors={k:v[start:start+4] for k,v in tensors.items()},
                meta_info={'temperature':1.0,'pad_token_id':w.state['tokenizer'].pad_token_id,
                    'sft_epochs':1,'sft_mini_batch_size':4,'sft_micro_batch_size_per_gpu':1})
            metrics=w.state['actor'].update_sft_policy(batch)
            results[-1]['loss_sum']=metrics['actor/sft_loss']
            results[-1]['call_ids']=[c['call_id'] for c in accepted[start:start+4]]
            w.atomic_json(w.RUN/'results.json',results)
        assert w.optimizer_steps(w.state['optimizer'])==initial['optimizer_steps']
        assert w.digest(w.parameter_hashes(w.state['model']))==initial['weight_digest']
        w.atomic_json(w.RUN/'complete.json',{'unchanged_weights':True,'unchanged_optimizer':True,
            'identity':initial,'minibatches':len(results),'results':results})

if __name__=='__main__':asyncio.run(main())

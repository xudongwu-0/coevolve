"""Call the existing native SearchQA RSFT worker offline, with explicit LR handoff."""
import argparse
import asyncio
from pathlib import Path

import search_worker_v2 as mw
import worker as w


async def main():
    p=argparse.ArgumentParser();p.add_argument('--rollouts',required=True);p.add_argument('--data',required=True)
    p.add_argument('--harness',required=True);p.add_argument('--expected-digest',required=True);args=p.parse_args()
    async with w.lifespan(None):
        assert w.identity()['weight_digest']==args.expected_digest
        # Loading an Adam state restores its saved LR too; apply this run's LR
        # after loading rather than merely changing the process environment.
        for group in w.state['optimizer'].param_groups:group['lr']=w.LEARNING_RATE
        assert all(g['lr']==w.LEARNING_RATE for g in w.state['optimizer'].param_groups)
        result=mw.train(mw.TrainRequest(round=1,rollouts=args.rollouts,train_data=args.data,
            harness=args.harness,expected_weight_digest=args.expected_digest))
        identity=w.load_model(Path(result['checkpoint']))
        assert all(g['lr']==w.LEARNING_RATE for g in w.state['optimizer'].param_groups)
        w.atomic_json(w.RUN/'complete.json',{'reload_verified':True,'identity':identity,'result':result,
            'actual_optimizer_learning_rates':[g['lr'] for g in w.state['optimizer'].param_groups]})


if __name__=='__main__':asyncio.run(main())

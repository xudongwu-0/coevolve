"""Prospectively freeze fresh HotpotQA splits, independent of model outcomes."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
from urllib.request import urlopen
import time

import pandas as pd
from pipeline import ROOT, sha, write_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    sizes = {'train':128, 'mh_val':48, 'monitor':64, 'test':128}
    revision = '1908d6afbbead072334abe2965f91bd2709910ab'
    spec = {'source':'hotpotqa/hotpot_qa', 'config':'distractor', 'source_split':'train',
            'source_revision':revision, 'sizes':sizes,
            'filter':'bridge questions; exclude yes/no answers; no reward or model filtering',
            'selection':'source offsets 10000..10799, then question SHA256 order; exclude all prior SearchQA corpus IDs',
            'retrieval':'BM25 over the ten official distractor context paragraphs; no supporting-fact labels supplied',
            'verifier':'fixed native WHALE normalized exact match',
            'test_policy':'reserved until baseline qualification and replacement protocol are frozen'}
    write_json(out/'specification.json', spec)
    excluded=set(); exclusions=[]
    for path in sorted((ROOT/'runs').glob('whale_search*/corpora.json')):
        if path.parent == out: continue
        excluded.update(json.loads(path.read_text()))
        exclusions.append({'path':str(path), 'sha256':sha(path)})
    def fetch(offset):
        path=out/f'source_{offset:05d}.json'
        if not path.exists():
            url=f'https://datasets-server.huggingface.co/rows?dataset=hotpotqa/hotpot_qa&config=distractor&split=train&offset={offset}&length=100'
            for attempt in range(5):
                try:
                    with urlopen(url,timeout=45) as response:
                        assert response.headers.get('x-revision')==revision
                        content=response.read()
                    path.write_bytes(content); break
                except OSError:
                    if attempt==4: raise
                    time.sleep(2**attempt)
        return path
    with ThreadPoolExecutor(max_workers=4) as pool:
        paths=list(pool.map(fetch,range(10000,10800,100)))
    eligible={}
    for path in paths:
        for entry in json.loads(path.read_text())['rows']:
            raw=entry['row']; key=hashlib.sha256(raw['question'].strip().encode()).hexdigest()
            if key not in excluded and raw['type']=='bridge' and raw['answer'].strip().lower() not in {'yes','no'}:
                eligible.setdefault(key,raw)
    assert len(eligible)>=sum(sizes.values())
    keys=sorted(eligible); corpora={}; splits={}; start=0
    for name,count in sizes.items():
        selected=keys[start:start+count]; start+=count
        rows=[]
        for index,key in enumerate(selected):
            raw=eligible[key]
            rows.append({'data_source':'hotpotqa_distractor_bridge',
                'extra_info':{'question':raw['question'],'index':index,'task_id':key},
                'reward_model':{'ground_truth':{'target':[raw['answer']]}}})
            corpora[key]=[{'title':title,'text':''.join(sentences)}
                for title,sentences in zip(raw['context']['title'],raw['context']['sentences'])]
        path=out/f'{name}.parquet'; pd.DataFrame(rows).to_parquet(path,index=False)
        splits[name]={'path':str(path),'sha256':sha(path),'count':count}
    write_json(out/'corpora.json',corpora)
    manifest=spec|{'splits':splits,'corpora':{'path':str(out/'corpora.json'),'sha256':sha(out/'corpora.json')},
        'prior_corpora':exclusions,'excluded_prior_questions':len(excluded),'eligible_count':len(eligible),
        'source_pages':[{'path':str(p),'sha256':sha(p)} for p in paths], 'question_disjoint':True}
    write_json(out/'manifest.json',manifest)
    print(json.dumps({'manifest':str(out/'manifest.json'),'sizes':sizes,'eligible':len(eligible),'excluded':len(excluded)}),flush=True)


if __name__=='__main__':main()

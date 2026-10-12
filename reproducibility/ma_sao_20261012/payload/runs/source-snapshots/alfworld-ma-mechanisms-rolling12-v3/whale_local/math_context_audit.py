"""Recompute actual context size from immutable request receipts, offline."""
from functools import lru_cache
from pathlib import Path
import json

from transformers import AutoTokenizer
from pipeline import sha


@lru_cache(maxsize=8)
def tokenizer(path):
    return AutoTokenizer.from_pretrained(path,local_files_only=True)


def recheck_context(path):
    path=Path(path);record=json.loads(path.read_text())
    tok=tokenizer(record['sampling_identity']['weights'])
    args=dict(tools=record['tools'],add_generation_prompt=True,enable_thinking=False)
    text=tok.apply_chat_template(record['messages'],tokenize=False,**args)
    ids=tok(text,add_special_tokens=False)['input_ids']
    count=len(ids)
    assert record['generated_tokens']==0 and record['context_limit']==4096
    assert count+record['requested_max_tokens']>4096
    recorded=record['prompt_tokens']
    if recorded!=count:
        legacy=tok.apply_chat_template(record['messages'],tokenize=True,**args)
        assert hasattr(legacy,'keys') and 'input_ids' in legacy
        assert recorded==len(legacy) and legacy['input_ids']==ids
    return {'receipt':str(path.resolve()),'receipt_sha256':sha(path),'recorded_prompt_tokens':recorded,
            'actual_prompt_tokens':count,'requested_max_tokens':record['requested_max_tokens'],
            'legacy_mapping_length_mistake_reproduced':recorded!=count,
            'native_context_rejection_valid':True,'original_receipt_unmodified':True}

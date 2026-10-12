"""Bounded localhost inference API over the retained, read-only W0 model."""
from contextlib import asynccontextmanager
from pathlib import Path
import os
import threading
import time
import uuid

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import torch
from transformers import AutoModelForImageTextToText, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
MODEL_ROOT = Path(os.environ.get("WHALE_MODEL_ROOT", str(ROOT / "local_assets/weights/W0"))).resolve()
NAME = "rsi-w0"
LOCK = threading.Lock()
state = {}


@asynccontextmanager
async def lifespan(app):
    torch.set_num_threads(4)
    state["tokenizer"] = AutoTokenizer.from_pretrained(MODEL_ROOT, local_files_only=True)
    state["model"] = AutoModelForImageTextToText.from_pretrained(
        MODEL_ROOT, local_files_only=True, dtype=torch.bfloat16,
        device_map="cuda:0", attn_implementation="sdpa").eval()
    yield
    state.clear()
    torch.cuda.empty_cache()


app = FastAPI(lifespan=lifespan)


class Message(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    model: str
    messages: list[Message] = Field(min_length=1)
    max_tokens: int = Field(default=128, ge=1, le=256)
    temperature: float = Field(default=0, ge=0, le=2)
    top_p: float = Field(default=1, gt=0, le=1)
    top_k: int = Field(default=20, ge=0)
    stream: bool = False
    chat_template_kwargs: dict = Field(default_factory=lambda: {"enable_thinking": False})


@app.get("/health")
def health():
    return {"ready": "model" in state, "model": NAME, "weights": str(MODEL_ROOT)}


@app.get("/v1/models")
def models():
    return {"object": "list", "data": [{"id": NAME, "object": "model", "owned_by": "local"}]}


@app.post("/v1/chat/completions")
def complete(request: ChatRequest):
    if request.model != NAME or request.stream:
        raise HTTPException(400, "This smoke server supports model rsi-w0 and non-streaming requests")
    with LOCK, torch.inference_mode():
        tokenizer, model = state["tokenizer"], state["model"]
        text = tokenizer.apply_chat_template([m.model_dump() for m in request.messages],
            tokenize=False, add_generation_prompt=True,
            enable_thinking=bool(request.chat_template_kwargs.get("enable_thinking", False)))
        inputs = tokenizer(text, return_tensors="pt").to("cuda:0")
        prompt_tokens = inputs["input_ids"].shape[-1]
        if prompt_tokens > 4096:
            raise HTTPException(400, "Smoke server prompt limit is 4096 tokens")
        kwargs = {"max_new_tokens": request.max_tokens, "do_sample": request.temperature > 0}
        if request.temperature > 0:
            kwargs.update(temperature=request.temperature, top_p=request.top_p, top_k=request.top_k)
        output = model.generate(**inputs, **kwargs)
        tokens = output[0, prompt_tokens:]
        content = tokenizer.decode(tokens, skip_special_tokens=True)
        count = len(tokens)
    return {"id": "chatcmpl-" + uuid.uuid4().hex, "object": "chat.completion",
            "created": int(time.time()), "model": NAME,
            "choices": [{"index": 0, "message": {"role": "assistant", "content": content},
                         "finish_reason": "length" if count == request.max_tokens else "stop"}],
            "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": count,
                      "total_tokens": prompt_tokens + count}}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=18541)

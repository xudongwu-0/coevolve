"""Implement only the API contract imported by WHALE's chess runner."""
from dataclasses import asdict, dataclass
import json
import os
from urllib.parse import urlsplit
from urllib.request import Request, build_opener, ProxyHandler


@dataclass
class ChatMessage:
    role: str
    content: str


@dataclass
class Completion:
    content: str
    usage: dict


class LLMClient:
    def __init__(self, config):
        if config.provider != "openai-compatible":
            raise ValueError("Local compatibility layer supports only openai-compatible")
        self.config = config

    def complete_response(self, messages, max_tokens=None):
        cfg = self.config
        body = {"model": cfg.model, "messages": [asdict(m) for m in messages],
                "max_tokens": max_tokens or cfg.max_tokens, "temperature": cfg.temperature,
                "top_p": cfg.top_p, "top_k": cfg.top_k,
                "chat_template_kwargs": cfg.chat_template_kwargs}
        headers = {"Content-Type": "application/json"}
        key = os.environ.get(cfg.api_key_env)
        if key:
            headers["Authorization"] = f"Bearer {key}"
        request = Request(cfg.api_base.rstrip("/") + "/chat/completions",
                          data=json.dumps(body).encode(), headers=headers)
        opener = build_opener(ProxyHandler({})) if urlsplit(cfg.api_base).hostname in {"localhost", "127.0.0.1", "::1"} else build_opener()
        with opener.open(request, timeout=180) as response:
            result = json.load(response)
        return Completion(result["choices"][0]["message"]["content"], result.get("usage", {}))

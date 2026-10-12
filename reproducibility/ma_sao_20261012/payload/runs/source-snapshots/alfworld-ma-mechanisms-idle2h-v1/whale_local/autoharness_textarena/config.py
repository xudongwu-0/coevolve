from dataclasses import dataclass, field


@dataclass
class LLMConfig:
    provider: str = "openai-compatible"
    model: str = "rsi-w0"
    api_base: str = "http://127.0.0.1:18541/v1"
    api_key_env: str = "WHALE_LOCAL_API_KEY"
    temperature: float = 0.0
    top_p: float = 1.0
    top_k: int = 20
    max_tokens: int = 128
    batch_concurrency: int = 1
    chat_template_kwargs: dict = field(default_factory=lambda: {"enable_thinking": False})

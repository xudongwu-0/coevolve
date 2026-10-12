"""Scoped compatibility fix for singleton Qwen3.5 padded training batches.

Transformers 5.3.0's helper incorrectly requires batch size > 1. Native
WHALE's local microbatch is 1, so left-padding then contaminates the linear
attention state. Preserve the helper's sequence-length guard for decoding,
but apply its existing mask operation for any batch size. No weight changes.
"""
import hashlib
import inspect


def mask_padding_states(hidden_states, attention_mask):
    if attention_mask is not None and attention_mask.shape[1] > 1:
        hidden_states = (hidden_states * attention_mask[:, :, None]).to(hidden_states.dtype)
    return hidden_states


def install():
    import transformers
    from transformers.models.qwen3_5 import modeling_qwen3_5 as model_code
    original = model_code.apply_mask_to_padding_states
    source = inspect.getsource(original)
    if original is mask_padding_states:
        return model_code._whale_padding_fix
    if 'attention_mask.shape[0] > 1' not in source:
        raise RuntimeError("Qwen3.5 padding implementation changed: revalidate the scoped compatibility fix")
    metadata = {"name": "qwen35-singleton-padding-mask-v1", "transformers_version": transformers.__version__,
                "original_function_sha256": hashlib.sha256(source.encode()).hexdigest(),
                "replacement_function_sha256": hashlib.sha256(inspect.getsource(mask_padding_states).encode()).hexdigest()}
    model_code.apply_mask_to_padding_states = mask_padding_states
    model_code._whale_padding_fix = metadata
    return metadata

# Qwen3.5-9B Target Agent SFT (A1)

This directory contains the direct Agent-SFT target model used as **A1** in
Harness-R1 experiments. It is a task-execution agent, not a Harness Engineer
that reads failure traces and produces harness patches.

## Training

- Base model: `Qwen/Qwen3.5-9B`
- Method: full-parameter supervised fine-tuning
- Data: 2,515 successful source trajectories, deduplicated by task
- Epochs: 2
- Learning rate: `1e-5`
- Effective training batch size: 24
- Seed: 42

The training trajectories cover the task-agent side of the experiments. The
separate `harness-r1/` and `agent-sft-harness-r1/` directories contain Harness
Engineer checkpoints and should not be substituted for this target agent.

## Files

The directory contains the final inference checkpoint and tokenizer artifacts.
Trainer state, intermediate checkpoints, and training logs are intentionally
excluded.

## Loading

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

repo = "ShaoShuai0605/Harness-R1"
subfolder = "target-agent-sft"

tokenizer = AutoTokenizer.from_pretrained(
    repo,
    subfolder=subfolder,
    trust_remote_code=True,
)
model = AutoModelForCausalLM.from_pretrained(
    repo,
    subfolder=subfolder,
    trust_remote_code=True,
    torch_dtype="auto",
    device_map="auto",
)
```

Use a recent Transformers or SGLang release with Qwen3.5 support.

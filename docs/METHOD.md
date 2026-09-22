# Method: learning the harness editor inside WHALE

Maintain `(W_t, E_t, H_t)`: solver weights, editor weights, and executable harness. `W` and `E` are independent Qwen3.5-2B checkpoints, not two prompts for a single shared parameter tensor.

## One editor phase

1. Freeze `W_t` and evaluate the incumbent `H_t` on 48 development tasks.
2. Give `E_t` the harness source and a small packet of observed failures. Reference answers are not included in the editing request.
3. Sample eight complete `CandidateEnv` modules. Validate the interface and runtime; score valid changed candidates on all 48 development tasks with the same solver, question order and sampling seeds.
4. Reward candidate `i` with `r_i = accuracy(W_t, H_i) - accuracy(W_t, H_t)`. Invalid/no-op candidates receive zero delta. Rewards cover successes as well as failures so that regressions matter.
5. Standardize rewards within the eight-candidate group. Update `E` using generated code tokens and recorded log probabilities. Skip the update when every group has zero variance.
6. Generate four proposals from the resulting checkpoint; evaluate them and the incumbent, then use the original WHALE Pareto frontier and acceptance rule to select the next harness.

The outcome-GRPO implementation uses population-standard-deviation normalization, asymmetric clipping `[0.8, 1.28]`, a detached importance correction capped at 2, no KL penalty, no format bonus, and sequence-mean then batch-mean loss. Each editor phase makes up to four optimizer updates using the same feedback batch; these are not four new independent feedback batches. See `harness_grpo.py`.

## Solver phase and alternation

After the first editor/harness phase, sample 32 training trajectories using **that arm's own current W/H**. Retain verifier-successful trajectories for native WHALE RSFT, using exact recorded response tokens. Save and reload the new solver checkpoint, then run the second editor/harness phase against this updated solver.

The three-arm pilot order is:

```text
W_initial, E_initial, H_initial
    -> [generate H candidates -> update E -> regenerate/select H1]
    -> [generate solver trajectories under H1 -> update W]
    -> [generate H candidates with updated W -> update E -> select H2]
    -> freeze all stage states -> evaluate every stage
```

The fixed-editor arm skips E updates. The external arm obtains proposals from an external actor instead of E. Solver sampling/training, the editable code interface and selection rules remain shared. This is sequential alternating optimization, not simultaneous gradients through W and E.

## What is inherited and what is added

| Component | Origin |
|---|---|
| Math ReTool base harness and strict boxed-answer reward | WHALE, unchanged upstream submodule |
| `DataParallelPPOActor.update_sft_policy` | WHALE's vendored math-domain verl actor |
| Harness Pareto/acceptance rule | WHALE meta-harness |
| Trainable editor, recorded code-token GRPO, dual-checkpoint handoff | Coevolve extension |
| Local model service, native environment adapter, isolated code execution, replica pool | Coevolve local execution support |

The editor can modify `SYSTEM_PROMPT`, `USER_PROMPT_TEMPLATE`, `MAX_TURNS` (1–3), and two asynchronous hooks: `no_tools_called` and `get_prompt_messages`. It inherits tools and token-budget enforcement. It cannot modify grading or reference answers. This is a restricted executable-code search space, not unrestricted software engineering.

## Interpretation boundaries

- Post-update selection uses only four post proposals and the incumbent. Pre-update proposals train E but are not retained in this selection pool. This controlled comparison is not a best-of-all-searched baseline.
- The fixed editor repeats the same four proposal seeds before/after. Additional fresh samples for that baseline were not evaluated.
- Candidate rewards use the same development set used for selection; observed editor training improvements do not establish transfer to unseen editing contexts.
- An unchanged model/harness can reuse a cached rollout; logical per-arm usage and physical cache reuse are different budgets.
- Test outcomes are opened after all stages are frozen. The question set nevertheless has prior use across historical iterations.
- No mechanism guarantees held-out performance to be monotone. The incumbent protects a development-set comparison, not all future questions.

These qualifications are part of the method's current experimental status, including the reported 93/128 result.

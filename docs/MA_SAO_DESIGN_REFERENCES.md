# MA-SAO: source-grounded design constraints

This note summarizes primary-source research for the H-W-H setting. It does not
report a new experimental win or turn the earlier recycling results into
centralized-critic results. The runtime still uses the pinned unofficial
SAO/AReaL implementation, not an official implementation inferred from a paper.

## Primary references

- [MAPPO: The Surprising Effectiveness of PPO in Cooperative, Multi-Agent Games](https://arxiv.org/html/2103.01955v4),
  especially Sections 3.2, 5.2 and 5.3.
- [Official MAPPO implementation](https://github.com/marlbenchmark/on-policy/tree/de66d7a4b23fac2513f56f96f73b3f5cb96695ac),
  also pinned in [the local reference](../third_party/mappo-reference).
- [MA-PPO for request dispatching in distributed-controller SDN](https://arxiv.org/html/2103.03022v1#S4.SS3),
  Section IV-C. The name MA-PPO is overloaded; this is a multi-agent method,
  unlike similarly named model-accelerated PPO.
- [Single-Rollout Asynchronous Optimization for Agentic Reinforcement Learning](https://arxiv.org/html/2607.07508v1),
  Sections 3.1, 3.2 and 4.1.
- [Generalized Advantage Estimation](https://arxiv.org/html/1506.02438),
  especially the estimator bias discussion and the same-data value-fitting caveat.
- [IMPALA and V-trace](https://arxiv.org/html/1802.01561), Section 4.

## Centralized information is different from shared parameters

The MAPPO policy receives local actor observations and separate centralized
critic observations. Its MPE runner constructs the latter from actual agent
observations, not merely a role or phase name:

- [Actor/critic input interface](https://github.com/marlbenchmark/on-policy/blob/de66d7a4b23fac2513f56f96f73b3f5cb96695ac/onpolicy/algorithms/r_mappo/algorithm/rMAPPOPolicy.py#L48-L74).
- [MPE centralized observations](https://github.com/marlbenchmark/on-policy/blob/de66d7a4b23fac2513f56f96f73b3f5cb96695ac/onpolicy/runner/shared/mpe_runner.py#L81-L93).
- [Separate per-agent policies and buffers](https://github.com/marlbenchmark/on-policy/blob/de66d7a4b23fac2513f56f96f73b3f5cb96695ac/onpolicy/runner/separated/base_runner.py#L85-L111).

A role-conditioned local value function is not automatically centralized.
Separate H/W critics can still be centralized if their inputs include relevant
joint information. Conversely, sharing one value checkpoint does not prove that
it has enough state information or that it performs counterfactual credit
assignment. H's paired candidate/reference reward is a separate mechanism.

For H/W with different reward definitions, use role-specific value targets;
canonical shared-reward MAPPO does not by itself justify one common scalar target.

## What action importance ratios do not fix

For a stored prefix h and sampled token a, the action ratio is

`r = exp(log pi_current(a | h) - log pi_behavior(a | h))`.

This addresses the conditional action distribution. It does not, by itself,
correct prefix visitation, partner-policy changes, tool transitions, termination
rules or outcome-dependent sample selection. Native SAO's masked ratio estimator
also deliberately retains approximation bias; it is not a general exact
trajectory change-of-measure estimator.

An unchanged W can yield token ratios of one while a trajectory terminated at
one tool horizon is still invalid as a completed return for a longer horizon.
Do not use low DIS rejection as proof that different harness contexts are
interchangeable. Greedy/top-k sampling needs separate support/probability checks;
raw model probabilities are not automatically the behavior distribution.

If data from several harnesses enter the actor update, define the objective as
a mixture over those contexts. Do not call it an unbiased selected-harness-only
objective. Success-only filtering likewise changes the sampling distribution.

## Design 1: bounded evaluation-experience recycling

This is the first recommended experiment; it requires no centralized-critic claim.

1. Freeze the solver while collecting candidate/reference evaluations. Retain
   exact tokens, prompts, observations, log-probabilities, versions, termination
   causes and the actual harness/tool configuration.
2. Declare the target context distribution. For the cleanest variant, admit only
   episodes matching one target harness and horizon. A mixture is possible, but
   its objective and hidden-context limitations must be explicit.
3. Split before inspecting rewards into disjoint critic and actor pools.
   Prefer task or coupled candidate/reference seed groups over independently
   splitting correlated episodes; keep paired episodes together.
4. Warm the critic on the first pool through the native value update while the
   actor parameters, optimizer state and schedule remain unchanged.
5. Compute fresh native values/GAE for the actor pool after warm-up. Preserve
   original behavior probabilities and native clipping/normalization semantics.
6. Admit each actor-pool episode once, then expire it after the declared block.
   Report actual optimizer reuse separately from buffer admission.

A completed old return estimates its behavior-policy value in its original
context. It does not become a current-policy or new-harness value target merely
by recomputing a value prediction. Disjoint warm-up reduces direct same-sample
value fitting but does not eliminate policy/context mismatch.

The SAO paper already discusses critic warm-up and faster critic learning.
Warm-up itself must not be presented as our novelty. The proposed distinction is
which already-paid partner-evaluation experience is retained and how it is used;
baseline tuning must not omit a useful native setting merely to create a weak
control.

Section 4.1 explicitly states a ten-step warm-up, critic learning rate 5e-6,
critic GAE lambda 1, general batch size 128, group size 1 and K=2. It does not
unambiguously define whether those ten steps are batches or optimizer updates,
their data source, or whether actor updates are disabled. Do not infer 1,280
distinct warm-up episodes or ten actor-frozen batches from that paragraph.
The same section describes three epochs of mathematical tool-integrated
reasoning fine-tuning on GPT-OSS-120B-generated data to initialize policy and
value models; that is not the configuration-editing SFT used in our small pilot.
These are protocol gaps to investigate, not proof of the cause of our regression.

For H replay after W changes, re-evaluate both the candidate and its original
reference under W's new weights. Preserve the old H prefix and behavior logps.
If intermediate H observations changed, recollect that trace instead of attaching
old actions to a new prefix. Refreshing rewards does not remove the difference
between old and current incumbent contexts.

## Design 2: causal joint-state value conditioning

Only after Design 1 is tested, change critic conditioning while keeping data,
native objectives and update budgets controlled. Potential inputs include the
already-existing harness, tool state, remaining token/tool budgets, partner
version/sampling settings and feedback genuinely observed before the action.

Do not prepend completed edits, later solver success, realized returns or winner
labels to earlier action tokens. Dynamic context must be aligned to when it was
available. A checkpoint hash is an identity label, not proof of sufficient state.
Actor inputs and action-token masks remain unchanged.

The runtime support for correctly aligned dynamic joint context must be verified
before this design is called implemented. Static phase prefixes do not satisfy
that verification.

## Smallest informative ablations

| Comparison | What it isolates |
|---|---|
| Native baseline vs extra optimization on existing data | Additional optimizer work |
| Matched extra optimization vs partner actor data | Additional experience source |
| Same actor data, existing value data vs disjoint partner warm-up | Value calibration data |
| Same data/updates, local vs genuine causal joint context | Additional critic information |
| Genuine vs matched-length shuffled context | Information rather than prefix length |
| H replay with old vs refreshed downstream rewards | Counterpart-dependent reward drift |

Report generated episodes, unique admitted episodes, masked/unmasked update
tokens, critic/actor steps, GPU time, age/context strata and held-out task/value
performance. Recycled evaluation examples are training data and must never be
reused as held-out evidence. These ablations are proposed causal tests, not
already-established results for our implementation.

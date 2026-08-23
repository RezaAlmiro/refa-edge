# 05 — The frontier, and the axis where open/small can win

## What frontier labs are doing in the areas this program touches (verified to Aug 2026)

- **Memory substrates are mainstream.** DeepSeek Engram (hashed conditional memory,
  reportedly the V4 blueprint); Meta Memory Layers (product keys verified to 128B memory
  params) and Sparse Delta Memory (1000× state at isoFLOPs); Google Titans/ATLAS
  (surprise-gated test-time memory); Sakana FwPKM (product keys × fast weights);
  production gated-delta backbones (Qwen3-Next, Kimi Linear/KDA, RWKV-7).
- **Efficiency toolkits are mature**: MoD, speculative decoding (lossless), MLA latent
  KV, BitNet ternary, SpikingBrain conversion training.
- **Hallucination work is post-hoc**: semantic-entropy probes and successors, plus
  incentive repair (Kalai et al.). Monitoring is bolted on, never built into the memory.
- **None of it has**: stance-typed slots, provenance inside the representation,
  certified abstention, queryable time, or an auditable revision operator. Verified by
  sweep search: these remain unoccupied.

## The three ceiling results that define the winnable axis

These are *structural*, not effort gaps — scale and RL do not fix them:

1. **Bengs et al. (+ 2024 follow-ups):** no proper scoring rule trains a faithful
   second-order (epistemic) uncertainty head. The frontier's "train the model to know
   what it knows" path has a theoretical ceiling.
2. **AbstentionBench (2026):** abstention does not improve with scale, and
   reasoning-finetuning actively degrades it. More compute does not buy honesty.
3. **Kalai et al.:** a statistical floor plus a benchmark-incentive problem guarantees
   confident guessing under current training economics.

Corollaries the program should keep exploiting: length generalization is
high-variance and mostly a training-distribution problem (fixable at ~0.1% cost);
capacity of linear memories is a law (√(R/(T−1))), not a mystery; and continual
self-updated memory corrupts without explicit gates (verified 2026).

## Therefore: the axis

Do not compete on capability-per-dollar-of-training — that axis is owned. Compete on
**properties per parameter**:

| Property | Frontier state | This program's approach |
|---|---|---|
| Knowing what it doesn't know | learned probes, provably unfaithful; doesn't scale | measured evidence counts + certificates + conformal wrappers |
| Saying where an answer came from | approximate post-hoc attribution | exact provenance polynomials over an affine memory |
| Revising beliefs | finetuning or context stuffing | key-addressed revision with auditable, typed semantics |
| Time | positional encodings | queryable Laplace age codes |
| Being measurable | closed eval teams | open per-organ battery anyone can run |

Each row is demonstrable at 1M parameters, publishable with theory attached, and — if
K1–K4 (see `04-CRITICAL-ASSESSMENT.md`) pass — composable into a genuinely different
model class whose value proposition is *trust per watt*, not tokens per second.

## Priority risks on this axis

- A frontier lab adds provenance tags to a memory-layer substrate (the most likely
  absorption). Mitigation: pre-register predictions and ship the battery first —
  scientific priority is citable, engineering priority is not.
- The properties turn out not to matter to users versus raw capability. Mitigation:
  target deployments where trust is the product (medicine-adjacent triage, law,
  education, on-device assistants) in framing and evals.
- The properties hold only on synthetic symbols (the K4 boundary). This is the honest
  biggest risk; it gets its own task lane.

# 06 — Compute levers, honestly rated

The mission's hardest clause is "cut the negative limitations of compute." Here is
every credible lever the research base supports, with an honest rating of what it
actually buys. Ratings: ▮▮▮ proven here or in the literature at relevant scale ·
▮▮ strong evidence, needs local validation · ▮ plausible, unproven.

## Inference & training mechanics (short horizon)

1. **▮▮▮ Chunkwise WY kernels + CUDA graphs** — the Python loop is launch-bound;
   the rewrite is *exact*. 10–50× wall clock, ~30× activation memory at REFA scale.
   O(1) state at inference is inherited from the linear-attention family. (Leap 1.)
2. **▮▮▮ μP proxy tuning** — collapse hyperparameter search to 30k-param proxies with
   zero-shot transfer. This is the single largest multiplier on *research* compute,
   which for a small lab is most compute. Needs the coordinate-check gate. (Leap 10.)
3. **▮▮ Conditional compute by surprise** — mixture-of-depths routing, risk-controlled
   early exit (CALM/LTT), lossless speculative decoding. The Engram Loop's one-currency
   claim (one prediction-error signal gates depth, writes, abstention) is testable at
   REFA scale and would unify three gates into one estimator. (Aim 4; sweep: substrate.)
4. **▮▮ Quantization / spiking** — BitNet-class ternary weights; SpikingBrain-style
   conversion training at ~2% of from-scratch tokens; int8 cold-bank state. At REFA's
   size these are engineering hygiene; at scale they are the energy story.

## Memory instead of parameters (the deep lever)

5. **▮▮▮ Sparse memory substrates** — product-key addressing gives √N-cost retrieval
   over huge slot counts; Sparse Delta Memory reaches 1000× state at isoFLOPs; DeepSeek
   Engram offloads to host RAM at <3% overhead. Knowledge living in cheap addressable
   memory rather than dense matmuls is the clearest published path to capability per
   FLOP — and it is exactly REFA's substrate family. (Leaps 11; sweep: substrate-2026.)
6. **▮▮ Consolidation instead of context** — the CLS loop: episodic store distilled
   into sparse parametric memory between sessions (sparse memory finetuning is the
   verified low-forgetting anchor). Replaces both long-context recomputation and full
   finetuning. Provenance-preserving distillation is a genuine open gap = opportunity.
7. **▮▮ Retrieval as a training-compute cut** — the RETRO/kNN-LM lineage showed
   competitive quality with far fewer parameters by externalizing knowledge; it also
   gives provenance by construction. Under-explored in this program; connect it to the
   evidential store. (sweep: ml-scout.)

## Learning rules (edge-training horizon)

8. **▮▮ O(1)-memory online gradients** — RTRL is *exact and cheap* for REFA's linear
   memory path (no unrolling); e-prop covers the GRU; RLS gives convex head training.
   Together: continual on-device adaptation without BPTT's memory bill. (sweep:
   local-learning.)
9. **▮ Analog / physical substrates** — equilibrium propagation reached ImageNet scale
   in 2026 with hardware demos; Waves-over-Weights P10 argues the control plane is the
   right analog target. Long horizon; simulation-first.

## The missing pillar (not yet swept — highest-priority new research)

10. **▮? Decentralized & collaborative training/inference** — DiLoCo-style
    low-communication training, the INTELLECT line of internet-scale open pretraining,
    Petals-style distributed inference, federated finetuning, gradient-compression
    (DisTrO-class) methods. For "available to all, not only a few," this is the pillar
    the program has zero coverage of. Note the native fit: **mergeable sketches +
    forward-decay banks are CRDT-like by construction** — REFA's memory was
    accidentally designed for federation (exact merges across devices). Sweep this
    area first; see `07-DIRECTIONS.md`.

## Honest limits — read before quoting any of the above

- Efficiency is not capability. Nothing above substitutes for scale on raw capability;
  the levers cut the cost of what a small model *can* do and widen who can do it.
- The brain's 20 W is an existence proof that today's costs are not physics — but the
  gap closes through substrates and algorithms co-designed over decades, not through
  any single trick in this list.
- Wall-clock claims (lever 1) are for REFA's size class; do not extrapolate them.
- Lever 10's field moves fast and has its own hype problem; sweep with the same
  adversarial-verification standard as everything else.

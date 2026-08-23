# 01 — Mission

## The owner's aim, in his words

> Find ways to cut all the negative limitations of compute, with the super scaled
> intelligence and ability — and for this to be available and open sourced for all,
> not only a few.

## What that credibly translates to

The aspiration is real and worth holding. It also cannot be pursued as stated without
self-deception, so this program operationalizes it as four goals, in order of how
defensible they currently are:

**G1 — Trustworthy intelligence that runs on hardware people actually own.**
Models whose abstention is certified, whose citations are exact, and whose memory is
auditable — properties frontier models *structurally lack* (see `05-FRONTIER-AXIS.md`)
— demonstrated on consumer GPUs. This is the program's strongest claim to novelty.

**G2 — Compute reduction as first-class science.**
Every lever that reduces training or inference compute without buying the reduction
with silent capability loss: linear-state substrates, conditional compute, consolidation
instead of context, local learning, proxy-scale tuning, quantization. Rated honestly in
`06-COMPUTE-LEVERS.md`. The brain runs on ~20 W; the gap between that and a datacenter
is the standing proof that current compute costs are not physical law.

**G3 — Instruments for everyone.**
Open benchmarks and diagnostics (the Confabulation Battery; crosstalk instrumentation;
per-organ failure profiles) that let *anyone* measure what only labs with eval teams can
measure today. Instruments outlive architectures and cannot be moated.

**G4 — A path to scaled capability that is open end-to-end.**
This is the hardest and least advanced goal. The honest state: no known architecture
substitutes for scale on raw capability. The credible open path is compositional —
efficient substrates (G2) + externalized knowledge (memory instead of parameters) +
decentralized/collaborative training (NOT yet swept — see `07-DIRECTIONS.md`, priority
one) + distillation from open models. Treat G4 as a direction to build toward, not a
claim to make.

## What we will not claim

- That any mechanism here beats frontier models at general capability. It doesn't, and
  pretending otherwise would burn the program's credibility — its only real asset.
- That biological inspiration is evidence. A mechanism ports only when it solves a
  *measured* ML problem; the biology supplies the search prior, never the proof.
- That efficiency gains at 1M parameters transfer to 1B. μP is the designed bridge;
  until its coordinate checks pass, small-scale numbers are small-scale numbers.

## Values (non-negotiable)

1. **Falsifiability** — every bet ships with a kill criterion, pre-registered.
2. **Retraction culture** — wrong claims are retracted loudly and kept as records.
3. **Exactness gates** — reference implementation + exact/certified test before benchmark.
4. **Honest reporting** — config, seed, raw output, hardware, commit hash, always.
5. **Openness** — MIT; no dependency on privileged compute, private data, or closed weights.
6. **The unglamorous truth wins** — an instrument, a negative result, or a 10× kernel
   speedup that is real beats an architecture story that is not.

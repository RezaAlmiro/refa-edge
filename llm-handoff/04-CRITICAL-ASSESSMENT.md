# 04 — Critical assessment (standing challenges)

Written by the previous agent at the owner's request for a *truly critical* verdict.
Treat every paragraph as a standing challenge: your job is to resolve these, not to
work around them. If you find this assessment itself is wrong somewhere, say so with
evidence — that is also progress.

## The deflating truths

1. **The core mechanism is commodity.** REFA's recurrence is the gated delta rule that
   Qwen3-Next, Kimi Linear, and RWKV-7 already ship in production. Meta, Google,
   DeepSeek, and Sakana are actively racing on memory substrates (Sparse Delta Memory,
   Titans/ATLAS, Engram, FwPKM). The recurrence, kernels, and routing are not ours and
   cannot be the differentiator.
2. **Most of the 20 leaps are competent assembly, not discovery.** Held to the standard
   "would anyone cite this as an origin," perhaps 3–4 survive. The neuroscience ports
   (excitability tags, query sculpting, ACh/NE, clone-or-revise) are workshop-paper
   material until they solve a *measured* problem.
3. **Zero experiments have run.** Verification confirmed citations are real and
   accurately characterized — it did not confirm any mechanism works. Everything is
   still hypothesis.
4. **The symbolic→text boundary is the most likely cause of death.** Every certificate
   (Bloom, CMS, IBLT) rides on exact (s,r,o) keys. A text encoder dissolves them into
   learned representations unless someone does real work at that boundary
   (canonicalization layers? entity linking as a first-class module? certified
   guarantees on canonicalized subsets?). Attack this early; do not leave it for last.
5. **Crowding and priority risk.** The memory-substrate race will plausibly absorb
   several of these ideas within 12–18 months. Scientific priority (pre-registered
   predictions, the battery, published negative results) is defensible; engineering
   priority is not.

## What survives a hostile committee (defend these, but try to kill them first)

- **Thesis A: uncertainty as a measured property of memory.** Backed by two verified
  ceiling results: trained second-order heads cannot be faithful (Bengs et al.), and
  abstention does not improve with scale (AbstentionBench). Measured evidence counts +
  certified sketches + conformal wrappers circumvent the ceiling rather than push on it.
  This is the program's strongest claim to a *different kind* of model.
- **Thesis B: provenance exact by construction.** Affine memories decompose reads into
  exact signed per-event responsibilities. Real, small, partial (linear path only — the
  workspace is not covered). Promising as a research direction, not a full-model story.
- **Thesis C: typed opposition memory.** No published system has support/opposition-typed
  slots; real brains (mushroom body MBONs, D1/D2 opponency, inhibitory engrams) do build
  opponent evidence channels. Biologically warranted, empirically untested.

## The four kill experiments (run before extending anything)

| # | Experiment | Kills | Keep-alive criterion |
|---|-----------|-------|----------------------|
| K1 | Typed M+/M− banks vs a single signed-value store, parameter-matched, full battery, ≥10 seeds | Thesis C / the founding bet | typed banks win on revision/conflict/abstention beyond noise |
| K2 | Gram-measured (b,d,u) head vs trained 3-logit head: risk-coverage + smECE, in- and out-of-length-distribution | Thesis A's practical half | measured head matches accuracy and beats calibration OOD |
| K3 | Parity-of-negations before/after the (0, 1+λ) gate rescale | the expressivity story | expressive mode solves and length-extrapolates; safe/current modes plateau |
| K4 | A minimal text front-end (even a toy canonicalizer) over the certified path | the symbolic boundary objection | some certified coverage survives paraphrase; quantify the degradation curve |

If two of the four fail, the honest conclusion is that this is an
**instruments-and-theory project** (battery + capacity/evidential math), not an
architecture project — and that conclusion should be embraced, published, and built on,
not hidden.

## Challenges to this assessment itself

- The "commodity core" judgment assumes frontier delta-rule work keeps its current
  direction. If the field pivots (e.g., back to quadratic attention with KV innovations),
  the small-recurrence niche could reopen. Watch it.
- The 3–4-survivors triage of the leaps was one agent's judgment under time pressure.
  Re-triage with fresh eyes; promote or demote with reasons.
- "Instruments beat architectures for solo researchers" is a prior, not a law. If K1–K3
  all pass cleanly, the architecture story strengthens materially and this document
  should be rewritten.

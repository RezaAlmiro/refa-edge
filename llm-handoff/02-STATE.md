# 02 — Exact current state (as of 2026-08-23)

## The three layers

1. **REFA Edge** (this repo) — runnable MVP. Typed support/opposition fast-weight banks
   with key-addressed delta-rule revision, soft relation routing over H=8 banks, GRU
   hypothesis workspace, 3-class output (supported/opposed/unknown). Baselines: GRU,
   small transformer, dense + streamed fast-weight comparators with float64
   output+gradient equivalence tests. Synthetic relational-revision tasks. Runs on an
   RTX 2060 6 GB.
2. **The Engram Loop** (proposal, 2026-08-22) — four-organ architecture program
   (hippocampal store / temporal stack / reality monitor / surprise economy + sleep)
   with a 26-row brain→ML ledger and a planned Confabulation Battery.
   Text: `../engram-loop-handoff/recovered/the-engram-loop.extracted.txt`.
3. **Waves Over Weights** (review, 2026-08-21) — waves-as-control review with ten
   proposals (P1–P10). Speculative tier; nothing in the 08-23 sweep advanced its
   evidential status. Text: `../engram-loop-handoff/recovered/waves-over-weights.extracted.txt`.

## Code facts you should not rediscover

(from `src/refa_edge/models/refa.py` and `models/common.py`)

- Update per event, per bank h, stance s ∈ {+,−}:
  `M ← λ_h·M − β·r_h·k(kᵀM) + β·r_h·1[s=polarity]·k·vᵀ`, with ‖k‖=1 (L2-normalized),
  v tanh-squashed, β = sigmoid(gate), r = softmax routing, λ learned per bank.
  Erase hits BOTH stances; write goes to the event's stance only.
- **This is exactly a constant-gate gated delta rule** (verified independently by four
  sweep areas). Transition spectrum {λ − βr, λ} ⊂ (−0.005, 0.995] — contractive, and
  provably unable to express parity-class (alternating-revision) state tracking.
- Triple binding is ADDITIVE (entity/relation embeddings summed with role offsets,
  LayerNormed) — the weakest binding form; the dataset's close-decoy unknowns stress it.
- Keys = queries = normalized projection of identity (R=16). Capacity ≈ R orthogonal
  associations per (bank, stance); crosstalk SNR = √(R/(T−1)).
- No time code of any kind: recency exists only through decay and erase-rewrite.
- Classifier sees the workspace + final-step recalls; query is always the last row.
- "Unknown" is a learned softmax class with no calibration story.
- The forward pass is a Python per-step loop — kernel-launch-bound, not FLOP-bound.

## What the 2026-08-23 ultra sweep established

38 agents (13 finders → adversarial verifiers → completeness critic → 4 follow-up
finders → 3-lens leap panel), ~3.5M research tokens. Products: 188 verified findings
(`../engram-loop-handoff/SWEEP-2026-08-23.md`), 20 leaps ranked in 3 tiers
(`../engram-loop-handoff/CONNECTIVE-LEAPS.md`), 3 retractions, 10 direct adoptions.

The three organizing discoveries:

1. **The delta-rule identity** — REFA's loop = gated delta rule ⇒ exact WY chunkwise
   parallelization (10–50× wall clock, ~30× activation VRAM), imported stability and
   expressivity theory, and the whole test-time-regression upgrade ladder.
2. **Measured evidence beats learned evidence** — subjective-logic opinions ↔ Beta
   counts; a trained second-order head provably cannot be faithful (Bengs et al.);
   evidence counts are computable from an erase-consistent write Gram; certified
   unknowns come free from sketches (Bloom FN=0, one-sided count bounds).
3. **A capability cliff found by theory** — eigenvalues in [0,1) block parity-class
   revision; rescaling the write gate to (0, 1+λ) fixes it at zero cost.

## What has NOT been done — read this twice

- **A1/K3 is the only kill experiment run.** The queued alternating-polarity task was
  rejected for label leakage and replaced by keyed write-count parity. Default
  expressive gating was bimodal (5/10 memory-only seeds solved); a separately
  preregistered spectrum-derived gate initialization reached median length-64 accuracy
  0.893 versus ~0.50 controls, and full REFA reached 0.930. This validates a small
  spectral mechanism, not the wider architecture. See
  `../docs/A1-KEYED-PARITY-RESULTS.md`.
- The founding bet (typed opposition banks beat a signed single store) is untested.
- The symbolic→text boundary (certified guarantees currently need exact (s,r,o) keys)
  is unaddressed.
- μP transfer for this architecture is unproven (coordinate checks not yet implemented).
- The Confabulation Battery exists only as a design.
- Waves Over Weights P1–P10: all still unbuilt and unevidenced.
- Two critic-identified areas never swept: region-based (box/cone) embeddings;
  schema-congruency-gated consolidation. Plus the big one: decentralized training
  (see `07-DIRECTIONS.md`).

## Where everything lives

- Branch: `claude/brain-research-machine-mapping-j2lbdc` (pushed).
- Published artifacts (claude.ai, owner's account): *The Engram Loop*, *Waves Over
  Weights*, *The Delta Ledger* (the 08-23 synthesis).
- Running log + next-actions queue: `../engram-loop-handoff/HANDOFF.md`.
- Immediate next decision: A2/K1 typed banks versus a signed single store. The frozen
  design constraints and kill rule are in `../docs/NEXT-RESEARCH-STATE.md`.

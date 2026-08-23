# Engram Loop / REFA Edge — project handoff

Last updated: 2026-08-23 (session: brain-research → machine-mapping ultra sweep).
Owner: Reza Almiro (almirofilms@gmail.com).

This directory is the durable home of the research program that surrounds the REFA Edge
codebase. A previous session kept this handoff only in an ephemeral workspace and it was
lost when the container was reclaimed; everything here is now committed to the repo so no
future session starts blind. Read this file first, then the documents it points to.

## What this project is

Three layers, from concrete to speculative:

1. **REFA Edge** (this repository) — a runnable, falsifiable research MVP: typed
   support/opposition fast-weight memory banks with key-addressed delta-rule revision,
   soft relation routing, and a recurrent hypothesis workspace, benchmarked fairly
   against GRU/Transformer/fast-weight baselines on synthetic relational revision tasks.
   Target hardware: RTX 2060 6 GB / 32 GB RAM. See `docs/ARCHITECTURE.md`.
2. **The Engram Loop** (research proposal, 22 Aug 2026) — a brain-derived architecture
   program: hippocampal episodic store (provenance-typed, surprise-gated), temporal stack
   (theta–gamma nested phase code + Laplace clock), reality monitor (trained second-order
   abstention head), and a surprise economy with a sleep/consolidation cycle. Contains a
   26-row brain→ML association ledger and a neuropsychology-derived Confabulation Battery.
   - Published artifact: https://claude.ai/code/artifact/68e448d0-462c-4ca0-a1a2-091696398aad
   - Recovered text: `recovered/the-engram-loop.extracted.txt`
3. **Waves Over Weights** (review + proposal set, 21 Aug 2026) — a full reading of
   Miller, Brincat & Roy 2026 "Analog Cognition and Consciousness" mapped against the ML
   literature; ten proposals (P1 ephaptic channel … P10 split-substrate hardware) in the
   waves-as-control gap that ML has not entered.
   - Published artifact: https://claude.ai/code/artifact/0ba1bc9d-3e04-41ef-b8ca-df2a34f21b5d
   - Recovered text: `recovered/waves-over-weights.extracted.txt`

The connective thread: REFA Edge is the *hands* of the program — the place where
brain-derived memory mechanisms become small, testable, edge-runnable code with
equivalence gates and honest benchmarks. The Engram Loop is the *program*; Waves Over
Weights is the *control-plane theory* it extends.

## State of the repo (as of this session)

- Branch: `claude/brain-research-machine-mapping-j2lbdc` (development branch).
- Core model `src/refa_edge/models/refa.py`: implemented, tested, Python per-step loop
  (clarity over speed). Dense/stream fast-weight equivalence gate passes in float64.
- Known design facts a future session should not rediscover:
  - Triple binding is **additive** (entity/relation embeddings summed with two learned
    role offsets, LayerNormed) — the weakest binding form; the dataset's close-decoy
    unknowns specifically stress it.
  - Keys = queries = normalized projection of the identity embedding (R=16 default);
    capacity per bank is bounded by key crosstalk in R dimensions.
  - The revision write erases key-matched content from BOTH stance banks, then writes
    only to the event's stance bank. Along the key direction the update factor is
    (λ − β·route) ∈ (λ−1, λ), so the recurrence is contractive for β, route ∈ (0,1).
  - Recency exists only through decay λ and erase-rewrite; there is no queryable time code.
  - The classifier reads the workspace plus the *final-step* support/opposition recalls
    (the query is always the last row in the dataset).
  - "Unknown" is a learned class with no calibration story yet.

## This session (2026-08-23): the ultra sweep

An ultracode workflow ran 13 parallel domain finders + adversarial verification of every
load-bearing claim + a completeness critic + follow-up finders + a three-lens connective
leap panel, covering: delta-rule parallelization math, associative capacity theory,
evidential uncertainty algebra, temporal codes, cognitive-map/binding math,
neuromodulatory gating, local learning, consolidation math, mid-2025→2026
brain↔machine mapping, memory substrates, mathematical frameworks audit, and two scouts
hunting areas absent from the existing ledgers.

Outputs of this session (read in this order):

1. `SWEEP-2026-08-23.md` — verified findings by area (the new evidence base).
2. `CONNECTIVE-LEAPS.md` — ranked novel design moves for REFA Edge, with math,
   novelty checks, first experiments, and kill criteria.
3. Updated sections of this handoff (below).

## Next actions queue

(Maintained across sessions — strike items when done, add provenance.)

1. Implement the top-ranked leaps from `CONNECTIVE-LEAPS.md` behind the repo's
   equivalence-gate discipline (each new mechanism needs a dense reference + streamed
   form + float64 output/gradient test before benchmarking).
2. Chunk-parallelize the REFA time loop (see SWEEP: delta-rule/WY findings) and record
   the wall-clock delta on the RTX 2060 profile.
3. Add the evidential head + conformal unknown wrapper (see SWEEP: evidential-math) and
   report risk–coverage curves next to accuracy.
4. Attach Laplace timestamps to events (SWEEP: temporal-math) and add the
   delayed-evidence / temporal-displacement task to the benchmark suite.
5. Phase 0 of the Engram Loop proposal (Confabulation Battery) can be prototyped against
   REFA itself: gist intrusion ≈ close-decoy errors; temporal displacement ≈ revision
   failures; calibrated abstention ≈ unknown-class type-2 ROC.

## Session log

- **2026-08-21** — Waves Over Weights review written and published (artifact above).
- **2026-08-22** — The Engram Loop proposal written on 16 verified literature sweeps;
  audited same day; DeepSeek-Engram positioning added post-audit.
- **2026-08-23** — Handoff directory recovered into the repo. Ultra sweep executed
  (13 finders → verify → gap critic → follow-ups → 3-lens leap panel). Findings and
  leaps committed alongside this file.

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

## This session (2026-08-23): the ultra sweep — COMPLETED

An ultracode workflow ran 13 parallel domain finders (188 findings), adversarial
verification of every load-bearing claim, a completeness critic that surfaced 6 missed
areas and spawned 4 follow-up finders (streaming sketches, rate-distortion memory
theory, truth discovery / source reliability, neural algorithmic reasoning), and a
three-lens panel (mathematics / neuroscience / edge-systems) that produced 20 connective
leaps. ~3.5M research tokens, 943 tool calls, ~2.7 h wall clock.

Outputs (read in this order):

1. `SWEEP-2026-08-23.md` — the full verified evidence base by area, with per-claim
   verification verdicts and three retractions from adversarial checking.
2. `CONNECTIVE-LEAPS.md` — the 20 leaps merged into families and ranked into three
   tiers, plus ten "direct adoptions" the sweeps mandate regardless of novelty.
3. This handoff's next-actions queue (below).

The one-sentence discovery that reorganizes the project: **REFA's update is exactly a
constant-gate gated delta rule**, term for term — which makes the 2024–26
DeltaNet/Gated-DeltaNet WY chunkwise algebra an *exact* (not approximate) rewrite of the
Python loop, imports sharp stability/expressivity theory (current gates confine
transition eigenvalues to ≈[0,1), provably blocking parity-class revision tracking —
fixable free), and positions every upgrade inside the test-time-regression framework.
Second organizing discovery: the (support, opposition, unknown) triple has an exact
existing algebra — subjective-logic opinions / Beta evidence — and the honest way to get
it is to *measure* evidence counts from an erase-consistent write Gram rather than train
a 3-logit head (which 2022–24 theory proves cannot be faithful). Third: certified
streaming sketches (Bloom / conservative count-min / IBLT) give REFA a deterministic,
untrained "unknown" code path with zero false negatives.

## Next actions queue

(Maintained across sessions — strike items when done, add provenance.)

**Priority override after the first kill experiment:** A2/K1 typed banks versus a
signed single store is the immediate next state. The older list below was written before
the kill-experiment pack and is not execution order. Finish K1, then K2 and K4, before
substrate optimization. The frozen K1 design constraints and kill rule are in
`docs/NEXT-RESEARCH-STATE.md`.

1. **Kernel deferred until the founding bet has data** (CONNECTIVE-LEAPS #1):
   stance-folded chunkwise WY kernel for the 16 (bank, stance) recurrences,
   float64-gated against the Python loop; then CUDA-graph capture. Expected 10–50× wall
   clock, ~30× activation-memory cut. Unblocks everything.
2. **[DONE 2026-08-23] Spectral write gate + corrected keyed-parity task** (#2): the
   originally proposed alternating-polarity task leaked parity through its final token.
   Replaced with identical SUPPORT writes to two interleaved keys and a query revealed
   last. Default expressive gating was bimodal (5/10 primary seeds); a fresh-seed,
   preregistered spectrum-derived bias initialization passed (memory-only OOD median
   0.893, full REFA 0.930, controls ~0.50). All spectral certificates passed. Post-hoc,
   23/23 expressive runs with minimum eigenvalue <= -0.99 solved versus 1/17 farther
   away: reflection magnitude, not sign alone, is the next hypothesis. See
   `docs/A1-KEYED-PARITY-RESULTS.md` and tracked raw JSON under `results/`.
3. **Provenance audit script** (#9): ~80 lines, zero model changes — replay episodes,
   verify the exact per-event responsibility decomposition, plot superseded-coefficient
   collapse. Validates the WY algebra before the kernel lands.
4. **Measured evidential stack** (#3–#6): write-Gram evidence counts → closed-form
   (b,d,u); ridge-coverage sidecar; Bloom/CMS/IBLT certified-unknown sidecar; ACI
   conformal wrapper. Report risk–coverage + smECE against the 3-logit baseline.
5. **Timestamp columns** (#7): K=6 log-spaced decay value-columns (erase resets age);
   recency-ordering probe task; complex-tap extension (#8) if magnitude saturates.
6. **muP-for-banks** (#10): width map + coordinate-check gate; move all tuning to
   30k-param proxies.
7. Tier 2 by task priority: Plateau Commit (delayed evidence), Silence-Is-Unknown
   (typed abstention), forward-decay + int8 cold-bank residency (offload), ACh/NE dial +
   streaming Dawid–Skene (provenance conflict), Clone-or-Revise (regime return).
8. Direct adoptions checklist (CONNECTIVE-LEAPS "Direct adoptions"): state-passing
   post-training, multi-seed medians+IQR, smECE, crosstalk instrumentation, DMFT GRU
   init, rotation-bound keys, load-balanced top-k routing, new battery tasks, ACT-R
   baseline.
9. Open reading from the critic (no follow-up run yet): region-based (box/cone)
   embeddings for facts-as-volumes; schema-congruency-gated consolidation
   (Tse / van Kesteren SLIMM).
10. Confabulation Battery Phase 0 prototype against REFA itself: gist intrusion ≈
    close-decoy errors; temporal displacement ≈ revision failures; calibrated abstention
    ≈ unknown-class type-2 ROC — now with the measured-evidence head as the instrument.

## Handing this program to another LLM

A complete, self-contained briefing pack lives at the repo root: **`llm-handoff/`** —
mission, exact state, research index, the standing critical assessment (with four kill
experiments), frontier comparison, compute levers, push directions with
creative-connection seeds, an ordered task queue, and a paste-ready boot prompt
(`llm-handoff/BOOT-PROMPT.md`). Any successor session — human-driven or another model —
should start there.

## Session log

- **2026-08-21** — Waves Over Weights review written and published (artifact above).
- **2026-08-22** — The Engram Loop proposal written on 16 verified literature sweeps;
  audited same day; DeepSeek-Engram positioning added post-audit.
- **2026-08-23** — Handoff directory recovered into the repo. Ultra sweep executed
  (13 finders → verify → gap critic → 4 follow-ups → 3-lens leap panel; 38 agents,
  188 findings, 20 leaps). `SWEEP-2026-08-23.md` and `CONNECTIVE-LEAPS.md` committed
  alongside this file; three claims retracted under adversarial verification; two
  filter-blocked verification passes re-run out-of-band (verdicts appended to the sweep
  document's addendum). Synthesis published as the third companion artifact.
- **2026-08-23** — First kill experiment executed. Rejected the queued K3 task for
  label leakage; implemented a leak-free keyed-parity generator, current/safe/expressive
  gate modes, runtime spectral certificate, memory-only diagnostic readout, and a
  reproducible 10-seed harness. Default A1 was inconclusive because expressive training
  split 5/10 between reflection and positive-spectrum basins. A separately preregistered
  shared bias=2 follow-up passed: memory-only length-64 median 0.893 (IQR 0.852–0.943),
  full REFA 0.930 (0.921–0.935), controls ~0.50. 120 confirmatory/follow-up runs, raw
  JSON, combined CSV/summary, report, and plot preserved. Corrected an interim verbal
  misstatement that equated two accuracy failures with loss of negative sign; both kept
  negative eigenvalues, showing sign is necessary here but not sufficient away from -1.

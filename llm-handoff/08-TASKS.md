# 08 — Task queue (ordered, with acceptance criteria)

Conventions: every experiment ≥10 seeds, report medians+IQR; calibration via smECE;
every mechanism behind an exact or certified gate; raw JSON + config + seed + hardware
+ commit hash preserved. Hardware assumption: one RTX 2060 6 GB (or better).

**Immediate next state (after A1): A2/K1.** Build the fair signed-store control and
freeze the revision/conflict/abstention battery before confirmatory seeds. Exact design
constraints and the kill rule: `../docs/NEXT-RESEARCH-STATE.md`. Substrate optimization
is deferred until this founding architectural bet has data.

## Lane A — Kill experiments (first; nothing else has standing until these run)

- [x] **A1 (K3): corrected keyed parity.** The queued alternating-polarity task leaked
      its label through the final event, so the executed task uses identical SUPPORT
      writes, two interleaved keys, and a query revealed last (train n=16, test n=64).
      Default expressive gating was bimodal (5/10 primary seeds); a separately
      preregistered bias=2 follow-up passed (memory-only median 0.893, full REFA 0.930,
      controls ~0.50; all contraction certificates passed). Result and raw artifacts:
      `../docs/A1-KEYED-PARITY-RESULTS.md`, `../results/`. Qualification: sign alone is
      insufficient; near-reflection magnitude predicted retention post hoc.
- [ ] **A2 (K1, the founding bet): typed banks vs signed single store.** Build a
      parameter-matched REFA variant with ONE bank set storing signed values
      (polarity → ±v) and compare on the full battery. Accept (bet survives): typed
      wins on revision/conflict/abstention beyond IQR overlap. If it dies: write it up,
      pivot per `04-CRITICAL-ASSESSMENT.md`.
- [ ] **A3 (K2): measured vs learned evidential head.** Implement the write-Gram
      recursion + closed-form (b,d,u) + ridge coverage (leaps 3–4; ~40 lines) and
      compare risk-coverage/smECE vs the 3-logit head, in-distribution and at 4×
      length. Accept: measured head ≥ accuracy, better OOD calibration.
- [ ] **A4 (K4): toy text boundary.** Paraphrase generator over (s,r,o) templates + a
      trivial canonicalizer; measure certified-coverage degradation curve. Accept:
      a nonzero certified subset survives; curve documented either way.

## Lane B — Substrate (parallel with A after A1)

- [ ] **B1: chunkwise WY kernel** (leap 1): forward float64 single-head vs loop ≤1e-9;
      then full stance-folded version; then CUDA-graph capture. Accept: exact match +
      ≥10× wall clock at T=512 + new pytest in the equivalence suite.
- [ ] **B2: provenance audit script** (leap 9): replay episodes, assert
      Σ aₑ(q)·vₑ = actual read to <1e-10; plot superseded-coefficient collapse.
      Zero model changes. Accept: identity holds; plot in `runs/`.
- [ ] **B3: state-passing post-training + crosstalk instrumentation** (direct
      adoptions 1 & 5). Accept: length-extrapolation delta reported; μ and SNR logged
      per run in results JSON.
- [ ] **B4: timestamp columns + recency probe** (leap 7). Accept: ordering probe >90%
      within a tap octave vs baseline ~chance.
- [ ] **B5: μP width map + coordinate check** (leap 10). Accept: Θ(1) scaling slopes
      within ±0.1 for w ∈ {1,2,4,8}; one confirmed hyperparameter transfer w=1→w=4.

## Lane C — Instruments (independent; highest value per GPU-hour)

- [ ] **C1: Confabulation Battery v0** as a standalone harness (gist intrusion, source
      attribution, temporal displacement, elapsed-time Weber, conflict arbitration,
      type-2 calibrated abstention) runnable against (a) REFA, (b) any HF model via an
      adapter. Accept: per-organ radar profile JSON + one public model profiled.
- [ ] **C2: certified sketch sidecar** (leaps 5–6) with its three exact property tests.
      Accept: property tests pass; certified-unknown path evaluated in the battery.

## Lane D — Research sweeps (background)

- [ ] **D1: decentralized training/inference sweep** (see `07-DIRECTIONS.md` push 2) —
      with the federated-evidential-memory question as its leap target.
- [ ] **D2: box/cone embeddings + schema-gated consolidation sweeps** (critic leftovers).
- [ ] **D3: leap re-triage** with fresh eyes; update CONNECTIVE-LEAPS tiers with reasons.

## Lane E — Program outputs

- [ ] **E1:** Write up whichever of A1–A4 lands first (positive or negative) as a short
      pre-registered report in `docs/`, with raw results committed.
- [ ] **E2:** Package the battery (C1) for release: README, license, model-adapter doc,
      one-command run.
- [ ] **E3:** Keep `../engram-loop-handoff/HANDOFF.md` session log current; append
      sweep results to the corpus with verification; update this pack where state changes.

## Definition of "the program advanced"

At least one of: a kill experiment produced data; a gate-passing mechanism merged; the
battery profiled a new model; a sweep added verified findings + a novel leap; a wrong
claim got retracted. Everything else is motion, not progress.

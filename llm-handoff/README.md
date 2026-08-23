# LLM handoff pack — REFA Edge / Engram Loop program

You are (probably) a language model that has just been handed this folder. It is a
complete, self-contained briefing for taking over an open research program. A human
(Reza Almiro) owns the direction; you are expected to work with a high degree of
autonomy, challenge everything in here, and leave the pack better than you found it.

## What this program is, in one paragraph

An open, edge-runnable research program on **memory-first architectures**: models whose
experience lives in explicit, typed, revisable evidence memory rather than only in
opaque weights — with uncertainty *measured* from the memory's own accounting (not
learned), provenance *computed exactly* (not approximated post-hoc), and every mechanism
gated by exact equivalence tests. The concrete testbed is REFA Edge (this repository): a
tiny support/opposition fast-weight model that a 2019 laptop GPU can train. The larger
aim is stated in `01-MISSION.md`: cut the compute limitations that keep capable AI in
the hands of a few labs, and keep everything open.

## Reading order

| # | File | What it gives you | Read when |
|---|------|-------------------|-----------|
| 0 | this file | orientation + rules of engagement | always, first |
| 1 | `01-MISSION.md` | aims, values, what we will and will not claim | always |
| 2 | `02-STATE.md` | exact current state: code facts, discoveries, what is NOT done | always |
| 3 | `03-RESEARCH-INDEX.md` | map of all 188 findings + 20 leaps, with pointers | always |
| 4 | `04-CRITICAL-ASSESSMENT.md` | the brutal self-critique; standing challenges | always |
| 5 | `05-FRONTIER-AXIS.md` | where frontier labs are; the axis where open/small wins | always |
| 6 | `06-COMPUTE-LEVERS.md` | every credible compute-cutting lever, honestly rated | when working on efficiency |
| 7 | `07-DIRECTIONS.md` | what to push next; creative-connection seeds; sweep methodology | when choosing work |
| 8 | `08-TASKS.md` | ordered task queue with acceptance criteria | when executing |
| — | `BOOT-PROMPT.md` | a paste-ready prompt to boot a fresh LLM into this pack | when re-handing off |

Deep evidence (large files, load sections on demand, not whole):

- `../engram-loop-handoff/SWEEP-2026-08-23.md` — 188 findings with per-claim
  adversarial-verification verdicts, sources, and retractions (~170k tokens; use the
  index in `03-RESEARCH-INDEX.md` to load only the areas you need).
- `../engram-loop-handoff/CONNECTIVE-LEAPS.md` — all 20 design leaps in full
  (mechanism, math, novelty audit, first experiment, risk) (~26k tokens).
- `../engram-loop-handoff/HANDOFF.md` — the running project log and next-actions queue.
- `../engram-loop-handoff/recovered/` — text of the two companion documents:
  *The Engram Loop* (architecture program) and *Waves Over Weights* (waves-as-control review).
- `../src/refa_edge/` — the code. Start with `models/refa.py` (~125 lines) and
  `data.py`; the whole core fits in a few thousand tokens.

## Rules of engagement

1. **Nothing in this pack is gospel.** The previous agent's synthesis, rankings, and
   leaps are hypotheses. Your first duty on any claim you build on is to try to kill it.
2. **Verification is part of the work, not an afterthought.** Every load-bearing claim
   you add must cite a source you actually located; characterize it accurately; retract
   loudly when wrong. This program has already retracted three claims — that is the
   culture, keep it.
3. **Exactness discipline.** Any new mechanism enters behind the repo's gate style: a
   reference implementation, then an exact (float64) or certified (one-sided/property)
   test, then benchmarks. Medians + IQR over ≥10 seeds; smECE, never binned ECE.
4. **Honest failure is a deliverable.** A clean negative result on one of our own bets
   is worth as much as a positive one. Do not rescue a dying hypothesis with task design.
5. **Open by default.** MIT license; results ship with config, seed, raw JSON, hardware,
   and commit hash (see repo README). Nothing in this program depends on privileged
   compute, private data, or closed weights — keep it that way.
6. **Update the ledger.** When you finish a work session, update
   `../engram-loop-handoff/HANDOFF.md` (session log + queue) and, if you ran research
   sweeps, append to the sweep corpus with the same verification standard.

# 03 — Research index

Map of the full evidence base in `../engram-loop-handoff/SWEEP-2026-08-23.md` (load
areas on demand — the file is ~170k tokens) and the 20 leaps in
`../engram-loop-handoff/CONNECTIVE-LEAPS.md`. ⚑ = load-bearing (adversarially checked).

## The 17 sweep areas

| Area (heading in SWEEP file) | One-line takeaway | Verification |
|---|---|---|
| `fastweight-math` | REFA = gated delta rule; WY chunkwise is exact; eigenvalue theory; TTR upgrade ladder (Longhorn/RLS/Titans/DeltaProduct); 2026 erase-decoupling cluster | 5/5 CONFIRMED |
| `capacity-math` | Crosstalk law SNR=√(R/(T−1)); Welch floor; Hopfield ladder; fly-hash expand-and-sparsify; MoM hard routing; Sparse Delta Memory 1000× state | 5/5 CONFIRMED |
| `evidential-math` | Subjective logic ↔ Beta bijection; Denoeux ignorance-vs-conflict; Bengs impossibility; conformal/ACI; Chow; Aug-2026 abstention-collapse law; smECE | 5/5 CONFIRMED |
| `temporal-math` | Laplace/SITH taps + Post inversion; HiPPO measures; LinOSS; timestamped fast weights = genuine gap; the T-trace design | 5/5 CONFIRMED |
| `relational-structure` | Bind multiplicatively (TEM, VSA, RotatE=FHRR); HRR capacity ≈ D/(2 ln K); resonators; barcodes/Vector-HaSH: keys partly fixed; keep matrix banks | 2 CONFIRMED, 3 PLAUSIBLE (math independently recomputed, held) |
| `gating-neuromod` | Three-factor rules; D1/D2 opponency mirrors stance split; Yu-Dayan ACh/NE; PBWM workspace gating; Backpropamine; e-prop; astrocytes | 5/5 CONFIRMED |
| `local-learning` | RTRL exact & O(1) for linear states; e-prop for the GRU; RLS heads; conceptor algebra; echo-state property holds by construction; PC/EqProp buy little here | 4 CONFIRMED, 1 PLAUSIBLE |
| `consolidation-math` | EWC/SI; null-space (AlphaEdit exact for KV layers); sparse memory finetuning (2510.15103); Benna-Fusi; SHY; gate-before-commit; provenance-preserving distillation = gap | 4 CONFIRMED, 2 REFUTED (retracted), 2 PLAUSIBLE |
| `brainmap-2026` | BTSP one-shot retrospective writes (Wu-Maass CAM, gBTSP); MICrONS like-to-like; fly connectome behaviors; POYO; PRH→human; inner-speech BCIs; engram allocation | 5/5 CONFIRMED (addendum; 2 attribution corrections) |
| `substrate-2026` | BDH state-as-synapses; DeepSeek Engram O(1) tables; product keys to 128B; PEER/UltraMem; FwPKM (product keys × fast weights); MoD; BitNet; SpikingBrain ~2% tokens; MLA | 4 CONFIRMED, 1 PLAUSIBLE |
| `math-frontiers` | Adopt: contraction/spectral analysis, TTR, frame theory, DMFT init, sheaf conflict score (later), EFE epistemic term. Skip: K-FAC, hyperbolic, OT, TDA at this scale | 4 CONFIRMED, 1 PLAUSIBLE |
| `neuro-scout` | Mushroom-body valence circuits (literal support/opposition memory, half-ported); inhibitory engrams; ACT-R calculus; latent-cause CRP gate; retrieval-induced forgetting | 5/5 CONFIRMED (addendum; 2 corrections: capacity ≈10^171 not 10^140; Ramaswami not Dolan) |
| `ml-scout` | State-tracking expressivity; μP; EDL; semiring provenance (Scallop); AbstentionBench (abstention doesn't scale); SM 7.5 engineering facts; kNN-LM provenance | 5/5 CONFIRMED |
| `followup: streaming sketches` | Banks ARE decayed linear sketches; one-sided certificates (Bloom FN=0, CU-CMS, MG, FD); IBLT audit; mergeability theory: the erase term blocks merges; forward decay fixes it | 4 CONFIRMED, 1 PLAUSIBLE |
| `followup: rate-distortion` | Memory = optimal lossy compression; DRM intrusions are the RD optimum; per-bank bit budgets; routing temperature = Lagrange multiplier; derivable battery curves | 5 CONFIRMED, 1 REFUTED (Fisher bound corrected: Ganguli-Huh-Sompolinsky, ≤N for arbitrary nets) |
| `followup: truth discovery` | Dawid-Skene EM; identifiability (≥3 independent sources); minimax spectral+EM; streaming variants at O(scalars/source); four-cause "unknown" = building site | 5/5 CONFIRMED |
| `followup: neural algorithmic reasoning` | Algorithmic alignment; CLRS hints; RASP-L; Chomsky-hierarchy dissociation; unexplored-states fix (~0.1% budget); length-gen is high-variance → medians+IQR | 5/5 CONFIRMED |

## The 20 leaps (full text in CONNECTIVE-LEAPS.md)

**Tier 1 (adopt now):** 1 stance-folded WY kernel · 2 dual-mode spectral gate + parity
task · 3 Gram-ridge evidential head · 4 ridge-coverage sidecar · 5 certified sketch
sidecar (Bloom/CMS/IBLT) · 6 interval opinions (imprecise Dirichlet over sketch bounds) ·
7 timestamp columns · 8 complex-decay taps · 9 WY-exact provenance ledger.

**Tier 2:** 10 μP-for-banks · 11 forward-decay reparameterization + int8 cold banks ·
12 Plateau Commit (BTSP) · 13 Silence-Is-Unknown (inhibitory-engram balance) ·
14 ACh/NE dial + streaming Dawid-Skene · 15 Clone-or-Revise (latent-cause gate + phase clones).

**Tier 3 (research bets):** 16 stance-rotation supersession (invertible revision,
conservation law) · 17 barcode ledger · 18 excitability tags (memory linking) ·
19 query sculpting (RIF) · 20 reverse-water-filling decay spectrum.

Post-panel novelty spot-checks (live searches) are recorded at the top of
CONNECTIVE-LEAPS.md; nearest adjacencies: ELA (arXiv:2605.18848), sin/cos-timestamp
gating in temporal recommenders, GDN-2 (arXiv:2605.22791).

## Retractions (do not cite)

arXiv 2604.05248, 2605.03229, 2605.09315, 2608.05810 — unlocatable; all their numbers
void. Fisher-memory bound corrected. Full notes at the top of the SWEEP file.

## Ten direct adoptions (no novelty claimed, mandatory hygiene)

State-passing post-training · medians+IQR over ≥10 seeds · smECE · ACI conformal
wrapper (never train "unknown" as an action) · crosstalk instrumentation · DMFT
marginal-stability GRU init · rotation-bound keys · MoM load balancing for top-k ·
theory-implied battery tasks (parity, recency, regime-return, delayed evidence, fan) ·
ACT-R as the classical baseline.

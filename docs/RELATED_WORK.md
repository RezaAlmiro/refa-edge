# Relationship to Pathway BDH and Soup

REFA Edge is not a fork of either project. This note records which lessons are being
tested, which are merely inspirations, and where the mechanisms differ.

## Pathway BDH

Primary sources:

- [The Dragon Hatchling: The Missing Link between the Transformer and Models of the
  Brain](https://arxiv.org/abs/2509.26507), Adrian Kosowski, Przemysław Uznański,
  Jan Chorowski, Zuzanna Stamirowska, and Michał Bartoszkiewicz (2025)
- [pathwaycom/bdh](https://github.com/pathwaycom/bdh)

The paper presents BDH as local graph dynamics and BDH-GPU as a tensor-friendly state
space model combining ReLU-lowrank blocks with linear attention. It reports
Transformer-like scaling at matched data and parameter counts, sparse positive
activations, modular and heavy-tailed interaction structure, and interpretable
synaptic state. Those are the authors' reported findings; REFA Edge has not
independently reproduced their large-scale experiments.

The public BDH work is important here for three reasons:

1. it treats recurrent state and local update structure as first-class alternatives
   to repeatedly materializing a full attention map;
2. it makes small, auditable implementations central to architectural research;
3. its bilinear interactions motivate asking when a dense expression has an exact
   streamed fast-state form.

REFA Edge includes a deliberately narrow comparator:

$$
y_t = \sum_{s<t}(q_t^\top k_s)v_s
= q_t^\top \left(\sum_{s<t}k_s v_s^\top\right)
$$

The repository computes the left side as a dense reference and the right side as a
streamed recurrence. Unit tests compare outputs and gradients in float64.

This proves equivalence for the stated comparator only. It does **not** prove that
REFA is equivalent to the complete official BDH architecture, and the baseline is
labeled “BDH-style” rather than official.

REFA then extends the fast state in a different direction:

- a learned route divides state into relation banks;
- support and opposition occupy different states;
- a key-addressed delta revises earlier matching evidence;
- a recurrent workspace constructs a temporary hypothesis from both sides.

The findings map to testable REFA decisions:

| BDH finding or property | REFA treatment |
|---|---|
| Working memory as dynamically updated interaction state | Fast evidence banks update during the sequence |
| Excitatory and inhibitory circuits | Computationally distinct support/opposition banks; no biological equivalence claimed |
| Emergent modular interaction graph | Relation banks impose a small initial modular prior |
| Sparse positive activation | Not enforced in v0.1; planned positive/signed-state ablation |
| Interpretability of state | Bank routes and contrastive reads are inspectable; audit tooling is planned |
| Generalization across reasoning time | Required future length-extrapolation benchmark |
| Transformer-like language scaling | Not claimed by this sub-million-parameter categorical MVP |

## Soup

Public source: [MakazhanAlpamys/Soup](https://github.com/MakazhanAlpamys/Soup)

Soup demonstrates that memory-efficient training claims must identify exactly which
objects reside on the GPU and when. Its public approach streams layers of a frozen,
quantized base while training small adapters. That is not the same as training an
entire new architecture from scratch.

The following systems lessons transfer to REFA Edge:

| Lesson | v0.1 treatment |
|---|---|
| Probe actual hardware before a long run | <code>doctor</code> and one-batch <code>fit-check</code> |
| Keep data transfer explicit | DataLoader uses pinned memory when CUDA is available |
| Prevent cross-sample contamination | Every evidence stream is an independent sample |
| Check optimized math against a reference | Dense/stream output and gradient gate |
| Preserve raw measurements | JSON, CSV, config, seed, versions, and hardware record |
| Separate feasibility from capability | README forbids interpreting smoke accuracy as a result |

Mechanisms intentionally not copied:

- streaming a frozen NF4 language model layer by layer;
- LoRA-specific optimizer/state assumptions;
- project-specific buffer and module APIs.

They solve a different storage problem. REFA's core is small enough to remain
resident in VRAM; its growing concern is evidence-bank residency, not frozen
parameter residency.

## Proposed selective relational residency

A future long-memory implementation can borrow the *systems pattern* without copying
the model mechanism:

1. keep frequently routed banks on the GPU;
2. keep inactive banks in pinned system RAM;
3. prefetch predicted next banks into a second buffer;
4. update only the resident top-k banks;
5. log transfer bytes and stalls alongside accuracy;
6. require a dense-resident equivalence gate on short sequences.

This is called **selective relational residency** in the REFA roadmap. It should be
implemented only after the resident reference model establishes a real accuracy or
context-length advantage, because offload complexity cannot rescue a weak learning
rule.

## Clean implementation boundary

No Pathway or Soup source file appears in this repository. References are conceptual
and bibliographic. If an official baseline is added later, it should remain in a
separate adapter with its upstream license, version, and exact commit recorded.

# REFA architecture note

## Research proposition

Many current models compress broad regularities into static parameters and recreate
context in a temporary token window. REFA tests a different division of labor:

1. learned parameters provide compact perceptual and update operators;
2. experience remains explicit as evidence-bearing memory;
3. meaning is produced by relations among observations, including agreement,
   opposition, temporal order, and shared identity;
4. an answer is a temporary latent hypothesis, not another permanent fact by default.

The name **Relational Evidence Field Architecture** refers to the structured field
formed by evidence items and their typed relations.

## Field vocabulary

| Term | Definition |
|---|---|
| Evidence event | An immutable observation with identity fields, stance, and time/order |
| Relation route | A learned distribution over typed memory banks |
| Support bank | State accumulating evidence that affirms a queried relation |
| Opposition bank | Separate state accumulating evidence against it |
| Revision delta | Key-addressed update that replaces matching older content |
| Contrastive recall | Reading support and opposition without averaging them prematurely |
| Hypothesis workspace | Small recurrent state in which recalled evidence becomes an answer |
| Consolidation gate | Proposed test deciding whether a hypothesis deserves durable storage |

No terminology from a separate intellectual tradition is required for the technical
model. These terms are defined operationally and can be measured.

## Implemented core

Let an event embedding be x, a relation route be:

$$
r = \operatorname{softmax}(W_r x / \tau)
$$

For bank h, memory rank R, and value dimension D, REFA keeps two fast states:

$$
M_h^{+}, M_h^{-} \in \mathbb{R}^{R \times D}
$$

The key and query are normalized. Before writing event t, the model reads:

$$
z_t^{s} = \sum_h r_{t,h} q_t^\top M_h^{s},
\qquad s \in \{+, -\}
$$

A direct additive write would let obsolete and current evidence accumulate forever.
The implemented revision delta first removes the value currently addressed by key k,
then writes the new value v only to the event's stance:

$$
M_h^{s} \leftarrow
\lambda_h M_h^{s}
- \beta_t r_{t,h} k_t(k_t^\top M_h^{s})
+ \beta_t r_{t,h}\,\mathbb{1}[s=p_t] k_t v_t^\top
$$

This is a differentiable fast-weight delta rule. It is key-addressed rather than
deleting an entire relation bank. Support and opposition remain separate through
retrieval. Their reads and the direct event representation enter a GRU workspace:

$$
u_t = \operatorname{Fuse}(x_t, z_t^+, z_t^-),
\qquad
h_t = \operatorname{GRUCell}(u_t, h_{t-1})
$$

The final classifier sees the query workspace and both contrastive reads.

## Meaning as relational closure

In this formulation, one observation does not need to carry the whole interpretation.
Suppose a query retrieves one supporting trace and one opposing trace. Their joint
presence defines a structured unresolved state. Temporal order, source reliability,
identity match, and other relations can constrain that state. The workspace computes
a latent closure over those constraints. That closure can have new information even
though no individual memory item contains the final answer.

The current synthetic task tests a minimal case:

- exact identity binding across subject, relation, and object;
- agreement versus opposition;
- later evidence superseding earlier matching evidence;
- unknown when no exact evidence exists.

## Complexity

Let T be sequence length, H banks, R memory rank, and D model width.

| Mechanism | Persistent activation/state memory | Main interaction cost |
|---|---:|---:|
| Dense self-attention | O(T²) attention map | O(T²D) |
| Dense fast-weight reference | O(T²) score map | O(T²R + T²D) |
| Streamed fast weight | O(RD) | O(TRD) |
| REFA typed fast state | O(HRD) | O(THRD) in this reference code |

REFA's state size does not grow with context length. The current Python time loop and
full soft routing prioritize clarity and gradient correctness, not maximum speed.
Sparse top-k routing and fused kernels are later engineering steps.

## What is implemented and what is not

| Capability | v0.1 status |
|---|---|
| Event field embeddings | Implemented |
| Relation-routed memory banks | Implemented, differentiable soft routing |
| Separate support/opposition state | Implemented |
| Key-addressed temporal revision | Implemented |
| Recurrent latent workspace | Implemented |
| Dense/stream exact comparator | Implemented with output and gradient test |
| Provenance graph and source reliability | Planned |
| Persistent memory across independent sessions | Planned |
| Consolidation confidence/calibration gate | Planned |
| Sparse bank residency/offload | Planned |
| Natural-language tokenizer and decoder | Planned |
| Multimodal observations | Planned |
| Fused CUDA/Triton kernel | Planned |

The distinction matters. A repository should not claim that an architectural diagram
is already a trained capability.

## Why this can fit an older GPU

- The included profiles range from tens of thousands to roughly 1.5 million
  parameters, rather than billions.
- Memory state is fixed by H × R × D rather than context squared.
- Automatic mixed precision is enabled in the RTX profile.
- The benchmark uses small categorical events, avoiding a huge language vocabulary.
- Checkpoints and result records are ordinary local files.

Low resource cost is a design constraint, not proof of high capability. The research
must establish whether structure compensates for scale on tasks that matter.

## Research roadmap

1. Validate the revision task across at least five seeds and parameter-matched models.
2. Add long-context length extrapolation where validation streams exceed training.
3. Add provenance conflict, delayed evidence, distractor density, and abstention tasks.
4. Calibrate probabilities and measure unknown detection, not only accuracy.
5. Implement sparse top-k bank routing with dense/stream equivalence tests.
6. Add selective relational residency: inactive banks live in pinned system RAM and
   only selected banks move to VRAM.
7. Build a small text event encoder and retrieval-grounded decoder without discarding
   the explicit evidence protocol.
8. Compare against the official public baselines under their documented settings.

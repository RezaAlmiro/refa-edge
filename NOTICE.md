# Notice and research provenance

REFA Edge is an independent clean implementation written for this repository.

Public work that informed the research direction:

1. Pathway's public Baby Dragon Hatchling repository:
   https://github.com/pathwaycom/bdh
2. Kosowski, Uznański, Chorowski, Stamirowska, and Bartoszkiewicz, "The
   Dragon Hatchling: The Missing Link between the Transformer and Models of the
   Brain," arXiv:2509.26507 (2025):
   https://arxiv.org/abs/2509.26507
3. Soup, a public implementation exploring memory-efficient layer streaming:
   https://github.com/MakazhanAlpamys/Soup

No source code from either repository is included or copied here.

The Pathway work motivates studying recurrent, local-learning, and fast-state
alternatives to conventional attention. This repository's dense/streamed fast-weight
pair is a small independent algebraic comparator and is not the official Pathway
implementation.

Soup motivates systems practices used as research requirements here: measure the real
hardware, make memory behavior explicit, use safe sample boundaries, and require
forward/gradient checks before claiming an optimization preserves a model. REFA Edge
does not reproduce Soup's frozen quantized base-model layer streaming.

Project-specific mechanisms include typed support/opposition banks, soft relation
routing, key-addressed contrastive revision, and the recurrent hypothesis workspace.

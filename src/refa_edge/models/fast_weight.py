from __future__ import annotations

import math

import torch
from torch import nn

from refa_edge.models.common import EventEmbedding


def dense_causal_fast_weight(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
) -> torch.Tensor:
    """Quadratic reference: y_t = sum over s<t of (q_t dot k_s) v_s."""

    seq_len = query.shape[1]
    scores = torch.einsum("btr,bsr->bts", query, key)
    strict_causal = torch.tril(
        torch.ones(seq_len, seq_len, dtype=torch.bool, device=query.device),
        diagonal=-1,
    )
    scores = scores.masked_fill(~strict_causal, 0.0)
    return torch.einsum("bts,bsd->btd", scores, value)


def streaming_causal_fast_weight(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
) -> torch.Tensor:
    """Linear-state exact rewrite of dense_causal_fast_weight."""

    batch_size, seq_len, rank = query.shape
    value_dim = value.shape[-1]
    state = torch.zeros(
        batch_size,
        rank,
        value_dim,
        dtype=query.dtype,
        device=query.device,
    )
    outputs: list[torch.Tensor] = []
    for step in range(seq_len):
        outputs.append(torch.einsum("br,brd->bd", query[:, step], state))
        state = state + torch.einsum("br,bd->brd", key[:, step], value[:, step])
    return torch.stack(outputs, dim=1)


class FastWeightBaseline(nn.Module):
    """A BDH-style dense or streamed fast-weight comparator.

    It is an independent minimal comparator, not Pathway's official BDH implementation.
    Both modes share the same parameters and are mathematically equivalent.
    """

    def __init__(
        self,
        num_entities: int,
        num_relations: int,
        d_model: int = 64,
        memory_rank: int = 16,
        dropout: float = 0.1,
        mode: str = "stream",
        **_: object,
    ) -> None:
        super().__init__()
        if mode not in {"dense", "stream"}:
            raise ValueError("mode must be 'dense' or 'stream'")
        self.mode = mode
        self.scale = 1.0 / math.sqrt(memory_rank)
        self.embedding = EventEmbedding(num_entities, num_relations, d_model, dropout)
        self.query = nn.Linear(d_model, memory_rank, bias=False)
        self.key = nn.Linear(d_model, memory_rank, bias=False)
        self.value = nn.Linear(d_model, d_model, bias=False)
        self.output = nn.Sequential(
            nn.LayerNorm(2 * d_model),
            nn.Linear(2 * d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 3),
        )

    def forward(self, events: torch.Tensor) -> torch.Tensor:
        embedded = self.embedding(events)
        query = self.query(embedded) * self.scale
        key = self.key(embedded)
        value = self.value(embedded)
        if self.mode == "dense":
            recalled = dense_causal_fast_weight(query, key, value)
        else:
            recalled = streaming_causal_fast_weight(query, key, value)
        return self.output(torch.cat([embedded[:, -1], recalled[:, -1]], dim=-1))

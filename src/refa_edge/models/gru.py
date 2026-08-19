from __future__ import annotations

import torch
from torch import nn

from refa_edge.models.common import EventEmbedding


class GRUBaseline(nn.Module):
    def __init__(
        self,
        num_entities: int,
        num_relations: int,
        d_model: int = 64,
        num_layers: int = 1,
        dropout: float = 0.1,
        **_: object,
    ) -> None:
        super().__init__()
        self.embedding = EventEmbedding(num_entities, num_relations, d_model, dropout)
        self.gru = nn.GRU(
            d_model,
            d_model,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.output = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, 3))

    def forward(self, events: torch.Tensor) -> torch.Tensor:
        embedded = self.embedding(events)
        sequence, _ = self.gru(embedded)
        return self.output(sequence[:, -1])

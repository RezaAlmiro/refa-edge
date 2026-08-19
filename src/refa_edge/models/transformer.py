from __future__ import annotations

import torch
from torch import nn

from refa_edge.models.common import EventEmbedding


class TransformerBaseline(nn.Module):
    def __init__(
        self,
        num_entities: int,
        num_relations: int,
        d_model: int = 64,
        num_layers: int = 1,
        num_heads: int = 4,
        dropout: float = 0.1,
        max_seq_len: int = 1024,
        **_: object,
    ) -> None:
        super().__init__()
        self.embedding = EventEmbedding(num_entities, num_relations, d_model, dropout)
        self.position = nn.Embedding(max_seq_len, d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.output = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, 3))

    def forward(self, events: torch.Tensor) -> torch.Tensor:
        seq_len = events.shape[1]
        if seq_len > self.position.num_embeddings:
            raise ValueError(f"Sequence length {seq_len} exceeds configured maximum")
        positions = torch.arange(seq_len, device=events.device)
        embedded = self.embedding(events) + self.position(positions).unsqueeze(0)
        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, device=events.device, dtype=torch.bool),
            diagonal=1,
        )
        encoded = self.encoder(embedded, mask=causal_mask)
        return self.output(encoded[:, -1])

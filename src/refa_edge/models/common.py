from __future__ import annotations

import torch
from torch import nn


class EventEmbedding(nn.Module):
    """Embeds a five-field relational event into one model vector."""

    def __init__(
        self,
        num_entities: int,
        num_relations: int,
        d_model: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.entity = nn.Embedding(num_entities, d_model)
        self.relation = nn.Embedding(num_relations, d_model)
        self.polarity = nn.Embedding(3, d_model)
        self.event_type = nn.Embedding(2, d_model)
        self.role = nn.Parameter(torch.empty(2, d_model))
        self.identity_norm = nn.LayerNorm(d_model)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        nn.init.normal_(self.role, std=0.02)

    def identity(self, events: torch.Tensor) -> torch.Tensor:
        subject = self.entity(events[..., 0]) + self.role[0]
        relation = self.relation(events[..., 1])
        obj = self.entity(events[..., 2]) + self.role[1]
        return self.identity_norm(subject + relation + obj)

    def encode(self, events: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        identity = self.identity(events)
        polarity = self.polarity(events[..., 3])
        event_type = self.event_type(events[..., 4])
        embedded = self.dropout(self.norm(identity + polarity + event_type))
        return identity, embedded

    def forward(self, events: torch.Tensor) -> torch.Tensor:
        return self.encode(events)[1]


def count_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)

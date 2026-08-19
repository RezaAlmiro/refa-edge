"""Minimal example of plugging a different PyTorch model into the harness."""

import torch
from torch import nn

from refa_edge.models.common import EventEmbedding


class MeanPoolComparator(nn.Module):
    def __init__(self, num_entities: int, num_relations: int, d_model: int) -> None:
        super().__init__()
        self.embedding = EventEmbedding(num_entities, num_relations, d_model)
        self.output = nn.Sequential(
            nn.LayerNorm(2 * d_model),
            nn.Linear(2 * d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, 3),
        )

    def forward(self, events: torch.Tensor) -> torch.Tensor:
        embedded = self.embedding(events)
        history = embedded[:, :-1].mean(dim=1)
        query = embedded[:, -1]
        return self.output(torch.cat([history, query], dim=-1))


def build_model(task_config: dict, model_config: dict) -> nn.Module:
    return MeanPoolComparator(
        num_entities=int(task_config["num_entities"]),
        num_relations=int(task_config["num_relations"]),
        d_model=int(model_config["d_model"]),
    )

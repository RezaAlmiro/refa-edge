from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F

from refa_edge.data import OPPOSE, SUPPORT
from refa_edge.models.common import EventEmbedding


class REFAMicro(nn.Module):
    """Typed relational fast memory with key-addressed revision.

    This implements the minimal testable slice of REFA:
    - soft relation routing into a small set of memory banks;
    - separate support and opposition state;
    - a delta-rule write that revises matching evidence;
    - a recurrent workspace that integrates direct and recalled information.
    """

    def __init__(
        self,
        num_entities: int,
        num_relations: int,
        d_model: int = 64,
        memory_rank: int = 16,
        num_banks: int = 8,
        dropout: float = 0.1,
        **_: object,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.memory_rank = memory_rank
        self.num_banks = num_banks
        self.embedding = EventEmbedding(num_entities, num_relations, d_model, dropout)
        self.address = nn.Linear(d_model, memory_rank, bias=False)
        self.value = nn.Linear(d_model, d_model, bias=False)
        self.router = nn.Linear(d_model, num_banks)
        self.write_gate = nn.Linear(d_model, 1)
        self.fusion = nn.Sequential(
            nn.Linear(3 * d_model, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model),
        )
        self.workspace = nn.GRUCell(d_model, d_model)
        self.output = nn.Sequential(
            nn.LayerNorm(3 * d_model),
            nn.Linear(3 * d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 3),
        )
        initial_decay = math.log(0.995 / (1.0 - 0.995))
        self.decay_logits = nn.Parameter(torch.full((num_banks,), initial_decay))
        self.route_temperature = nn.Parameter(torch.tensor(1.0))

    def forward(self, events: torch.Tensor) -> torch.Tensor:
        identity, embedded = self.embedding.encode(events)
        batch_size, seq_len, _ = embedded.shape
        dtype = embedded.dtype
        device = embedded.device
        support_memory = torch.zeros(
            batch_size,
            self.num_banks,
            self.memory_rank,
            self.d_model,
            dtype=dtype,
            device=device,
        )
        oppose_memory = torch.zeros_like(support_memory)
        workspace = torch.zeros(batch_size, self.d_model, dtype=dtype, device=device)
        final_support = torch.zeros_like(workspace)
        final_oppose = torch.zeros_like(workspace)
        decay = torch.sigmoid(self.decay_logits).view(1, self.num_banks, 1, 1)
        temperature = self.route_temperature.clamp(0.25, 4.0)

        for step in range(seq_len):
            x = embedded[:, step]
            identity_step = identity[:, step]
            address = F.normalize(self.address(identity_step), dim=-1)
            q = address
            k = address
            v = torch.tanh(self.value(identity_step))
            route = torch.softmax(self.router(identity_step) / temperature, dim=-1)

            recalled_support = torch.einsum(
                "br,bhrd,bh->bd", q, support_memory, route
            )
            recalled_oppose = torch.einsum(
                "br,bhrd,bh->bd", q, oppose_memory, route
            )
            fused = self.fusion(torch.cat([x, recalled_support, recalled_oppose], dim=-1))
            workspace = self.workspace(fused, workspace)
            final_support = recalled_support
            final_oppose = recalled_oppose

            is_history = (events[:, step, 4] == 0).to(dtype)
            polarity = events[:, step, 3]
            beta = torch.sigmoid(self.write_gate(x)).view(batch_size, 1, 1, 1)
            address_weight = route.view(batch_size, self.num_banks, 1, 1)
            write_strength = beta * address_weight * is_history.view(batch_size, 1, 1, 1)

            predicted_support = torch.einsum("br,bhrd->bhd", k, support_memory)
            predicted_oppose = torch.einsum("br,bhrd->bhd", k, oppose_memory)
            erase_support = torch.einsum("br,bhd->bhrd", k, predicted_support)
            erase_oppose = torch.einsum("br,bhd->bhrd", k, predicted_oppose)
            candidate = torch.einsum("br,bd->brd", k, v).unsqueeze(1)

            support_mask = (polarity == SUPPORT).to(dtype).view(batch_size, 1, 1, 1)
            oppose_mask = (polarity == OPPOSE).to(dtype).view(batch_size, 1, 1, 1)
            support_memory = (
                decay * support_memory
                - write_strength * erase_support
                + write_strength * support_mask * candidate
            )
            oppose_memory = (
                decay * oppose_memory
                - write_strength * erase_oppose
                + write_strength * oppose_mask * candidate
            )

        return self.output(torch.cat([workspace, final_support, final_oppose], dim=-1))

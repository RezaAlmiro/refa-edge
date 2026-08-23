from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F

from refa_edge.data import OPPOSE, SUPPORT
from refa_edge.models.common import EventEmbedding

WRITE_GATE_MODES = ("current", "safe", "expressive")
READOUT_MODES = ("full", "memory_only", "workspace_only")


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
        write_gate_mode: str = "current",
        readout_mode: str = "full",
        track_spectral_diagnostics: bool = False,
        write_gate_bias_init: float | None = None,
        **_: object,
    ) -> None:
        super().__init__()
        if write_gate_mode not in WRITE_GATE_MODES:
            raise ValueError(f"write_gate_mode must be one of {WRITE_GATE_MODES}")
        if readout_mode not in READOUT_MODES:
            raise ValueError(f"readout_mode must be one of {READOUT_MODES}")
        self.d_model = d_model
        self.memory_rank = memory_rank
        self.num_banks = num_banks
        self.write_gate_mode = write_gate_mode
        self.readout_mode = readout_mode
        self.track_spectral_diagnostics = track_spectral_diagnostics
        self.last_spectral_diagnostics: dict[str, float | bool] = {}
        self.embedding = EventEmbedding(num_entities, num_relations, d_model, dropout)
        self.address = nn.Linear(d_model, memory_rank, bias=False)
        self.value = nn.Linear(d_model, d_model, bias=False)
        self.router = nn.Linear(d_model, num_banks)
        self.write_gate = nn.Linear(d_model, 1)
        if write_gate_bias_init is not None:
            nn.init.constant_(self.write_gate.bias, float(write_gate_bias_init))
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
        self.memory_output = nn.Sequential(
            nn.Linear(3 * d_model + 4, d_model),
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
        decay_by_bank = decay.view(1, self.num_banks)
        temperature = self.route_temperature.clamp(0.25, 4.0)
        diagnostic_rates: list[torch.Tensor] = []
        diagnostic_eigenvalues: list[torch.Tensor] = []
        diagnostic_base_gates: list[torch.Tensor] = []
        diagnostic_route_maxima: list[torch.Tensor] = []

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
            gate_logits = self.write_gate(x).view(batch_size, 1)
            if self.write_gate_mode == "current":
                base_gate = torch.sigmoid(gate_logits).expand(-1, self.num_banks)
            elif self.write_gate_mode == "safe":
                positive_gate = F.softplus(gate_logits)
                base_gate = (positive_gate / (1.0 + positive_gate)).expand(
                    -1, self.num_banks
                )
            else:
                base_gate = torch.sigmoid(gate_logits) * (1.0 + decay_by_bank)
            effective_rate = base_gate * route * is_history.view(batch_size, 1)
            write_strength = effective_rate.view(batch_size, self.num_banks, 1, 1)

            if self.track_spectral_diagnostics and bool(is_history.any()):
                history_mask = is_history.bool()
                diagnostic_rates.append(effective_rate[history_mask].detach())
                diagnostic_eigenvalues.append(
                    (decay_by_bank - effective_rate)[history_mask].detach()
                )
                diagnostic_base_gates.append(base_gate[history_mask].detach())
                diagnostic_route_maxima.append(route[history_mask].amax(dim=-1).detach())

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

        if self.track_spectral_diagnostics and diagnostic_rates:
            rates = torch.cat([item.reshape(-1) for item in diagnostic_rates])
            eigenvalues = torch.cat([item.reshape(-1) for item in diagnostic_eigenvalues])
            base_gates = torch.cat([item.reshape(-1) for item in diagnostic_base_gates])
            route_maxima = torch.cat([item.reshape(-1) for item in diagnostic_route_maxima])
            spectral_radius = torch.maximum(
                eigenvalues.abs().amax(), decay_by_bank.detach().amax()
            )
            self.last_spectral_diagnostics = {
                "base_gate_mean": float(base_gates.mean().cpu()),
                "base_gate_max": float(base_gates.max().cpu()),
                "effective_rate_mean": float(rates.mean().cpu()),
                "effective_rate_max": float(rates.max().cpu()),
                "min_transition_eigenvalue": float(eigenvalues.min().cpu()),
                "max_transition_eigenvalue": float(eigenvalues.max().cpu()),
                "negative_eigenvalue_fraction": float((eigenvalues < 0).float().mean().cpu()),
                "spectral_radius": float(spectral_radius.cpu()),
                "route_max_mean": float(route_maxima.mean().cpu()),
                "contraction_certified": bool(float(spectral_radius.cpu()) <= 1.0 + 1e-6),
            }
        else:
            self.last_spectral_diagnostics = {}

        if self.readout_mode == "memory_only":
            value_scale = v.square().sum(dim=-1, keepdim=True).clamp_min(1e-8)
            memory_features = torch.cat(
                [
                    final_support,
                    final_oppose,
                    v,
                    final_support.norm(dim=-1, keepdim=True),
                    final_oppose.norm(dim=-1, keepdim=True),
                    (final_support * v).sum(dim=-1, keepdim=True) / value_scale,
                    (final_oppose * v).sum(dim=-1, keepdim=True) / value_scale,
                ],
                dim=-1,
            )
            return self.memory_output(memory_features)
        if self.readout_mode == "workspace_only":
            final_support = torch.zeros_like(final_support)
            final_oppose = torch.zeros_like(final_oppose)
        return self.output(torch.cat([workspace, final_support, final_oppose], dim=-1))

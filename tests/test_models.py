import pytest
import torch

from refa_edge.models.refa import REFAMicro
from refa_edge.models.registry import BUILTIN_MODELS, build_model


@pytest.mark.parametrize("name", BUILTIN_MODELS)
def test_model_forward_and_backward(name: str) -> None:
    task = {"num_entities": 16, "num_relations": 4, "seq_len": 8}
    model_config = {
        "d_model": 16,
        "memory_rank": 4,
        "num_banks": 3,
        "num_layers": 1,
        "num_heads": 4,
        "dropout": 0.0,
    }
    model = build_model(name, task, model_config)
    events = torch.zeros(3, 8, 5, dtype=torch.long)
    events[..., 0] = torch.randint(0, 16, (3, 8))
    events[..., 1] = torch.randint(0, 4, (3, 8))
    events[..., 2] = torch.randint(0, 16, (3, 8))
    events[..., 3] = torch.randint(0, 2, (3, 8))
    events[:, -1, 3] = 2
    events[:, -1, 4] = 1
    logits = model(events)
    assert logits.shape == (3, 3)
    logits.square().mean().backward()
    assert any(parameter.grad is not None for parameter in model.parameters())


@pytest.mark.parametrize("gate_mode", ["current", "safe", "expressive"])
def test_refa_gate_modes_have_a_runtime_contraction_certificate(gate_mode: str) -> None:
    model = REFAMicro(
        num_entities=8,
        num_relations=2,
        d_model=8,
        memory_rank=2,
        num_banks=1,
        dropout=0.0,
        write_gate_mode=gate_mode,
        readout_mode="memory_only",
        track_spectral_diagnostics=True,
    )
    with torch.no_grad():
        model.write_gate.weight.zero_()
        model.write_gate.bias.fill_(100.0)
    events = torch.zeros(2, 5, 5, dtype=torch.long)
    events[:, -1, 3] = 2
    events[:, -1, 4] = 1
    logits = model(events)
    assert logits.shape == (2, 3)
    diagnostics = model.last_spectral_diagnostics
    assert diagnostics["contraction_certified"] is True
    assert diagnostics["spectral_radius"] <= 1.0 + 1e-6
    if gate_mode == "expressive":
        assert diagnostics["min_transition_eigenvalue"] < -0.99
    else:
        assert diagnostics["min_transition_eigenvalue"] > -0.01


def test_write_gate_bias_can_be_initialized_from_the_spectrum() -> None:
    model = REFAMicro(
        num_entities=8,
        num_relations=2,
        d_model=8,
        memory_rank=2,
        num_banks=1,
        write_gate_bias_init=2.0,
    )
    assert torch.allclose(model.write_gate.bias, torch.tensor([2.0]))

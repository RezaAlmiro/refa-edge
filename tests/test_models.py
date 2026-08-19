import pytest
import torch

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

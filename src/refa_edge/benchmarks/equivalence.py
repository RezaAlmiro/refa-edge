from __future__ import annotations

from typing import Any

import torch

from refa_edge.models.fast_weight import (
    dense_causal_fast_weight,
    streaming_causal_fast_weight,
)


def _run(
    implementation,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
) -> tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
    q = query.clone().detach().requires_grad_(True)
    k = key.clone().detach().requires_grad_(True)
    v = value.clone().detach().requires_grad_(True)
    output = implementation(q, k, v)
    loss = output.square().mean() + output.sum() * 0.01
    gradients = torch.autograd.grad(loss, (q, k, v))
    return output.detach(), tuple(gradient.detach() for gradient in gradients)


def check_dense_stream_equivalence(
    seed: int = 11,
    tolerance: float = 1e-9,
) -> dict[str, Any]:
    generator = torch.Generator().manual_seed(seed)
    query = torch.randn(2, 9, 5, generator=generator, dtype=torch.float64)
    key = torch.randn(2, 9, 5, generator=generator, dtype=torch.float64)
    value = torch.randn(2, 9, 7, generator=generator, dtype=torch.float64)

    dense_output, dense_gradients = _run(dense_causal_fast_weight, query, key, value)
    stream_output, stream_gradients = _run(streaming_causal_fast_weight, query, key, value)
    output_error = float((dense_output - stream_output).abs().max().item())
    gradient_error = max(
        float((dense - stream).abs().max().item())
        for dense, stream in zip(dense_gradients, stream_gradients, strict=True)
    )
    return {
        "passed": output_error <= tolerance and gradient_error <= tolerance,
        "tolerance": tolerance,
        "max_output_absolute_error": output_error,
        "max_gradient_absolute_error": gradient_error,
    }

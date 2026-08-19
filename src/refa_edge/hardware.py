from __future__ import annotations

import os
import platform
from typing import Any

import torch
from torch import nn


def resolve_device(requested: str = "auto") -> torch.device:
    requested = requested.lower()
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA was requested, but PyTorch cannot see an NVIDIA GPU. "
            "Run 'refa-edge doctor' and check your PyTorch installation."
        )
    return torch.device(requested)


def hardware_report() -> dict[str, Any]:
    report: dict[str, Any] = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "logical_cpu_count": os.cpu_count(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "recommended_config": "configs/cpu.yaml",
    }
    if torch.cuda.is_available():
        index = torch.cuda.current_device()
        properties = torch.cuda.get_device_properties(index)
        total_gib = properties.total_memory / (1024**3)
        report.update(
            {
                "cuda_runtime": torch.version.cuda,
                "gpu": properties.name,
                "gpu_memory_gib": round(total_gib, 2),
                "recommended_config": (
                    "configs/rtx2060_6gb.yaml" if total_gib >= 5.5 else "configs/smoke.yaml"
                ),
            }
        )
    return report


def print_hardware_report(report: dict[str, Any]) -> None:
    print("REFA Edge hardware check")
    print("=" * 28)
    print(f"Python:       {report['python']}")
    print(f"PyTorch:      {report['torch']}")
    print(f"CUDA visible: {report['cuda_available']}")
    if report["cuda_available"]:
        print(f"GPU:          {report['gpu']}")
        print(f"GPU memory:   {report['gpu_memory_gib']} GiB")
        print(f"CUDA runtime: {report['cuda_runtime']}")
    print(f"Start with:   {report['recommended_config']}")


def run_fit_check(config: dict[str, Any], model_name: str) -> dict[str, Any]:
    """Run one real forward/backward pass using the configured batch and sequence."""

    from refa_edge.models.common import count_parameters
    from refa_edge.models.registry import build_model

    device = resolve_device(str(config["device"]))
    task = config["task"]
    train = config["train"]
    batch_size = int(train["batch_size"])
    seq_len = int(task["seq_len"])
    model = build_model(model_name, task, config["model"]).to(device).train()
    events = torch.zeros(batch_size, seq_len, 5, dtype=torch.long, device=device)
    events[..., 0] = torch.randint(
        int(task["num_entities"]), (batch_size, seq_len), device=device
    )
    events[..., 1] = torch.randint(
        int(task["num_relations"]), (batch_size, seq_len), device=device
    )
    events[..., 2] = torch.randint(
        int(task["num_entities"]), (batch_size, seq_len), device=device
    )
    events[..., 3] = torch.randint(2, (batch_size, seq_len), device=device)
    events[:, -1, 3] = 2
    events[:, -1, 4] = 1
    labels = torch.randint(3, (batch_size,), device=device)
    use_amp = bool(train["amp"]) and device.type == "cuda"
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
    try:
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
            loss = nn.functional.cross_entropy(model(events), labels)
        loss.backward()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        return {
            "passed": True,
            "model": model_name,
            "device": str(device),
            "batch_size": batch_size,
            "sequence_length": seq_len,
            "parameters": count_parameters(model),
            "amp": use_amp,
            "peak_cuda_allocated_mib": (
                torch.cuda.max_memory_allocated(device) / (1024**2)
                if device.type == "cuda"
                else None
            ),
            "peak_cuda_reserved_mib": (
                torch.cuda.max_memory_reserved(device) / (1024**2)
                if device.type == "cuda"
                else None
            ),
        }
    except (torch.OutOfMemoryError, RuntimeError) as exc:
        is_oom = isinstance(exc, torch.OutOfMemoryError) or "out of memory" in str(exc).lower()
        if not is_oom:
            raise
        return {
            "passed": False,
            "model": model_name,
            "device": str(device),
            "batch_size": batch_size,
            "sequence_length": seq_len,
            "error": "out_of_memory",
            "suggestion": "Halve train.batch_size and run fit-check again.",
        }
    finally:
        del model, events, labels
        if device.type == "cuda":
            torch.cuda.empty_cache()

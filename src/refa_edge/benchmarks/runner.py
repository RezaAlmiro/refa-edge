from __future__ import annotations

import csv
import json
import random
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from refa_edge import __version__
from refa_edge.data import LABEL_NAMES, RelationalRevisionDataset, task_spec_from_config
from refa_edge.hardware import hardware_report, resolve_device
from refa_edge.models.common import count_parameters
from refa_edge.models.registry import BUILTIN_MODELS, build_model, load_external_factory


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _loader(
    dataset: RelationalRevisionDataset,
    batch_size: int,
    shuffle: bool,
    seed: int,
    num_workers: int,
) -> DataLoader:
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        generator=generator,
        pin_memory=torch.cuda.is_available(),
    )


def _evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[float, float, dict[str, float], list[list[int]]]:
    model.eval()
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    total_correct = 0
    total_examples = 0
    class_correct = torch.zeros(3, dtype=torch.long)
    class_total = torch.zeros(3, dtype=torch.long)
    confusion = torch.zeros(3, 3, dtype=torch.long)
    with torch.inference_mode():
        for events, labels in loader:
            events = events.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            logits = model(events)
            total_loss += float(criterion(logits, labels).item()) * labels.shape[0]
            predictions = logits.argmax(dim=-1)
            total_correct += int((predictions == labels).sum().item())
            total_examples += labels.shape[0]
            labels_cpu = labels.cpu()
            predictions_cpu = predictions.cpu()
            for target in range(3):
                target_mask = labels_cpu == target
                class_total[target] += target_mask.sum()
                class_correct[target] += (predictions_cpu[target_mask] == target).sum()
            for target, prediction in zip(labels_cpu, predictions_cpu, strict=True):
                confusion[int(target), int(prediction)] += 1
    per_class = {
        LABEL_NAMES[index]: float(class_correct[index].item() / class_total[index].item())
        for index in range(3)
    }
    return (
        total_loss / total_examples,
        total_correct / total_examples,
        per_class,
        confusion.tolist(),
    )


def _train_one(
    name: str,
    model: nn.Module,
    train_dataset: RelationalRevisionDataset,
    val_dataset: RelationalRevisionDataset,
    config: dict[str, Any],
    device: torch.device,
) -> tuple[dict[str, Any], nn.Module]:
    train_config = config["train"]
    seed = int(config["seed"])
    train_loader = _loader(
        train_dataset,
        int(train_config["batch_size"]),
        True,
        seed,
        int(train_config["num_workers"]),
    )
    val_loader = _loader(
        val_dataset,
        int(train_config["batch_size"]),
        False,
        seed,
        int(train_config["num_workers"]),
    )
    model = model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(train_config["learning_rate"]),
        weight_decay=float(train_config["weight_decay"]),
    )
    criterion = nn.CrossEntropyLoss()
    use_amp = bool(train_config["amp"]) and device.type == "cuda"
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    else:  # pragma: no cover - compatibility with older supported PyTorch builds
        scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)

    start = time.perf_counter()
    final_train_loss = 0.0
    for _epoch in range(int(train_config["epochs"])):
        model.train()
        running_loss = 0.0
        examples = 0
        for events, labels in train_loader:
            events = events.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=use_amp,
            ):
                logits = model(events)
                loss = criterion(logits, labels)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(train_config["grad_clip"]))
            scaler.step(optimizer)
            scaler.update()
            running_loss += float(loss.detach().item()) * labels.shape[0]
            examples += labels.shape[0]
        final_train_loss = running_loss / examples

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    train_seconds = time.perf_counter() - start
    val_loss, val_accuracy, per_class_accuracy, confusion = _evaluate(
        model, val_loader, device
    )
    total_seen = len(train_dataset) * int(train_config["epochs"])
    peak_memory_mib = (
        torch.cuda.max_memory_allocated(device) / (1024**2) if device.type == "cuda" else None
    )
    result = {
        "model": name,
        "parameters": count_parameters(model),
        "train_loss": final_train_loss,
        "validation_loss": val_loss,
        "validation_accuracy": val_accuracy,
        "accuracy_supported": per_class_accuracy["supported"],
        "accuracy_opposed": per_class_accuracy["opposed"],
        "accuracy_unknown": per_class_accuracy["unknown"],
        "confusion_matrix_json": json.dumps(confusion),
        "train_seconds": train_seconds,
        "examples_per_second": total_seen / train_seconds,
        "peak_cuda_memory_mib": peak_memory_mib,
        "epochs": int(train_config["epochs"]),
        "batch_size": int(train_config["batch_size"]),
        "amp": use_amp,
        "seed": seed,
    }
    return result, model


def _write_results(
    output_dir: Path,
    config: dict[str, Any],
    device: torch.device,
    results: list[dict[str, Any]],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 1,
        "refa_edge_version": __version__,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "torch_version": torch.__version__,
        "device": str(device),
        "hardware": hardware_report(),
        "config": config,
        "results": results,
        "claim_status": "local_measurement_only",
    }
    with (output_dir / "results.json").open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    if results:
        with (output_dir / "results.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(results[0].keys()))
            writer.writeheader()
            writer.writerows(results)


def _save_checkpoint(
    output_dir: Path,
    name: str,
    model: nn.Module,
    config: dict[str, Any],
) -> None:
    if name not in BUILTIN_MODELS:
        return
    checkpoint = {
        "format_version": 1,
        "refa_edge_version": __version__,
        "model_name": name,
        "task_config": config["task"],
        "model_config": config["model"],
        "state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()},
    }
    torch.save(checkpoint, output_dir / f"{name}.pt")


def run_benchmark(
    config: dict[str, Any],
    model_names: list[str],
    external_factory: str | None = None,
    external_name: str = "external",
    output_override: str | Path | None = None,
) -> list[dict[str, Any]]:
    if not model_names and external_factory is None:
        raise ValueError("Choose at least one built-in model or an external factory")
    unknown = sorted(set(model_names) - set(BUILTIN_MODELS))
    if unknown:
        raise ValueError(f"Unknown models: {', '.join(unknown)}")

    seed = int(config["seed"])
    set_seed(seed)
    device = resolve_device(str(config["device"]))
    spec = task_spec_from_config(config["task"])
    train_dataset = RelationalRevisionDataset(
        int(config["task"]["train_samples"]), spec, seed=seed
    )
    val_dataset = RelationalRevisionDataset(
        int(config["task"]["val_samples"]), spec, seed=seed + 10_000_019
    )
    output_dir = Path(output_override or config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    factories: list[tuple[str, Any]] = [
        (name, lambda chosen=name: build_model(chosen, config["task"], config["model"]))
        for name in model_names
    ]
    if external_factory is not None:
        external_builder = load_external_factory(external_factory)
        factories.append(
            (
                external_name,
                lambda: external_builder(config["task"], config["model"]),
            )
        )

    results: list[dict[str, Any]] = []
    for index, (name, model_factory) in enumerate(factories, start=1):
        print(f"[{index}/{len(factories)}] Training {name} on {device}...")
        set_seed(seed)
        model = model_factory()
        result, trained_model = _train_one(
            name, model, train_dataset, val_dataset, config, device
        )
        results.append(result)
        _save_checkpoint(output_dir, name, trained_model, config)
        print(
            f"  accuracy={result['validation_accuracy']:.3f} "
            f"params={result['parameters']:,} "
            f"speed={result['examples_per_second']:.1f} examples/s"
        )
        del trained_model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    _write_results(output_dir, config, device, results)
    print(f"Results written to {output_dir.resolve()}")
    return results

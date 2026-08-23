from __future__ import annotations

import hashlib
import json
import random
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from refa_edge import __version__
from refa_edge.benchmarks.runner import set_seed
from refa_edge.data import KeyedParityDataset, keyed_parity_spec_from_config
from refa_edge.hardware import hardware_report, resolve_device
from refa_edge.models.common import count_parameters
from refa_edge.models.refa import REFAMicro


def _git_state() -> dict[str, Any]:
    def run(*args: str) -> bytes:
        completed = subprocess.run(
            ["git", *args],
            check=True,
            capture_output=True,
        )
        return completed.stdout

    try:
        commit = run("rev-parse", "HEAD").decode().strip()
        diff = run("diff", "--binary", "HEAD")
        untracked_paths = run("ls-files", "--others", "--exclude-standard").decode().splitlines()
    except (FileNotFoundError, subprocess.CalledProcessError):
        return {"commit": None, "dirty": None, "working_tree_sha256": None}
    untracked_source = b""
    for relative_path in sorted(untracked_paths):
        if relative_path.startswith("results/"):
            continue
        path = Path(relative_path)
        if path.is_file():
            untracked_source += relative_path.encode() + b"\0" + path.read_bytes() + b"\0"
    digest = hashlib.sha256(diff + b"\0" + untracked_source).hexdigest()
    return {
        "commit": commit,
        "dirty": bool(diff or untracked_source),
        "working_tree_sha256": digest,
    }


def _loader(
    dataset: KeyedParityDataset,
    batch_size: int,
    shuffle: bool,
    seed: int,
    num_workers: int,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        generator=torch.Generator().manual_seed(seed),
        pin_memory=torch.cuda.is_available(),
    )


def _target_counts(events: torch.Tensor) -> torch.Tensor:
    query = events[:, -1:, :3]
    return torch.all(events[:, :-1, :3] == query, dim=-1).sum(dim=-1)


def _merge_diagnostics(
    aggregate: dict[str, Any],
    current: dict[str, float | bool],
    weight: int,
) -> None:
    if not current:
        return
    aggregate["weight"] += weight
    aggregate["min_transition_eigenvalue"] = min(
        aggregate["min_transition_eigenvalue"],
        float(current["min_transition_eigenvalue"]),
    )
    aggregate["max_transition_eigenvalue"] = max(
        aggregate["max_transition_eigenvalue"],
        float(current["max_transition_eigenvalue"]),
    )
    aggregate["spectral_radius"] = max(
        aggregate["spectral_radius"], float(current["spectral_radius"])
    )
    aggregate["contraction_certified"] = bool(
        aggregate["contraction_certified"] and current["contraction_certified"]
    )
    for key in (
        "base_gate_mean",
        "base_gate_max",
        "effective_rate_mean",
        "effective_rate_max",
        "negative_eigenvalue_fraction",
        "route_max_mean",
    ):
        aggregate[key] += float(current[key]) * weight


def _finalize_diagnostics(aggregate: dict[str, Any]) -> dict[str, Any]:
    weight = int(aggregate.pop("weight"))
    if weight == 0:
        return {}
    for key in (
        "base_gate_mean",
        "base_gate_max",
        "effective_rate_mean",
        "effective_rate_max",
        "negative_eigenvalue_fraction",
        "route_max_mean",
    ):
        aggregate[key] /= weight
    return aggregate


def _evaluate(
    model: REFAMicro,
    dataset: KeyedParityDataset,
    batch_size: int,
    device: torch.device,
    train_count_ceiling: int,
    num_workers: int,
) -> dict[str, Any]:
    loader = _loader(dataset, batch_size, False, 0, num_workers)
    criterion = nn.CrossEntropyLoss(reduction="sum")
    total_loss = 0.0
    total_correct = 0
    class_correct = [0, 0]
    class_total = [0, 0]
    count_correct: dict[int, int] = {}
    count_total: dict[int, int] = {}
    unseen_correct = 0
    unseen_total = 0
    diagnostics: dict[str, Any] = {
        "weight": 0,
        "min_transition_eigenvalue": float("inf"),
        "max_transition_eigenvalue": float("-inf"),
        "spectral_radius": 0.0,
        "contraction_certified": True,
        "base_gate_mean": 0.0,
        "base_gate_max": 0.0,
        "effective_rate_mean": 0.0,
        "effective_rate_max": 0.0,
        "negative_eigenvalue_fraction": 0.0,
        "route_max_mean": 0.0,
    }
    model.eval()
    model.track_spectral_diagnostics = True
    with torch.inference_mode():
        for events, labels in loader:
            events = events.to(device)
            labels = labels.to(device)
            logits = model(events)
            predictions = logits[:, :2].argmax(dim=-1)
            total_loss += float(criterion(logits, labels).item())
            correct = predictions == labels
            total_correct += int(correct.sum().item())
            counts = _target_counts(events)
            for label in (0, 1):
                mask = labels == label
                class_total[label] += int(mask.sum().item())
                class_correct[label] += int(correct[mask].sum().item())
            for count in counts.unique().tolist():
                count_int = int(count)
                mask = counts == count_int
                count_total[count_int] = count_total.get(count_int, 0) + int(mask.sum().item())
                count_correct[count_int] = count_correct.get(count_int, 0) + int(
                    correct[mask].sum().item()
                )
            unseen_mask = counts > train_count_ceiling
            unseen_total += int(unseen_mask.sum().item())
            unseen_correct += int(correct[unseen_mask].sum().item())
            _merge_diagnostics(diagnostics, model.last_spectral_diagnostics, labels.shape[0])
    model.track_spectral_diagnostics = False
    return {
        "loss": total_loss / len(dataset),
        "accuracy": total_correct / len(dataset),
        "accuracy_even": class_correct[0] / class_total[0],
        "accuracy_odd": class_correct[1] / class_total[1],
        "unseen_target_count_accuracy": (
            unseen_correct / unseen_total if unseen_total else None
        ),
        "unseen_target_count_examples": unseen_total,
        "accuracy_by_target_count": {
            str(count): count_correct[count] / count_total[count]
            for count in sorted(count_total)
        },
        "spectral_diagnostics": _finalize_diagnostics(diagnostics),
    }


def _train_one(
    config: dict[str, Any],
    seed: int,
    gate_mode: str,
    readout_mode: str,
    device: torch.device,
) -> dict[str, Any]:
    task = config["task"]
    model_config = {
        **config["model"],
        "write_gate_mode": gate_mode,
        "readout_mode": readout_mode,
        "track_spectral_diagnostics": False,
    }
    set_seed(seed)
    train_spec = keyed_parity_spec_from_config(task, int(task["train_history_len"]))
    test_spec = keyed_parity_spec_from_config(task, int(task["test_history_len"]))
    train_dataset = KeyedParityDataset(int(task["train_samples"]), train_spec, seed)
    validation_dataset = KeyedParityDataset(
        int(task["val_samples"]), train_spec, seed + 10_000_019
    )
    test_dataset = KeyedParityDataset(
        int(task["test_samples"]), test_spec, seed + 20_000_033
    )
    train_config = config["train"]
    train_loader = _loader(
        train_dataset,
        int(train_config["batch_size"]),
        True,
        seed,
        int(train_config["num_workers"]),
    )
    model = REFAMicro(
        num_entities=int(task["num_entities"]),
        num_relations=int(task["num_relations"]),
        **model_config,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(train_config["learning_rate"]),
        weight_decay=float(train_config["weight_decay"]),
    )
    criterion = nn.CrossEntropyLoss()
    epoch_history: list[dict[str, float]] = []
    start = time.perf_counter()
    for epoch in range(int(train_config["epochs"])):
        model.train()
        total_loss = 0.0
        total_examples = 0
        gate_gradient_sum = 0.0
        gate_gradient_steps = 0
        for events, labels in train_loader:
            events = events.to(device)
            labels = labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(events)
            loss = criterion(logits, labels)
            loss.backward()
            gate_grad = model.write_gate.weight.grad
            if gate_grad is not None:
                gate_gradient_sum += float(gate_grad.detach().norm().cpu())
                gate_gradient_steps += 1
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), float(train_config["grad_clip"])
            )
            optimizer.step()
            total_loss += float(loss.detach().item()) * labels.shape[0]
            total_examples += labels.shape[0]
        epoch_history.append(
            {
                "epoch": float(epoch + 1),
                "train_loss": total_loss / total_examples,
                "write_gate_gradient_norm": (
                    gate_gradient_sum / gate_gradient_steps if gate_gradient_steps else 0.0
                ),
            }
        )
    train_seconds = time.perf_counter() - start
    train_count_ceiling = int(task["train_history_len"]) - int(task["num_active_keys"]) + 1
    validation = _evaluate(
        model,
        validation_dataset,
        int(train_config["batch_size"]),
        device,
        train_count_ceiling,
        int(train_config["num_workers"]),
    )
    test = _evaluate(
        model,
        test_dataset,
        int(train_config["batch_size"]),
        device,
        train_count_ceiling,
        int(train_config["num_workers"]),
    )
    return {
        "seed": seed,
        "gate_mode": gate_mode,
        "readout_mode": readout_mode,
        "parameters": count_parameters(model),
        "train_seconds": train_seconds,
        "examples_per_second": (
            len(train_dataset) * int(train_config["epochs"]) / train_seconds
        ),
        "epoch_history": epoch_history,
        "validation_id": validation,
        "test_ood_length": test,
        "final_decay": torch.sigmoid(model.decay_logits).detach().cpu().tolist(),
        "final_route_temperature": float(model.route_temperature.detach().cpu()),
    }


def _quantiles(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "median": float(np.median(array)),
        "q25": float(np.quantile(array, 0.25)),
        "q75": float(np.quantile(array, 0.75)),
    }


def summarize_parity_results(results: list[dict[str, Any]]) -> dict[str, Any]:
    groups: dict[str, Any] = {}
    for readout_mode in sorted({str(item["readout_mode"]) for item in results}):
        groups[readout_mode] = {}
        for gate_mode in ("current", "safe", "expressive"):
            selected = [
                item
                for item in results
                if item["readout_mode"] == readout_mode and item["gate_mode"] == gate_mode
            ]
            if not selected:
                continue
            groups[readout_mode][gate_mode] = {
                "runs": len(selected),
                "validation_accuracy": _quantiles(
                    [float(item["validation_id"]["accuracy"]) for item in selected]
                ),
                "ood_length_accuracy": _quantiles(
                    [float(item["test_ood_length"]["accuracy"]) for item in selected]
                ),
                "unseen_target_count_accuracy": _quantiles(
                    [
                        float(item["test_ood_length"]["unseen_target_count_accuracy"])
                        for item in selected
                    ]
                ),
                "min_transition_eigenvalue": _quantiles(
                    [
                        float(
                            item["test_ood_length"]["spectral_diagnostics"][
                                "min_transition_eigenvalue"
                            ]
                        )
                        for item in selected
                    ]
                ),
                "negative_eigenvalue_fraction": _quantiles(
                    [
                        float(
                            item["test_ood_length"]["spectral_diagnostics"][
                                "negative_eigenvalue_fraction"
                            ]
                        )
                        for item in selected
                    ]
                ),
                "train_seconds": _quantiles(
                    [float(item["train_seconds"]) for item in selected]
                ),
            }

    verdict = "not_evaluated"
    rationale = "The preregistered primary memory-only readout is absent."
    primary = groups.get("memory_only", {})
    if all(mode in primary for mode in ("current", "safe", "expressive")):
        expressive = primary["expressive"]
        expressive_ood = expressive["ood_length_accuracy"]["median"]
        best_control = max(
            primary["current"]["ood_length_accuracy"]["median"],
            primary["safe"]["ood_length_accuracy"]["median"],
        )
        min_eigenvalue = expressive["min_transition_eigenvalue"]["median"]
        negative_fraction = expressive["negative_eigenvalue_fraction"]["median"]
        mechanism_active = min_eigenvalue <= -0.5 and negative_fraction >= 0.1
        if not mechanism_active:
            verdict = "inconclusive_mechanism_not_engaged"
            rationale = "Expressive runs did not enter the preregistered negative-spectrum regime."
        elif expressive_ood >= 0.80 and expressive_ood - best_control >= 0.15:
            verdict = "supports_spectral_parity_hypothesis"
            rationale = "Expressive gating met the accuracy and superiority thresholds."
        else:
            verdict = "does_not_support_spectral_parity_hypothesis"
            rationale = "Negative eigenvalues were engaged without the preregistered OOD advantage."
    return {"groups": groups, "preregistered_verdict": verdict, "rationale": rationale}


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    temporary.replace(path)


def run_parity_kill(
    config: dict[str, Any],
    output_override: str | Path | None = None,
) -> dict[str, Any]:
    experiment = config["experiment"]
    seeds = [int(seed) for seed in experiment["seeds"]]
    gate_modes = [str(mode) for mode in experiment["gate_modes"]]
    readout_modes = [str(mode) for mode in experiment["readout_modes"]]
    output_dir = Path(output_override or config["output_dir"])
    device = resolve_device(str(config["device"]))
    payload: dict[str, Any] = {
        "schema_version": 1,
        "experiment": "A1_keyed_parity_spectral_gate",
        "claim_status": "preregistered_local_measurement",
        "refa_edge_version": __version__,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "git": _git_state(),
        "hardware": hardware_report(),
        "device": str(device),
        "config": config,
        "results": [],
    }
    combinations = [
        (readout_mode, gate_mode, seed)
        for readout_mode in readout_modes
        for gate_mode in gate_modes
        for seed in seeds
    ]
    random.Random(9_741).shuffle(combinations)
    for index, (readout_mode, gate_mode, seed) in enumerate(combinations, start=1):
        print(
            f"[{index}/{len(combinations)}] readout={readout_mode} "
            f"gate={gate_mode} seed={seed} device={device}"
        )
        result = _train_one(config, seed, gate_mode, readout_mode, device)
        payload["results"].append(result)
        payload["summary"] = summarize_parity_results(payload["results"])
        _write_json(output_dir / "raw_results.json", payload)
        print(
            f"  id={result['validation_id']['accuracy']:.3f} "
            f"ood={result['test_ood_length']['accuracy']:.3f} "
            f"min_eig={result['test_ood_length']['spectral_diagnostics']['min_transition_eigenvalue']:.3f}"
        )
    payload["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
    payload["summary"] = summarize_parity_results(payload["results"])
    _write_json(output_dir / "raw_results.json", payload)
    _write_json(output_dir / "summary.json", payload["summary"])
    print(f"Results written to {output_dir.resolve()}")
    return payload

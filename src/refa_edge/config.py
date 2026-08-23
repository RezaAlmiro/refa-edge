from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml

DEFAULT_CONFIG: dict[str, Any] = {
    "seed": 7,
    "device": "auto",
    "task": {
        "name": "relational_revision",
        "train_samples": 1000,
        "val_samples": 250,
        "seq_len": 32,
        "num_entities": 64,
        "num_relations": 8,
        "revision_probability": 0.7,
    },
    "model": {
        "d_model": 64,
        "memory_rank": 16,
        "num_banks": 8,
        "num_layers": 1,
        "num_heads": 4,
        "dropout": 0.1,
        "write_gate_mode": "current",
        "readout_mode": "full",
        "track_spectral_diagnostics": False,
        "write_gate_bias_init": None,
    },
    "train": {
        "epochs": 5,
        "batch_size": 32,
        "learning_rate": 8e-4,
        "weight_decay": 0.01,
        "grad_clip": 1.0,
        "amp": True,
        "num_workers": 0,
    },
    "output_dir": "runs/default",
}


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def load_config(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle) or {}
    config = _deep_merge(DEFAULT_CONFIG, loaded)
    validate_config(config)
    return config


def validate_config(config: dict[str, Any]) -> None:
    task = config["task"]
    model = config["model"]
    train = config["train"]
    if task["seq_len"] < 4:
        raise ValueError("task.seq_len must be at least 4")
    if task["num_entities"] < 4 or task["num_relations"] < 2:
        raise ValueError("The task needs at least 4 entities and 2 relations")
    if not 0.0 <= task["revision_probability"] <= 1.0:
        raise ValueError("task.revision_probability must be between 0 and 1")
    if task.get("name") == "keyed_parity":
        train_history_len = int(task["train_history_len"])
        test_history_len = int(task["test_history_len"])
        num_active_keys = int(task["num_active_keys"])
        if train_history_len < num_active_keys + 1:
            raise ValueError("keyed parity needs room for every active key and both parities")
        if test_history_len <= train_history_len:
            raise ValueError("task.test_history_len must exceed task.train_history_len")
    if model["d_model"] % model["num_heads"] != 0:
        raise ValueError("model.d_model must be divisible by model.num_heads")
    if min(model["d_model"], model["memory_rank"], model["num_banks"]) < 1:
        raise ValueError("Model dimensions must be positive")
    if model["write_gate_mode"] not in {"current", "safe", "expressive"}:
        raise ValueError("Unknown model.write_gate_mode")
    if model["readout_mode"] not in {"full", "memory_only", "workspace_only"}:
        raise ValueError("Unknown model.readout_mode")
    if min(train["epochs"], train["batch_size"]) < 1:
        raise ValueError("train.epochs and train.batch_size must be positive")

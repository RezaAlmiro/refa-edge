from __future__ import annotations

import importlib
from collections.abc import Callable

from torch import nn

from refa_edge.models.fast_weight import FastWeightBaseline
from refa_edge.models.gru import GRUBaseline
from refa_edge.models.refa import REFAMicro
from refa_edge.models.transformer import TransformerBaseline

BUILTIN_MODELS = ("refa", "gru", "transformer", "fast_dense", "fast_stream")


def _arguments(task: dict, model: dict) -> dict:
    return {
        **model,
        "num_entities": int(task["num_entities"]),
        "num_relations": int(task["num_relations"]),
        "max_seq_len": int(task["seq_len"]),
    }


def build_model(name: str, task: dict, model: dict) -> nn.Module:
    arguments = _arguments(task, model)
    if name == "refa":
        return REFAMicro(**arguments)
    if name == "gru":
        return GRUBaseline(**arguments)
    if name == "transformer":
        return TransformerBaseline(**arguments)
    if name == "fast_dense":
        return FastWeightBaseline(mode="dense", **arguments)
    if name == "fast_stream":
        return FastWeightBaseline(mode="stream", **arguments)
    raise KeyError(f"Unknown built-in model: {name}. Choose from {', '.join(BUILTIN_MODELS)}")


def load_external_factory(spec: str) -> Callable[[dict, dict], nn.Module]:
    """Load module:function without executing shell commands or evaluating text."""

    if ":" not in spec:
        raise ValueError("External factory must use module:function syntax")
    module_name, function_name = spec.split(":", maxsplit=1)
    module = importlib.import_module(module_name)
    factory = getattr(module, function_name)
    if not callable(factory):
        raise TypeError(f"{spec} is not callable")
    return factory

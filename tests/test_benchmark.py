from pathlib import Path

from refa_edge.benchmarks.runner import run_benchmark


def test_tiny_benchmark_writes_machine_readable_results(tmp_path: Path) -> None:
    config = {
        "seed": 5,
        "device": "cpu",
        "task": {
            "train_samples": 18,
            "val_samples": 9,
            "seq_len": 6,
            "num_entities": 12,
            "num_relations": 3,
            "revision_probability": 0.5,
        },
        "model": {
            "d_model": 12,
            "memory_rank": 3,
            "num_banks": 2,
            "num_layers": 1,
            "num_heads": 3,
            "dropout": 0.0,
        },
        "train": {
            "epochs": 1,
            "batch_size": 6,
            "learning_rate": 0.001,
            "weight_decay": 0.0,
            "grad_clip": 1.0,
            "amp": False,
            "num_workers": 0,
        },
        "output_dir": str(tmp_path),
    }
    results = run_benchmark(
        config,
        ["refa", "gru"],
        external_factory="refa_edge.examples.external_model:build_model",
        external_name="mean_pool",
    )
    assert [result["model"] for result in results] == ["refa", "gru", "mean_pool"]
    assert (tmp_path / "results.json").exists()
    assert (tmp_path / "results.csv").exists()
    assert (tmp_path / "refa.pt").exists()

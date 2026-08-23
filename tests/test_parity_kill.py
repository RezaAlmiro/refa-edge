from pathlib import Path

from refa_edge.benchmarks.parity_kill import run_parity_kill


def test_tiny_parity_kill_run_writes_raw_and_summary_results(tmp_path: Path) -> None:
    config = {
        "seed": 3,
        "device": "cpu",
        "task": {
            "name": "keyed_parity",
            "train_samples": 16,
            "val_samples": 8,
            "test_samples": 8,
            "seq_len": 9,
            "train_history_len": 4,
            "test_history_len": 8,
            "num_entities": 8,
            "num_relations": 2,
            "num_active_keys": 2,
        },
        "model": {
            "d_model": 8,
            "memory_rank": 2,
            "num_banks": 1,
            "dropout": 0.0,
        },
        "train": {
            "epochs": 1,
            "batch_size": 4,
            "learning_rate": 0.001,
            "weight_decay": 0.0,
            "grad_clip": 1.0,
            "num_workers": 0,
        },
        "experiment": {
            "seeds": [3],
            "gate_modes": ["expressive"],
            "readout_modes": ["memory_only"],
        },
        "output_dir": str(tmp_path),
    }
    payload = run_parity_kill(config)
    assert len(payload["results"]) == 1
    assert (tmp_path / "raw_results.json").exists()
    assert (tmp_path / "summary.json").exists()

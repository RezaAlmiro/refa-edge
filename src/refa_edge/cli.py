from __future__ import annotations

import argparse
import json
from pathlib import Path

from refa_edge.benchmarks.equivalence import check_dense_stream_equivalence
from refa_edge.benchmarks.runner import run_benchmark
from refa_edge.config import load_config
from refa_edge.hardware import hardware_report, print_hardware_report, run_fit_check
from refa_edge.models.registry import BUILTIN_MODELS


def _model_list(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="refa-edge",
        description="Train and compare local relational-memory models.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("doctor", help="Check whether PyTorch can see your GPU")

    fit_check = subparsers.add_parser(
        "fit-check",
        help="Try one real training batch before committing to a long run",
    )
    fit_check.add_argument("--config", required=True)
    fit_check.add_argument("--model", choices=BUILTIN_MODELS, default="refa")
    fit_check.add_argument("--device", choices=["auto", "cpu", "cuda"], default=None)

    smoke = subparsers.add_parser("smoke", help="Run the tiny end-to-end sanity check")
    smoke.add_argument(
        "--models",
        type=_model_list,
        default=["refa", "gru", "transformer", "fast_stream"],
        help="Comma-separated model names",
    )
    smoke.add_argument("--device", choices=["auto", "cpu", "cuda"], default=None)
    smoke.add_argument("--output", default=None)

    benchmark = subparsers.add_parser("benchmark", help="Train models under one protocol")
    benchmark.add_argument("--config", required=True)
    benchmark.add_argument(
        "--models",
        type=_model_list,
        default=list(BUILTIN_MODELS),
        help=f"Comma-separated: {','.join(BUILTIN_MODELS)}",
    )
    benchmark.add_argument("--external-factory", default=None)
    benchmark.add_argument("--external-name", default="external")
    benchmark.add_argument("--device", choices=["auto", "cpu", "cuda"], default=None)
    benchmark.add_argument("--output", default=None)

    equivalence = subparsers.add_parser(
        "check-equivalence",
        help="Prove the dense and streaming fast-weight calculations agree",
    )
    equivalence.add_argument("--tolerance", type=float, default=1e-9)

    serve = subparsers.add_parser("serve", help="Start the optional local REST API")
    serve.add_argument("--checkpoint", required=True)
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)
    serve.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.command == "doctor":
        print_hardware_report(hardware_report())
        return
    if args.command == "check-equivalence":
        report = check_dense_stream_equivalence(tolerance=args.tolerance)
        print(json.dumps(report, indent=2))
        if not report["passed"]:
            raise SystemExit(1)
        return
    if args.command == "fit-check":
        config = load_config(args.config)
        if args.device is not None:
            config["device"] = args.device
        report = run_fit_check(config, args.model)
        print(json.dumps(report, indent=2))
        if not report["passed"]:
            raise SystemExit(1)
        return
    if args.command == "serve":
        from refa_edge.api import serve_checkpoint

        serve_checkpoint(args.checkpoint, args.host, args.port, args.device)
        return

    config_path = Path("configs/smoke.yaml") if args.command == "smoke" else Path(args.config)
    config = load_config(config_path)
    if args.device is not None:
        config["device"] = args.device
    external_factory = getattr(args, "external_factory", None)
    external_name = getattr(args, "external_name", "external")
    run_benchmark(
        config,
        args.models,
        external_factory=external_factory,
        external_name=external_name,
        output_override=args.output,
    )


if __name__ == "__main__":
    main()

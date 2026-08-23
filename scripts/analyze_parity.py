from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS = {
    "preregistered": ROOT / "results/a1_keyed_parity/raw_results.json",
    "gate_init_followup": ROOT / "results/a1_gate_init_followup/raw_results.json",
}
OUTPUT_DIR = ROOT / "results/a1_combined_analysis"
SVG_PATH = ROOT / "docs/assets/a1_spectrum_accuracy.svg"


def quantiles(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "median": float(np.median(array)),
        "q25": float(np.quantile(array, 0.25)),
        "q75": float(np.quantile(array, 0.75)),
    }


def load_rows() -> list[dict]:
    rows: list[dict] = []
    for phase, path in EXPERIMENTS.items():
        payload = json.loads(path.read_text(encoding="utf-8"))
        for result in payload["results"]:
            diagnostics = result["test_ood_length"]["spectral_diagnostics"]
            rows.append(
                {
                    "phase": phase,
                    "readout": result["readout_mode"],
                    "gate": result["gate_mode"],
                    "seed": int(result["seed"]),
                    "validation_accuracy": float(result["validation_id"]["accuracy"]),
                    "ood_accuracy": float(result["test_ood_length"]["accuracy"]),
                    "unseen_count_accuracy": float(
                        result["test_ood_length"]["unseen_target_count_accuracy"]
                    ),
                    "min_eigenvalue": float(diagnostics["min_transition_eigenvalue"]),
                    "negative_fraction": float(diagnostics["negative_eigenvalue_fraction"]),
                    "spectral_radius": float(diagnostics["spectral_radius"]),
                    "contraction_certified": bool(diagnostics["contraction_certified"]),
                }
            )
    return rows


def summarize(rows: list[dict]) -> dict:
    grouped: dict[str, dict] = {}
    for phase in EXPERIMENTS:
        grouped[phase] = {}
        for readout in ("memory_only", "full"):
            grouped[phase][readout] = {}
            for gate in ("current", "safe", "expressive"):
                selected = [
                    row
                    for row in rows
                    if row["phase"] == phase
                    and row["readout"] == readout
                    and row["gate"] == gate
                ]
                grouped[phase][readout][gate] = {
                    "runs": len(selected),
                    "ood_accuracy": quantiles([row["ood_accuracy"] for row in selected]),
                    "unseen_count_accuracy": quantiles(
                        [row["unseen_count_accuracy"] for row in selected]
                    ),
                    "solved_runs_at_0.80": sum(row["ood_accuracy"] >= 0.80 for row in selected),
                    "min_eigenvalue": quantiles(
                        [row["min_eigenvalue"] for row in selected]
                    ),
                    "all_contraction_certified": all(
                        row["contraction_certified"] for row in selected
                    ),
                }

    followup_memory = grouped["gate_init_followup"]["memory_only"]
    expressive_rows = [row for row in rows if row["gate"] == "expressive"]
    near_reflection = [row for row in expressive_rows if row["min_eigenvalue"] <= -0.99]
    away_from_reflection = [row for row in expressive_rows if row["min_eigenvalue"] > -0.99]
    followup_rule = {
        "all_memory_expressive_negative_fraction_at_least_0.90": all(
            row["negative_fraction"] >= 0.90
            for row in rows
            if row["phase"] == "gate_init_followup"
            and row["readout"] == "memory_only"
            and row["gate"] == "expressive"
        ),
        "memory_expressive_median_at_least_0.80": (
            followup_memory["expressive"]["ood_accuracy"]["median"] >= 0.80
        ),
        "memory_expressive_q25_at_least_0.75": (
            followup_memory["expressive"]["ood_accuracy"]["q25"] >= 0.75
        ),
        "memory_margin_over_best_control_at_least_0.15": (
            followup_memory["expressive"]["ood_accuracy"]["median"]
            - max(
                followup_memory["current"]["ood_accuracy"]["median"],
                followup_memory["safe"]["ood_accuracy"]["median"],
            )
            >= 0.15
        ),
    }
    followup_rule["passed"] = all(followup_rule.values())
    return {
        "groups": grouped,
        "followup_preregistered_rule": followup_rule,
        "posthoc_reflection_proximity": {
            "threshold": "min_transition_eigenvalue <= -0.99",
            "near_reflection_runs": len(near_reflection),
            "near_reflection_solved_at_0.80": sum(
                row["ood_accuracy"] >= 0.80 for row in near_reflection
            ),
            "away_from_reflection_runs": len(away_from_reflection),
            "away_from_reflection_solved_at_0.80": sum(
                row["ood_accuracy"] >= 0.80 for row in away_from_reflection
            ),
            "status": "post_hoc_descriptive_not_confirmatory",
        },
    }


def write_csv(rows: list[dict]) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with (OUTPUT_DIR / "all_runs.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_svg(rows: list[dict]) -> None:
    selected = [row for row in rows if row["gate"] == "expressive"]
    width, height = 900, 540
    left, right, top, bottom = 82, 30, 58, 72
    plot_width = width - left - right
    plot_height = height - top - bottom
    x_min, x_max = -1.05, 0.60
    y_min, y_max = 0.45, 1.00

    def x_position(value: float) -> float:
        return left + (value - x_min) / (x_max - x_min) * plot_width

    def y_position(value: float) -> float:
        return top + (y_max - value) / (y_max - y_min) * plot_height

    colors = {"preregistered": "#386CB0", "gate_init_followup": "#E6550D"}
    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<style>text{font-family:Inter,Arial,sans-serif;fill:#202124}.axis{stroke:#444;stroke-width:1.2}.grid{stroke:#ddd;stroke-width:1}.threshold{stroke:#777;stroke-width:1.2;stroke-dasharray:6 5}.point{stroke:#222;stroke-width:0.8;opacity:.88}</style>',
        '<text x="450" y="28" text-anchor="middle" font-size="19" font-weight="600">Expressive-gate OOD accuracy vs learned transition eigenvalue</text>',
        '<text x="450" y="49" text-anchor="middle" font-size="12" fill="#555">Each point is one seed; circles = memory-only, squares = full REFA</text>',
    ]
    for tick in (-1.0, -0.75, -0.5, -0.25, 0.0, 0.25, 0.5):
        x = x_position(tick)
        svg.append(f'<line class="grid" x1="{x:.1f}" y1="{top}" x2="{x:.1f}" y2="{height-bottom}"/>')
        svg.append(f'<text x="{x:.1f}" y="{height-bottom+24}" text-anchor="middle" font-size="12">{tick:.2f}</text>')
    for tick in (0.5, 0.6, 0.7, 0.8, 0.9, 1.0):
        y = y_position(tick)
        svg.append(f'<line class="grid" x1="{left}" y1="{y:.1f}" x2="{width-right}" y2="{y:.1f}"/>')
        svg.append(f'<text x="{left-12}" y="{y+4:.1f}" text-anchor="end" font-size="12">{tick:.2f}</text>')
    svg.extend(
        [
            f'<line class="threshold" x1="{x_position(-0.99):.1f}" y1="{top}" x2="{x_position(-0.99):.1f}" y2="{height-bottom}"/>',
            f'<line class="threshold" x1="{left}" y1="{y_position(0.8):.1f}" x2="{width-right}" y2="{y_position(0.8):.1f}"/>',
            f'<line class="axis" x1="{left}" y1="{height-bottom}" x2="{width-right}" y2="{height-bottom}"/>',
            f'<line class="axis" x1="{left}" y1="{top}" x2="{left}" y2="{height-bottom}"/>',
            f'<text x="{left + plot_width/2:.1f}" y="{height-22}" text-anchor="middle" font-size="14">Minimum key-direction transition eigenvalue</text>',
            f'<text transform="translate(22 {top + plot_height/2:.1f}) rotate(-90)" text-anchor="middle" font-size="14">Length-64 accuracy</text>',
        ]
    )
    for row in selected:
        x = x_position(row["min_eigenvalue"])
        y = y_position(row["ood_accuracy"])
        color = colors[row["phase"]]
        if row["readout"] == "memory_only":
            svg.append(f'<circle class="point" cx="{x:.1f}" cy="{y:.1f}" r="5.2" fill="{color}"/>')
        else:
            svg.append(f'<rect class="point" x="{x-4.8:.1f}" y="{y-4.8:.1f}" width="9.6" height="9.6" fill="{color}"/>')
    legend_y = 78
    for index, (phase, color) in enumerate(colors.items()):
        x = 585 + index * 155
        svg.append(f'<circle cx="{x}" cy="{legend_y}" r="5" fill="{color}" stroke="#222"/>')
        label = "A1 default init" if phase == "preregistered" else "bias=2 follow-up"
        svg.append(f'<text x="{x+10}" y="{legend_y+4}" font-size="11">{label}</text>')
    svg.append("</svg>")
    SVG_PATH.parent.mkdir(parents=True, exist_ok=True)
    SVG_PATH.write_text("\n".join(svg), encoding="utf-8")


def main() -> None:
    rows = load_rows()
    write_csv(rows)
    summary = summarize(rows)
    (OUTPUT_DIR / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    write_svg(rows)
    print(json.dumps(summary["followup_preregistered_rule"], indent=2))
    print(json.dumps(summary["posthoc_reflection_proximity"], indent=2))


if __name__ == "__main__":
    main()

# Tracked experiment artifacts

These directories contain raw, machine-readable measurements that support committed
research claims. Do not replace them with selected or rounded results.

- `a1_keyed_parity/` — frozen default-initialization A1 matrix: 3 gates × 2 readouts ×
  10 seeds.
- `a1_gate_init_followup/` — fresh-seed, preregistered bias-initialization follow-up:
  3 gates × 2 readouts × 10 seeds.
- `a1_combined_analysis/` — deterministic combined CSV and summary produced by
  `python scripts/analyze_parity.py`.
- `a1_keyed_parity_dev/` — final pre-confirmatory development check; not included in
  confirmatory statistics.

Each raw JSON records configuration, epoch losses, seed, hardware, parent Git commit,
dirty-tree status, and a SHA-256 fingerprint of the exact working source. The report in
`docs/A1-KEYED-PARITY-RESULTS.md` states the decision rules and limitations.

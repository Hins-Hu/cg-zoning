#!/usr/bin/env python3
"""Run the Chattanooga equity case study (Figure 5 / HESDC results).

For each replicate, runs the CG entry point on Chattanooga twice with the same
seed: once with the demand_max objective (baseline) and once with the
equity_combined objective, then aggregates the per-replicate metrics and
threshold-sensitivity tables into summary CSV/JSON files under
output/equity/. Requires the equity score CSV built by cg/equity.py. The
paper experiments used the proprietary weekday OD table as
data/demand_chatt.csv (see data/README.md).
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path


REPO_DIR = Path(__file__).resolve().parent.parent
PYTHON = Path(sys.executable)
RUNNER = REPO_DIR / "experiments" / "run_cg.py"
DEFAULT_EQUITY_CSV = REPO_DIR / "output" / "equity" / "chattanooga_res8_equity_scores.csv"
DEFAULT_OUTPUT_ROOT = REPO_DIR / "output" / "equity"
DEFAULT_PERCENTILES = (0.70, 0.75, 0.80, 0.85, 0.90)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resolution", type=int, default=8)
    parser.add_argument("--budget", type=int, default=8)
    parser.add_argument("--single-zone-budget", type=float, default=2.0)
    parser.add_argument("--alpha", type=float, default=5.0)
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--cg-time-limit", type=int, default=1200)
    parser.add_argument("--pricing-time-limit", type=int, default=120)
    parser.add_argument("--pricing-method", type=str, default="random_init")
    parser.add_argument("--num-random-runs", type=int, default=10)
    parser.add_argument("--num-columns", type=int, default=1)
    parser.add_argument("--replicates", type=int, default=1)
    parser.add_argument("--base-seed", type=int, default=0)
    parser.add_argument("--equity-score-csv", type=Path, default=DEFAULT_EQUITY_CSV)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    return parser.parse_args()


def build_run_command(
    args: argparse.Namespace,
    output_dir: Path,
    objective_mode: str,
    random_seed: int,
) -> list[str]:
    return [
        str(PYTHON),
        str(RUNNER),
        "--city",
        "chatt",
        "--resolution",
        str(args.resolution),
        "--budget",
        str(args.budget),
        "--single_zone_budget",
        str(args.single_zone_budget),
        "--alpha",
        str(args.alpha),
        "--beta",
        str(args.beta),
        "--cg_time_limit",
        str(args.cg_time_limit),
        "--pricing_time_limit",
        str(args.pricing_time_limit),
        "--pricing_method",
        args.pricing_method,
        "--num_random_runs",
        str(args.num_random_runs),
        "--num_columns",
        str(args.num_columns),
        "--random_seed",
        str(random_seed),
        "--objective_mode",
        objective_mode,
        "--equity_score_csv",
        str(args.equity_score_csv),
        "--output_dir",
        str(output_dir),
    ]


def run_one(command: list[str], cwd: Path) -> None:
    print("Running:", " ".join(command))
    subprocess.run(command, cwd=cwd, check=True)


def load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2))


def summarize_metrics(run_root: Path, replicates: list[dict]) -> None:
    metric_keys = [
        "raw_demand_served",
        "average_served_origin_equity",
        "served_demand_from_top_quartile_equity_origins",
        "served_share_from_top_quartile_equity_origins",
        "served_rate_from_top_quartile_equity_origins",
        "cg_runtime_seconds",
    ]

    rows = []
    aggregate = {
        "replicates": [],
        "means": {"baseline": {}, "equity": {}, "delta_equity_minus_baseline": {}},
    }

    for item in replicates:
        seed = item["seed"]
        baseline_metrics = load_json(item["baseline_dir"] / "case_study_metrics.json")
        equity_metrics = load_json(item["equity_dir"] / "case_study_metrics.json")
        aggregate["replicates"].append(
            {
                "replicate_id": item["replicate_id"],
                "seed": seed,
                "baseline_dir": str(item["baseline_dir"]),
                "equity_dir": str(item["equity_dir"]),
                "baseline_metrics": baseline_metrics,
                "equity_metrics": equity_metrics,
            }
        )
        row = {
            "replicate_id": item["replicate_id"],
            "seed": seed,
        }
        for key in metric_keys:
            row[f"baseline_{key}"] = baseline_metrics[key]
            row[f"equity_{key}"] = equity_metrics[key]
            row[f"delta_{key}"] = equity_metrics[key] - baseline_metrics[key]
        rows.append(row)

    for key in metric_keys:
        baseline_values = [row[f"baseline_{key}"] for row in rows]
        equity_values = [row[f"equity_{key}"] for row in rows]
        delta_values = [row[f"delta_{key}"] for row in rows]
        aggregate["means"]["baseline"][key] = sum(baseline_values) / len(baseline_values)
        aggregate["means"]["equity"][key] = sum(equity_values) / len(equity_values)
        aggregate["means"]["delta_equity_minus_baseline"][key] = sum(delta_values) / len(delta_values)

    summary_csv = run_root / "replicate_summary.csv"
    with summary_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    write_json(run_root / "aggregate_summary.json", aggregate)


def summarize_threshold_sensitivity(run_root: Path, replicates: list[dict]) -> None:
    rows = []
    for percentile in DEFAULT_PERCENTILES:
        entry = {
            "percentile": percentile,
            "baseline_served_rate_mean": 0.0,
            "equity_served_rate_mean": 0.0,
            "delta_served_rate_mean": 0.0,
            "baseline_threshold_mean": 0.0,
            "equity_threshold_mean": 0.0,
        }
        baseline_rates = []
        equity_rates = []
        baseline_thresholds = []
        equity_thresholds = []

        for item in replicates:
            with (item["baseline_dir"] / "equity_threshold_sensitivity.csv").open() as f:
                baseline_rows = list(csv.DictReader(f))
            with (item["equity_dir"] / "equity_threshold_sensitivity.csv").open() as f:
                equity_rows = list(csv.DictReader(f))

            baseline_row = next(r for r in baseline_rows if abs(float(r["percentile"]) - percentile) < 1e-9)
            equity_row = next(r for r in equity_rows if abs(float(r["percentile"]) - percentile) < 1e-9)
            baseline_rates.append(float(baseline_row["served_rate_from_high_equity_origins"]))
            equity_rates.append(float(equity_row["served_rate_from_high_equity_origins"]))
            baseline_thresholds.append(float(baseline_row["threshold"]))
            equity_thresholds.append(float(equity_row["threshold"]))

        entry["baseline_served_rate_mean"] = sum(baseline_rates) / len(baseline_rates)
        entry["equity_served_rate_mean"] = sum(equity_rates) / len(equity_rates)
        entry["delta_served_rate_mean"] = entry["equity_served_rate_mean"] - entry["baseline_served_rate_mean"]
        entry["baseline_threshold_mean"] = sum(baseline_thresholds) / len(baseline_thresholds)
        entry["equity_threshold_mean"] = sum(equity_thresholds) / len(equity_thresholds)
        rows.append(entry)

    with (run_root / "threshold_sensitivity_summary.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    write_json(run_root / "threshold_sensitivity_summary.json", rows)


def main() -> int:
    args = parse_args()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_root = args.output_root / f"chatt_case_study_{timestamp}"
    run_root.mkdir(parents=True, exist_ok=True)

    replicates = []
    for replicate_idx in range(args.replicates):
        seed = args.base_seed + replicate_idx
        replicate_name = f"replicate_{replicate_idx + 1:02d}_seed_{seed}"
        baseline_dir = run_root / replicate_name / "baseline_demand_max"
        equity_dir = run_root / replicate_name / "equity_combined_mu05"
        baseline_dir.mkdir(parents=True, exist_ok=True)
        equity_dir.mkdir(parents=True, exist_ok=True)

        run_one(build_run_command(args, baseline_dir, "demand_max", seed), REPO_DIR)
        run_one(build_run_command(args, equity_dir, "equity_combined", seed), REPO_DIR)

        replicates.append(
            {
                "replicate_id": replicate_idx + 1,
                "seed": seed,
                "baseline_dir": baseline_dir,
                "equity_dir": equity_dir,
            }
        )

    summarize_metrics(run_root, replicates)
    summarize_threshold_sensitivity(run_root, replicates)

    print("\nCase study outputs:")
    print(f"  root:     {run_root}")
    for item in replicates:
        print(
            f"  rep {item['replicate_id']:02d}, seed {item['seed']}: "
            f"{item['baseline_dir']} | {item['equity_dir']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Result dumping and case-study metric helpers for the experiment entry points.

Extracted verbatim from the CG entry point: equity-score loading, objective
construction, case-study metrics, threshold sensitivity, and the JSON/CSV/TXT
writers. Output filenames are unchanged.
"""

import itertools
import json
import os

import pandas as pd


def load_equity_scores(equity_score_csv):
    if equity_score_csv is None or not os.path.exists(equity_score_csv):
        return None

    equity_df = pd.read_csv(equity_score_csv)
    required_cols = {"cell_id", "equity_score"}
    missing_cols = required_cols - set(equity_df.columns)
    if missing_cols:
        raise ValueError(f"Missing required equity score columns: {sorted(missing_cols)}")

    equity_df = equity_df.dropna(subset=["equity_score"]).copy()
    return {row["cell_id"]: float(row["equity_score"]) for _, row in equity_df.iterrows()}


def build_objective_demand(demand, objective_mode, equity_scores):
    if objective_mode == "demand_max":
        return demand

    if objective_mode == "equity_combined":
        if equity_scores is None:
            raise ValueError("Equity scores are required for the equity_combined objective.")
        return {
            i: {j: float(equity_scores[i]) * demand[i][j] for j in demand[i]}
            for i in demand
        }

    raise ValueError(f"Unknown objective_mode: {objective_mode}")


def calculate_case_study_metrics(selected_zones, demand, objective_demand, equity_scores):
    valid_pairs = set()
    all_cells = set()

    for zone in selected_zones:
        for pair in itertools.permutations(zone, 2):
            valid_pairs.add(pair)
        all_cells.update(zone)

    raw_total = 0.0
    objective_total = 0.0
    weighted_raw_total = 0.0
    served_high_equity_raw = 0.0
    total_high_equity_raw = 0.0

    high_equity_cells = set()
    if equity_scores:
        score_series = pd.Series(equity_scores, dtype=float)
        threshold = float(score_series.quantile(0.75))
        high_equity_cells = {
            cell_id for cell_id, score in equity_scores.items() if float(score) >= threshold
        }
    else:
        threshold = None

    for i in demand:
        if i in high_equity_cells:
            total_high_equity_raw += sum(demand[i].values())

    for i, j in valid_pairs:
        raw = float(demand[i][j])
        raw_total += raw
        objective_total += float(objective_demand[i][j])
        if equity_scores:
            weighted_raw_total += float(equity_scores[i]) * raw
            if i in high_equity_cells:
                served_high_equity_raw += raw

    for i in all_cells:
        raw = float(demand[i][i])
        raw_total += raw
        objective_total += float(objective_demand[i][i])
        if equity_scores:
            weighted_raw_total += float(equity_scores[i]) * raw
            if i in high_equity_cells:
                served_high_equity_raw += raw

    average_served_origin_equity = weighted_raw_total / raw_total if raw_total > 0 else 0.0
    served_share_high_equity = served_high_equity_raw / raw_total if raw_total > 0 else 0.0
    served_rate_high_equity = (
        served_high_equity_raw / total_high_equity_raw if total_high_equity_raw > 0 else 0.0
    )

    return {
        "raw_demand_served": raw_total,
        "objective_weighted_demand_served": objective_total,
        "average_served_origin_equity": average_served_origin_equity,
        "served_demand_from_top_quartile_equity_origins": served_high_equity_raw,
        "served_share_from_top_quartile_equity_origins": served_share_high_equity,
        "served_rate_from_top_quartile_equity_origins": served_rate_high_equity,
        "top_quartile_equity_threshold": threshold,
    }


def save_case_study_metrics(output_dir, metrics):
    json_path = os.path.join(output_dir, "case_study_metrics.json")
    txt_path = os.path.join(output_dir, "case_study_metrics.txt")

    with open(json_path, "w") as f:
        json.dump(metrics, f, indent=2)

    with open(txt_path, "w") as f:
        for key, value in metrics.items():
            f.write(f"{key}: {value}\n")


def save_selected_zones(output_dir, zones):
    json_path = os.path.join(output_dir, "selected_zones.json")
    serializable_zones = [sorted(list(zone)) for zone in zones]
    with open(json_path, "w") as f:
        json.dump(serializable_zones, f, indent=2)


def calculate_equity_threshold_sensitivity(
    selected_zones,
    demand,
    equity_scores,
    percentiles=(0.70, 0.75, 0.80, 0.85, 0.90),
):
    if not equity_scores:
        return []

    valid_pairs = set()
    all_cells = set()
    for zone in selected_zones:
        for pair in itertools.permutations(zone, 2):
            valid_pairs.add(pair)
        all_cells.update(zone)

    raw_total = 0.0
    served_raw_by_origin = {}
    total_raw_by_origin = {}

    for i in demand:
        total_raw_by_origin[i] = float(sum(demand[i].values()))
        served_raw_by_origin[i] = 0.0

    for i, j in valid_pairs:
        raw = float(demand[i][j])
        raw_total += raw
        served_raw_by_origin[i] += raw

    for i in all_cells:
        raw = float(demand[i][i])
        raw_total += raw
        served_raw_by_origin[i] += raw

    score_series = pd.Series(equity_scores, dtype=float)
    rows = []
    for percentile in percentiles:
        threshold = float(score_series.quantile(percentile))
        high_equity_cells = {
            cell_id for cell_id, score in equity_scores.items() if float(score) >= threshold
        }
        served_high_equity_raw = sum(served_raw_by_origin[cell_id] for cell_id in high_equity_cells)
        total_high_equity_raw = sum(total_raw_by_origin[cell_id] for cell_id in high_equity_cells)
        rows.append(
            {
                "percentile": float(percentile),
                "threshold": threshold,
                "high_equity_cell_count": len(high_equity_cells),
                "served_demand_from_high_equity_origins": served_high_equity_raw,
                "total_demand_from_high_equity_origins": total_high_equity_raw,
                "served_share_from_high_equity_origins": (
                    served_high_equity_raw / raw_total if raw_total > 0 else 0.0
                ),
                "served_rate_from_high_equity_origins": (
                    served_high_equity_raw / total_high_equity_raw if total_high_equity_raw > 0 else 0.0
                ),
            }
        )

    return rows


def save_equity_threshold_sensitivity(output_dir, sensitivity_rows):
    if not sensitivity_rows:
        return

    json_path = os.path.join(output_dir, "equity_threshold_sensitivity.json")
    csv_path = os.path.join(output_dir, "equity_threshold_sensitivity.csv")

    with open(json_path, "w") as f:
        json.dump(sensitivity_rows, f, indent=2)

    pd.DataFrame(sensitivity_rows).to_csv(csv_path, index=False)


def save_connectivity_repair_summary(output_path, scope, repair_stats, original_total_demand, repaired_total_demand):
    payload = {
        "scope": scope,
        "original_total_demand_served": original_total_demand,
        "repaired_total_demand_served": repaired_total_demand,
        "total_cells_added": sum(len(stat.added_cells) for stat in repair_stats),
        "zones": [
            {
                "zone_index": stat.zone_index,
                "original_size": stat.original_size,
                "repaired_size": stat.repaired_size,
                "added_cell_count": len(stat.added_cells),
                "added_cells": stat.added_cells,
            }
            for stat in repair_stats
        ],
    }
    with open(output_path, "w") as f:
        json.dump(payload, f, indent=2)

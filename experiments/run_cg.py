"""Column Generation experiment entry point (Tables 2-4, Figures 3-4).

Orchestrates one CG run on a real city: loads the road network and demand,
builds the H3 hex instance, calls cg.algo.column_generation with the chosen
pricing method (exact_ilp, exact_qp, or the random_init heuristic), then
writes metrics, zone maps, and optional connectivity-repair outputs to
--output_dir. Run from the repo root. See README for paper parameters.
"""

import os
import sys

# Make the repo root importable so `cg` resolves when this entry point is run as a
# script (convention: run from the repo root so `data/` and `output/` resolve).
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import time

import numpy as np
import osmnx as ox
import pandas as pd

from cg.algo import column_generation
from cg.connectivity_repair import repair_zones
from cg.utils import calculate_total_demand_served, construct_block_graph
from experiments.shared.city_data import (
    aggregate_demand_to_hexagons,
    build_hex_grid,
    load_center_distances,
    load_city_boundary,
    normalize_square_distances,
)
from experiments.shared.experiment_plots import (
    plot_column_size_distribution,
    plot_demand_histogram,
    plot_demand_maps,
    plot_distance_histograms,
    save_four_zones_side_by_side,
    save_four_zones_side_by_side_with_patches,
    save_zone_map,
    save_zone_map_with_patches,
)
from experiments.shared.results import (
    build_objective_demand,
    calculate_case_study_metrics,
    calculate_equity_threshold_sensitivity,
    load_equity_scores,
    save_case_study_metrics,
    save_connectivity_repair_summary,
    save_equity_threshold_sensitivity,
    save_selected_zones,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Run column generation for real-world zone optimization.")
    parser.add_argument("--city", type=str, default="chatt", help="City name (e.g., 'chatt', 'boston', 'atlanta')")
    parser.add_argument("--cg_time_limit", type=int, default=600, help="Time limit for column generation in seconds")
    parser.add_argument("--pricing_time_limit", type=int, default=120, help="Time limit for pricing in seconds")
    parser.add_argument("--alpha", type=float, default=5, help="Weight for the linear zone cost")
    parser.add_argument("--beta", type=float, default=1, help="Intercept for the linear zone cost")
    parser.add_argument("--budget", type=int, default=8, help="Budget for all zones")
    parser.add_argument("--num_columns", type=int, default=1, help="Number of columns to generate per iteration")
    parser.add_argument("--num_random_runs", type=int, default=5, help="Number of random runs for pricing")
    parser.add_argument("--pricing_method", type=str, default='exact_ilp', help="Pricing method (e.g., 'exact_ilp')")
    parser.add_argument("--resolution", type=int, default=7, help="H3 resolution for hexagons")
    parser.add_argument("--output_dir", type=str, default="output", help="Output directory for results")
    parser.add_argument("--single_zone_budget", type=float, default=2, help="Budget for the pricing to get a single zone")
    parser.add_argument("--random_seed", type=int, default=None, help="Random seed for reproducible heuristic runs")
    parser.add_argument(
        "--objective_mode",
        type=str,
        default="demand_max",
        choices=["demand_max", "equity_combined"],
        help="Optimization objective to use for the case study runs.",
    )
    parser.add_argument(
        "--equity_score_csv",
        type=str,
        default=None,
        help="CSV containing cell-level combined equity scores (see cg/equity.py). "
             "Only needed for the equity_combined objective.",
    )
    parser.add_argument(
        "--post_process_connectivity",
        action="store_true",
        help="Apply MTZ connectivity repair after zone selection and save repaired figures.",
    )
    parser.add_argument(
        "--post_process_scope",
        type=str,
        default="all_cells",
        choices=["all_cells", "zero_demand"],
        help="Candidate set for connectivity repair.",
    )
    parser.add_argument(
        "--diagnostic_plots",
        action="store_true",
        help="Also write the diagnostic plots (distance and demand "
             "histograms, column-size distribution).",
    )

    return parser.parse_args()


def get_connectivity_repair_candidates(hex_gdf, scope):
    if scope == "zero_demand":
        zero_demand_mask = (hex_gdf['out_demand'] == 0) & (hex_gdf['in_demand'] == 0)
        return set(hex_gdf.index[zero_demand_mask].tolist())
    return set(hex_gdf.index.tolist())


def main(args):
    if args.random_seed is not None:
        np.random.seed(args.random_seed)

    os.makedirs(args.output_dir, exist_ok=True)

    """
    Print experiment parameters
    """
    print("\n" + "="*60)
    print("EXPERIMENT PARAMETERS")
    print("="*60)
    print(f"City: {args.city}")
    print(f"Resolution: {args.resolution}")
    print(f"Budget: {args.budget}")
    print(f"Alpha: {args.alpha}")
    print(f"Beta: {args.beta}")
    print(f"CG Time Limit: {args.cg_time_limit} seconds")
    print(f"Pricing Time Limit: {args.pricing_time_limit} seconds")
    print(f"Number of Columns: {args.num_columns}")
    print(f"Number of Random Runs: {args.num_random_runs}")
    print(f"Pricing Method: {args.pricing_method}")
    print(f"Output Directory: {args.output_dir}")
    print(f"Single Zone Budget: {args.single_zone_budget}")
    print(f"Random Seed: {args.random_seed}")
    print(f"Objective Mode: {args.objective_mode}")
    print(f"Equity Score CSV: {args.equity_score_csv}")
    print("="*60 + "\n")

    """
    Read the city map, road network, and hex grid
    """
    G = ox.load_graphml(f'data/{args.city}_graph.graphml')
    boundary = load_city_boundary(args.city)
    resolution = args.resolution
    hex_gdf, hexagons = build_hex_grid(G, boundary, resolution)

    """
    Pre-process demand
    """
    raw_demand = pd.read_csv(f'data/demand_{args.city}.csv', usecols=['origin_lat', 'origin_lon', 'destination_lat', 'destination_lon'])
    demand = aggregate_demand_to_hexagons(raw_demand, boundary, hex_gdf, hexagons)

    """
    Demand visualization
    """
    plot_demand_maps(hex_gdf, resolution, args.output_dir)

    """
    Construct the graph for the H3 system
    """
    H_hex = construct_block_graph(hex_gdf, G)

    """
    Load pre-computed shortest paths and check the statistics
    """
    center_distances = load_center_distances(args.city, resolution)
    norm_square_center_distances = normalize_square_distances(center_distances, power=2, norm_max=1)

    demand_list = [v for inner in demand.values() for v in inner.values()]
    truncated_demand_list = [v for inner in demand.values() for v in inner.values() if v <= 50 and v > 0]
    if args.diagnostic_plots:
        plot_distance_histograms(center_distances, norm_square_center_distances, resolution, args.output_dir)
        plot_demand_histogram(truncated_demand_list, resolution, args.output_dir)

    # Calculate all the demand
    total_demand = sum(demand_list)
    print("The total demand is: ", total_demand)
    total_demand_under_50 = sum(truncated_demand_list)
    print("The total demand under 50 is: ", total_demand_under_50)

    """
    Solve the zoning problem using CG
    """
    cg_start_time = time.time()

    equity_scores = load_equity_scores(args.equity_score_csv)
    if args.objective_mode == "equity_combined" and equity_scores is None:
        raise ValueError("Combined equity objective requested, but no equity score CSV was found.")
    if equity_scores is not None:
        missing_score_nodes = sorted(set(hexagons) - set(equity_scores.keys()))
        if missing_score_nodes:
            raise ValueError(
                f"Equity score CSV is missing scores for {len(missing_score_nodes)} hex cells."
            )

    objective_demand = build_objective_demand(demand, args.objective_mode, equity_scores)

    zones, columns, duals, master_LP_objs, avg_pricing_time = column_generation(
                                                            list(H_hex.nodes), budget=args.budget, demand=demand,
                                                            objective_demand=objective_demand,
                                                            distances=norm_square_center_distances,
                                                            alpha=args.alpha, beta=args.beta,
                                                            num_columns=args.num_columns,
                                                            cg_time_limit=args.cg_time_limit,
                                                            pricing_time_limit=args.pricing_time_limit,
                                                            pricing_method=args.pricing_method,
                                                            num_random_runs=args.num_random_runs,
                                                            single_zone_budget=args.single_zone_budget)

    cg_elapsed_time = time.time() - cg_start_time

    # Save average pricing time to file
    with open(f"{args.output_dir}/avg_pricing_time.txt", "w") as f:
        f.write(f"Average pricing time: {avg_pricing_time:.2f} seconds\n")

    """
    Verification
    """
    # Verify that each column generated is unique (no cycling), meaning CG is working correctly, though not necessarily efficient
    columns_frozen = [frozenset(columns[i][0]) for i in columns.keys()]
    has_duplicates = len(columns_frozen) != len(set(columns_frozen))
    print("Has duplicate columns: ", has_duplicates)

    print("Number of columns:", len(columns_frozen))
    print("Number of distinct columns:", len(set(columns_frozen)))

    # Extract the sizes of the generated columns and plot the distribution
    if args.diagnostic_plots:
        column_sizes = [len(columns[i][0]) for i in columns.keys()]
        plot_column_size_distribution(column_sizes, args.output_dir)

    """
    Visualize optimal zones
    """
    save_zone_map(
        hex_gdf,
        zones,
        f"{args.output_dir}/zones_res{resolution}_budget{args.budget}.png",
    )
    save_four_zones_side_by_side(
        hex_gdf,
        zones,
        f"{args.output_dir}/four_zones_plot.png",
    )
    save_selected_zones(args.output_dir, zones)

    """
    Calculate the total demand served
    """
    TT_demand = calculate_total_demand_served(zones, demand)
    print("The total demand served is: ", TT_demand)

    # Save the total demand served to a text file for easier post-processing
    with open(f"{args.output_dir}/total_demand_served.txt", "w") as f:
        f.write(str(TT_demand))
        f.write("\n")

    case_study_metrics = calculate_case_study_metrics(
        selected_zones=zones,
        demand=demand,
        objective_demand=objective_demand,
        equity_scores=equity_scores,
    )
    case_study_metrics.update(
        {
            "city": args.city,
            "resolution": args.resolution,
            "budget": args.budget,
            "single_zone_budget": args.single_zone_budget,
            "alpha": args.alpha,
            "beta": args.beta,
            "cg_time_limit": args.cg_time_limit,
            "pricing_time_limit": args.pricing_time_limit,
            "pricing_method": args.pricing_method,
            "num_random_runs": args.num_random_runs,
            "num_columns": args.num_columns,
            "random_seed": args.random_seed,
            "objective_mode": args.objective_mode,
            "avg_pricing_time_seconds": avg_pricing_time,
            "cg_runtime_seconds": cg_elapsed_time,
            "selected_zone_count": len(zones),
        }
    )
    save_case_study_metrics(args.output_dir, case_study_metrics)
    threshold_sensitivity = calculate_equity_threshold_sensitivity(
        selected_zones=zones,
        demand=demand,
        equity_scores=equity_scores,
    )
    save_equity_threshold_sensitivity(args.output_dir, threshold_sensitivity)

    """
    Post-process zones with MTZ connectivity repair
    """
    if args.post_process_connectivity:
        candidate_cells = get_connectivity_repair_candidates(hex_gdf, args.post_process_scope)
        repaired_zones, repair_stats = repair_zones(
            zones=zones,
            distances=norm_square_center_distances,
            candidate_cells=candidate_cells,
        )
        added_cells_by_zone = {
            stat.zone_index: stat.added_cells
            for stat in repair_stats
        }
        repaired_total_demand = calculate_total_demand_served(repaired_zones, demand)

        print("\n" + "="*60)
        print("CONNECTIVITY REPAIR")
        print("="*60)
        print(f"Scope: {args.post_process_scope}")
        print(f"Total cells added: {sum(len(stat.added_cells) for stat in repair_stats)}")
        print(f"Repaired total demand served: {repaired_total_demand}")
        print("="*60 + "\n")

        with open(f"{args.output_dir}/total_demand_served_postprocessed.txt", "w") as f:
            f.write(str(repaired_total_demand))
            f.write("\n")

        save_connectivity_repair_summary(
            output_path=f"{args.output_dir}/connectivity_repair_summary.json",
            scope=args.post_process_scope,
            repair_stats=repair_stats,
            original_total_demand=TT_demand,
            repaired_total_demand=repaired_total_demand,
        )

        save_zone_map_with_patches(
            hex_gdf,
            repaired_zones,
            f"{args.output_dir}/zones_res{resolution}_budget{args.budget}_postprocessed.png",
            added_cells_by_zone=added_cells_by_zone,
        )
        save_four_zones_side_by_side_with_patches(
            hex_gdf,
            repaired_zones,
            f"{args.output_dir}/four_zones_plot_postprocessed.png",
            added_cells_by_zone=added_cells_by_zone,
        )

    # Print column generation time at the end
    print(f"\nColumn Generation completed in {cg_elapsed_time:.2f} seconds ({cg_elapsed_time/60:.2f} minutes)\n")


if __name__ == "__main__":
    args = parse_args()
    main(args)

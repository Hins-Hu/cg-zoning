"""CliqueGen baseline experiment entry point (Table 2, Figure 3).

Orchestrates one CliqueGen + ZoningILP run on a real city: builds the same
H3 hex instance as the CG entry point, enumerates all candidate zones within the
single-zone budget (cliquegen.clique_generator_on_map), selects zones with
the max-coverage ILP (cliquegen.solve_ILP), and writes the zone map and a
results_{city}_res{r}.txt summary to --output_dir. Run from the repo root.
"""

import os
import sys

# Make the repo root importable so `cg` and `cliquegen` resolve when this
# entry point is run as a script (convention: run from the repo root).
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import time

import osmnx as ox
import pandas as pd

from cg.utils import calculate_total_demand_served, construct_block_graph
from cliquegen.cliquegen import clique_generator_on_map, solve_ILP
from experiments.shared.city_data import (
    aggregate_demand_to_hexagons,
    build_hex_grid,
    load_center_distances,
    load_city_boundary,
    normalize_square_distances,
)
from experiments.shared.experiment_plots import visualize_four_zones_side_by_side


def parse_args():
    parser = argparse.ArgumentParser(description="Run CliqueGen + ZoningILP for real-world zone optimization.")
    parser.add_argument("--city", type=str, default="chatt", help="City name (e.g., 'chatt', 'boston', 'atlanta')")
    parser.add_argument("--resolution", type=int, default=7, help="H3 resolution for hexagons")
    parser.add_argument("--single_zone_budget", type=float, default=1.6, help="Budget for a single zone")
    parser.add_argument("--budget", type=float, default=8, help="Total budget for all zones")
    parser.add_argument("--alpha", type=float, default=5, help="Weight for the linear zone cost")
    parser.add_argument("--beta", type=float, default=1, help="Intercept for the linear zone cost")
    parser.add_argument("--time_limit", type=float, default=None, help="Time limit in seconds for clique generation (None for no limit)")
    parser.add_argument("--output_dir", type=str, default="output", help="Output directory for results")
    return parser.parse_args()


def main(args):
    # Extract parameters
    city = args.city
    resolution = args.resolution
    SINGLE_ZONE_BUDGET = args.single_zone_budget
    BUDGET = args.budget
    ALPHA = args.alpha
    BETA = args.beta
    time_limit = args.time_limit
    output_dir = args.output_dir

    os.makedirs(output_dir, exist_ok=True)

    # ### Read the city map, road network, and hex grid

    G = ox.load_graphml(f'data/{city}_graph.graphml')
    boundary = load_city_boundary(city)
    hex_gdf, hexagons = build_hex_grid(G, boundary, resolution)

    # ### Pre-process the demand

    raw_demand = pd.read_csv(f'data/demand_{city}.csv', usecols=['origin_lat', 'origin_lon', 'destination_lat', 'destination_lon'])
    demand = aggregate_demand_to_hexagons(raw_demand, boundary, hex_gdf, hexagons)

    # ### Construct the graph for the H3 system

    H_hex = construct_block_graph(hex_gdf, G)

    # ### Load pre-computed shortest paths and check the statistics

    center_distances = load_center_distances(city, resolution)
    norm_square_center_distances = normalize_square_distances(center_distances, power=2, norm_max=1)

    demand_list = [v for inner in demand.values() for v in inner.values()]
    truncated_demand_list = [v for inner in demand.values() for v in inner.values() if v <= 50 and v > 0]

    # Calculate all the demand
    total_demand = sum(demand_list)
    print("The total demand is: ", total_demand)
    total_demand_under_50 = sum(truncated_demand_list)
    print("The total demand under 50 is: ", total_demand_under_50)

    # ### Solving the zoning problem using CliqueGen + ZoningILP

    # Start timing for clique generation
    start_time_total = time.time()
    start_time_clique = time.time()

    lst, clique_costs, cardinality, time_limit_reached = clique_generator_on_map(H_hex, norm_square_center_distances, alpha=ALPHA, beta=BETA, budget=SINGLE_ZONE_BUDGET, connectivity_threshold=1, time_limit=time_limit)

    clique_generation_time = time.time() - start_time_clique
    print(f"Clique generation time: {clique_generation_time:.2f} seconds")
    if time_limit_reached:
        print(f"Time limit of {time_limit}s reached - returning best cliques found so far")
    print(len(lst))
    print("Highest Cardinality", cardinality)

    # Start timing for ILP solving
    start_time_ilp = time.time()

    selected_zones_our = solve_ILP(H_hex, lst, clique_costs, demand, budget=BUDGET)

    ilp_solving_time = time.time() - start_time_ilp
    print(f"ILP solving time: {ilp_solving_time:.2f} seconds")
    print("Selected zones:", selected_zones_our)
    demand_served_our = calculate_total_demand_served(selected_zones_our, demand)
    print("Total demand served:", demand_served_our)

    total_time = time.time() - start_time_total
    print(f"Total time: {total_time:.2f} seconds")

    # Visualize optimal zones side by side
    visualize_four_zones_side_by_side(hex_gdf, selected_zones_our, output_dir)

    # Format results as text with key: value pairs
    results_text = (
        f"city: {city}\n"
        f"resolution: {resolution}\n"
        f"single_zone_budget: {SINGLE_ZONE_BUDGET}\n"
        f"beta: {BETA}\n"
        f"alpha: {ALPHA}\n"
        f"budget: {BUDGET}\n"
        f"time_limit: {time_limit if time_limit else 'None'}\n"
        f"time_limit_reached: {time_limit_reached}\n"
        f"num_cliques_generated: {len(lst)}\n"
        f"highest_cardinality: {cardinality}\n"
        f"clique_generation_time: {clique_generation_time:.2f}\n"
        f"ilp_solving_time: {ilp_solving_time:.2f}\n"
        f"total_time: {total_time:.2f}\n"
        f"num_zones_selected: {len(selected_zones_our)}\n"
        f"total_demand_served: {demand_served_our}\n"
    )

    results_file = os.path.join(output_dir, f"results_{city}_res{resolution}.txt")
    with open(results_file, 'w') as f:
        f.write(results_text)
    print(f"\nResults saved to {results_file}")


if __name__ == "__main__":
    args = parse_args()
    main(args)

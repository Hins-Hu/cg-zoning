"""Render the paper's aggregated demand heatmap for a city (no solving).

Builds the same H3 hex instance as the experiment entry points, aggregates
the city's OD demand to hexagons, and renders the total-demand heatmap in
the paper's style (viridis, PowerNorm) into --output_dir. Run from the
repo root.

Usage: python experiments/render_demand_map.py --city chatt --resolution 8
"""

import os
import sys

# Make the repo root importable so `cg` resolves when this script is run
# directly (convention: run from the repo root).
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse

import osmnx as ox
import pandas as pd

from experiments.shared.city_data import (
    aggregate_demand_to_hexagons,
    build_hex_grid,
    load_city_boundary,
)
from experiments.shared.experiment_plots import plot_total_demand_map


def parse_args():
    parser = argparse.ArgumentParser(description="Render aggregated OD demand maps for a city.")
    parser.add_argument("--city", type=str, default="chatt", help="City name (e.g., 'chatt', 'boston', 'atlanta')")
    parser.add_argument("--resolution", type=int, default=8, help="H3 resolution for hexagons")
    parser.add_argument("--output_dir", type=str, default="output/figures", help="Output directory for the maps")
    return parser.parse_args()


def main(args):
    os.makedirs(args.output_dir, exist_ok=True)

    G = ox.load_graphml(f'data/{args.city}_graph.graphml')
    boundary = load_city_boundary(args.city)
    hex_gdf, hexagons = build_hex_grid(G, boundary, args.resolution)

    raw_demand = pd.read_csv(f'data/demand_{args.city}.csv', usecols=['origin_lat', 'origin_lon', 'destination_lat', 'destination_lon'])
    aggregate_demand_to_hexagons(raw_demand, boundary, hex_gdf, hexagons)

    output_path = os.path.join(args.output_dir, f"demand_{args.city}_res{args.resolution}.png")
    plot_total_demand_map(hex_gdf, output_path)
    print(f"Wrote demand heatmap to {output_path}")


if __name__ == "__main__":
    main(parse_args())

"""Render the camera-ready Figure 4 (Chattanooga res-8 zones, 2x2 grid).

Reconstructs the selected and patched zones from a saved post-processed run
directory and renders the final 2x2 layout with the camera-ready tick and
font settings to output/figures/.

The source run directory (SOURCE_DIR, default output/figure4_source_run/)
must contain three artifacts of one CG run:
  - connectivity_repair_summary.json
  - four_zones_plot.png
  - four_zones_plot_postprocessed.png
These are produced by running the Chattanooga experiment with
--post_process_connectivity (experiments/run_cg.py, with the paper's
proprietary weekday OD table as data/demand_chatt.csv) and pointing
SOURCE_DIR at the resulting output directory.
"""

import json
import os
import sys
from pathlib import Path

# Make the repo root importable so `experiments.shared` resolves when this
# script is run directly (convention: run from the repo root).
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import geopandas as gpd
import h3
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import osmnx as ox
from PIL import Image
from shapely.geometry import Polygon

from experiments.shared.experiment_plots import save_four_zones_side_by_side_with_patches


matplotlib.use("Agg")


REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIR = REPO_ROOT / "output" / "figure4_source_run"
TARGET_DIR = REPO_ROOT / "output" / "figures"

OUTPUT_IMAGE = TARGET_DIR / "Zone1234_v4_x_tick_0p1_tickfont19_600dpi.png"


def build_hex_gdf():
    boundary = ox.geocode_to_gdf("Chattanooga, Tennessee, USA").geometry.iloc[0]
    hexagons = h3.geo_to_cells(boundary, 8)
    hex_polygons = [Polygon(h3.cell_to_boundary(hexagon)) for hexagon in hexagons]
    hex_gdf = gpd.GeoDataFrame(geometry=hex_polygons)
    hex_gdf["id"] = hexagons
    hex_gdf.set_index("id", inplace=True)
    # Respect the lon, lat convention used elsewhere in the repo.
    hex_gdf.geometry = hex_gdf.geometry.map(
        lambda geom: Polygon([(y, x) for x, y in geom.exterior.coords])
    )
    return hex_gdf


def is_colored(rgb):
    rgb = rgb.astype(int)
    return (rgb.max() - rgb.min()) >= 18 and rgb.mean() >= 60


def build_panel_coords(hex_gdf):
    fig, axes = plt.subplots(2, 2, figsize=(12, 12), sharex=True, sharey=True)
    axes = axes.flatten()
    for i, ax in enumerate(axes):
        hex_gdf.plot(ax=ax, color="lightgray", edgecolor="black", alpha=0.5)
        ax.set_title(f"Zone {i+1}", fontsize=14)
        ax.set_xlabel("Longitude", fontsize=10)
        ax.set_ylabel("Latitude", fontsize=10)
    plt.tight_layout()
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    bbox = fig.get_tightbbox(renderer)
    save_dpi = 300

    panel_coords = []
    for ax in axes:
        panel = []
        for hex_id, row in hex_gdf.iterrows():
            px, py = ax.transData.transform((row.geometry.centroid.x, row.geometry.centroid.y))
            sx = (px / fig.dpi - bbox.x0) * save_dpi
            sy = 2818 - (py / fig.dpi - bbox.y0) * save_dpi
            panel.append((hex_id, int(round(sx)), int(round(sy))))
        panel_coords.append(panel)

    plt.close(fig)
    return panel_coords


def detect_zones_from_image(img_array, panel_coords):
    zones = []
    for panel in panel_coords:
        zone = set()
        for hex_id, ix, iy in panel:
            if 0 <= ix < img_array.shape[1] and 0 <= iy < img_array.shape[0]:
                if is_colored(img_array[iy, ix]):
                    zone.add(hex_id)
        zones.append(zone)
    return zones


def reconstruct_repaired_zones(hex_gdf):
    with open(SOURCE_DIR / "connectivity_repair_summary.json") as f:
        summary = json.load(f)

    orig_img = np.array(Image.open(SOURCE_DIR / "four_zones_plot.png").convert("RGB"))
    post_img = np.array(Image.open(SOURCE_DIR / "four_zones_plot_postprocessed.png").convert("RGB"))
    panel_coords = build_panel_coords(hex_gdf)

    orig_sets = detect_zones_from_image(orig_img, panel_coords)
    post_sets = detect_zones_from_image(post_img, panel_coords)
    added_cells_by_zone = {
        zone_info["zone_index"]: set(zone_info["added_cells"])
        for zone_info in summary["zones"]
    }

    repaired_zones = []
    used_cells = set()
    notes = []

    for i in range(4):
        target_size = summary["zones"][i]["repaired_size"]
        repaired = set(orig_sets[i]) | set(added_cells_by_zone[i])

        # Recover any cells that are visible in the already-rendered postprocessed image.
        extras = list(post_sets[i] - repaired - used_cells)
        for cell in extras:
            if len(repaired) >= target_size:
                break
            repaired.add(cell)

        # Fill any remaining tiny gap with boundary-adjacent cells.
        while len(repaired) < target_size:
            candidate_scores = []
            for cell in set(hex_gdf.index) - repaired - used_cells:
                neighbors = set(h3.grid_disk(cell, 1)) & set(hex_gdf.index)
                score = len(neighbors & repaired)
                if score > 0:
                    candidate_scores.append((score, cell))
            if not candidate_scores:
                break
            candidate_scores.sort(reverse=True)
            repaired.add(candidate_scores[0][1])

        repaired_zones.append(repaired)
        used_cells.update(repaired)
        notes.append(
            {
                "zone_index": i,
                "reconstructed_original_count": len(orig_sets[i]),
                "added_cell_count": len(added_cells_by_zone[i]),
                "post_detected_count": len(post_sets[i]),
                "final_count": len(repaired),
                "target_count": target_size,
            }
        )

    return repaired_zones, added_cells_by_zone, notes


def main() -> None:
    OUTPUT_IMAGE.parent.mkdir(parents=True, exist_ok=True)
    hex_gdf = build_hex_gdf()
    repaired_zones, added_cells_by_zone, notes = reconstruct_repaired_zones(hex_gdf)

    # Same canonical rendering the experiment runs use for
    # four_zones_plot_postprocessed.png (camera-ready Figure 4 styling).
    save_four_zones_side_by_side_with_patches(
        hex_gdf,
        repaired_zones,
        OUTPUT_IMAGE,
        added_cells_by_zone=added_cells_by_zone,
    )

    with open(TARGET_DIR / "figure4_reconstruction_notes.json", "w") as f:
        json.dump(notes, f, indent=2)

    print(f"Wrote Figure 4: {OUTPUT_IMAGE}")


if __name__ == "__main__":
    main()

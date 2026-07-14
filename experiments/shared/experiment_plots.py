"""Plotting helpers for the experiment entry points.

Zone-map rendering (connectivity-repair patches drawn in the paper's
white-outline style) and the optional diagnostic demand/distance plots.
Extracted from the original experiment scripts; filenames and figure styling match the
paper.
"""

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.ticker as ticker
from matplotlib.colors import PowerNorm
import numpy as np


def save_zone_map(hex_gdf, zones, output_path):
    save_zone_map_with_patches(hex_gdf, zones, output_path)


def save_zone_map_with_patches(hex_gdf, zones, output_path, added_cells_by_zone=None):
    colors = list(mcolors.TABLEAU_COLORS.values())
    fig, ax = plt.subplots(figsize=(10, 10))
    hex_gdf.plot(ax=ax, color='lightgray', edgecolor='black', alpha=0.5)

    for i, hex_set in enumerate(zones):
        base_color = colors[i % len(colors)]
        added_cells = set() if added_cells_by_zone is None else set(added_cells_by_zone.get(i, []))
        original_cells = set(hex_set) - added_cells

        if original_cells:
            subset_hex_gdf = hex_gdf.loc[list(original_cells)]
            subset_hex_gdf.plot(
                ax=ax,
                color=base_color,
                edgecolor='black',
                alpha=0.8,
                label=f"Group {i+1}",
            )

        if added_cells:
            # Repaired (absorbed) cells in the paper's white-outline style.
            added_hex_gdf = hex_gdf.loc[list(added_cells)]
            added_hex_gdf.plot(
                ax=ax,
                color=base_color,
                edgecolor='white',
                alpha=0.95,
                linewidth=1.8,
                label=f"Group {i+1} patch",
            )

    ax.set_title("Zoning Visualization")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def save_four_zones_side_by_side(hex_gdf, zones, output_path):
    save_four_zones_side_by_side_with_patches(hex_gdf, zones, output_path)


def save_four_zones_side_by_side_with_patches(hex_gdf, zones, output_path, added_cells_by_zone=None):
    fig, axes = plt.subplots(2, 2, figsize=(12, 9.2), sharex=True, sharey=True)
    colors = list(mcolors.TABLEAU_COLORS.values())
    axes = axes.flatten()

    for i in range(4):
        ax = axes[i]
        row = i // 2
        col = i % 2
        hex_gdf.plot(ax=ax, color='lightgray', edgecolor='black', alpha=0.5)

        if i < len(zones):
            base_color = colors[i % len(colors)]
            added_cells = set() if added_cells_by_zone is None else set(added_cells_by_zone.get(i, []))
            original_cells = set(zones[i]) - added_cells

            if original_cells:
                zone_hex_gdf = hex_gdf.loc[hex_gdf.index.isin(original_cells)]
                zone_hex_gdf.plot(ax=ax, color=base_color, edgecolor='black', alpha=0.8)

            if added_cells:
                # Repaired (absorbed) cells in the paper's white-outline style.
                added_hex_gdf = hex_gdf.loc[hex_gdf.index.isin(added_cells)]
                added_hex_gdf.plot(
                    ax=ax,
                    color=base_color,
                    edgecolor='white',
                    alpha=0.95,
                    linewidth=1.8,
                )

            ax.set_title(f"Zone {i+1}")
        else:
            ax.set_title(f"Zone {i+1} (Empty)")

        # Camera-ready Figure 4 styling (tick fonts, 0.1-degree x ticks).
        ax.set_title(ax.get_title(), fontsize=18, pad=5)
        ax.tick_params(axis="both", labelsize=19)
        ax.tick_params(axis="x", labelbottom=True)
        ax.xaxis.set_major_locator(ticker.MultipleLocator(0.1))
        ax.xaxis.set_major_formatter(ticker.FormatStrFormatter("%.1f"))
        if col == 0:
            ax.tick_params(axis="y", labelleft=True, labelright=False)
            ax.yaxis.set_ticks_position("left")
        else:
            ax.tick_params(axis="y", labelleft=False, labelright=True)
            ax.yaxis.set_ticks_position("right")

    fig.subplots_adjust(
        left=0.075,
        right=0.995,
        bottom=0.08,
        top=0.94,
        wspace=0.09,
        hspace=0.14,
    )
    fig.supxlabel("Longitude", fontsize=19, y=0.028)
    fig.supylabel("Latitude", fontsize=19, x=0.008, y=0.53)
    plt.savefig(output_path, dpi=600, bbox_inches='tight', pad_inches=0.01)
    plt.close()


def visualize_four_zones_side_by_side(hex_gdf, zones, output_dir):
    """
    Visualize four optimal zones in a 2x2 grid and save the figure.

    Parameters:
        hex_gdf (GeoDataFrame): GeoDataFrame containing hexagon geometries.
        zones (list of sets): List of zones to visualize.
        output_dir (str): Directory the figure is saved into.
    """
    # Set up the figure and axes
    fig, axes = plt.subplots(2, 2, figsize=(12, 12), sharex=True, sharey=True)

    # Define colors for the zones
    colors = list(mcolors.TABLEAU_COLORS.values())

    # Flatten the axes for easier iteration
    axes = axes.flatten()

    # Plot each zone in a separate subplot
    for i in range(4):
        ax = axes[i]

        # Plot all hexagons in light gray as the base layer
        hex_gdf.plot(ax=ax, color='lightgray', edgecolor='black', alpha=0.5)

        if i < len(zones):
            # Highlight the current zone
            zone_hex_gdf = hex_gdf.loc[hex_gdf.index.isin(zones[i])]
            zone_hex_gdf.plot(ax=ax, color=colors[i % len(colors)], edgecolor='black', alpha=0.8)
            ax.set_title(f"Zone {i+1}", fontsize=14)
        else:
            # Leave the plot empty if there are fewer zones
            ax.set_title(f"Zone {i+1} (Empty)", fontsize=14)

        ax.set_xlabel("Longitude", fontsize=10)
        ax.set_ylabel("Latitude", fontsize=10)

    # Adjust layout and save the plot
    plt.tight_layout()
    plt.savefig(f"{output_dir}/four_zones_plot.png", dpi=300, bbox_inches='tight')
    plt.close()


def plot_demand_maps(hex_gdf, resolution, output_dir):
    """Render the in/out demand panel and the total-demand map for a run."""

    fig, axes = plt.subplots(1, 2, figsize=(20, 10))

    # Out demand plot
    hex_out_demand_plot = hex_gdf.plot(ax=axes[0], column='out_demand', cmap='viridis', legend=True,
                                legend_kwds={'label': "Out-Demand by Census Block", 'orientation': "horizontal"},
                                norm=PowerNorm(gamma=0.5))

    axes[0].set_title(f'Out-Demand Level by Resolution {resolution} H3 Hexagons', fontsize=20)
    axes[0].set_xlabel('Longitude', fontsize=15)
    axes[0].set_ylabel('Latitude', fontsize=15)

    # In demand plot
    hex_in_demand_plot = hex_gdf.plot(ax=axes[1], column='in_demand', cmap='viridis', legend=True,
                                legend_kwds={'label': "In-Demand by Census Block", 'orientation': "horizontal"},
                                norm=PowerNorm(gamma=0.5))

    axes[1].set_title(f'In-Demand Level by Resolution {resolution} H3 Hexagons', fontsize=20)
    axes[1].set_xlabel('Longitude', fontsize=15)
    axes[1].set_ylabel('Latitude', fontsize=15)

    plt.tight_layout()
    plt.savefig(f"{output_dir}/demand_in_out_res{resolution}.png", dpi=600, bbox_inches='tight')
    plt.close()

    # Total-demand heatmap in the paper's style (single canonical rendering,
    # shared with experiments/render_demand_map.py).
    plot_total_demand_map(hex_gdf, f"{output_dir}/demand_total_res{resolution}.png")


def plot_total_demand_map(hex_gdf, output_path):
    """Render the paper's aggregated total-demand heatmap for a city.

    Ported verbatim from the original notebook cell that produced the demand
    figure in the paper (viridis colormap, PowerNorm gamma 0.5, no title,
    unlabeled vertical colorbar, 600 dpi).
    """

    # Create total demand column
    hex_gdf['total_demand'] = hex_gdf['in_demand'] + hex_gdf['out_demand']

    # Set up figure and axis
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))

    # Define colormap and norm
    cmap = plt.get_cmap('viridis')
    norm = PowerNorm(gamma=0.5, vmin=hex_gdf['total_demand'].min(), vmax=hex_gdf['total_demand'].max())

    # Plot without automatic legend
    hex_gdf.plot(ax=ax, column='total_demand', cmap=cmap, norm=norm)

    # Manually add a tighter colorbar
    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm._A = []  # dummy array for colorbar
    cbar = fig.colorbar(sm, ax=ax, orientation='vertical', fraction=0.036, pad=0.05)
    cbar.ax.tick_params(labelsize=16)

    # Titles and labels
    ax.set_xlabel('Longitude', fontsize=20)
    ax.set_ylabel('Latitude', fontsize=20)
    ax.tick_params(axis='both', which='major', labelsize=16)

    plt.tight_layout()
    plt.savefig(output_path, dpi=600, bbox_inches='tight')
    plt.close()


def plot_distance_histograms(center_distances, norm_square_center_distances, resolution, output_dir):
    """Render the raw and normalized distance distribution histograms."""

    center_distances_list = [v for inner in center_distances.values() for v in inner.values() if not np.isinf(v)]

    plt.figure(figsize=(8, 5))
    plt.hist(center_distances_list, bins=100, edgecolor='black', alpha=0.7)  # Adjust bin count as needed
    plt.xlabel('Distance')
    plt.ylabel('Frequency')
    plt.title('Distance Distribution')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.savefig(f"{output_dir}/distance_distribution_res{resolution}.png", dpi=300, bbox_inches='tight')
    plt.close()

    norm_square_center_distances_list = [v for inner in norm_square_center_distances.values() for v in inner.values() if not np.isinf(v)]

    plt.figure(figsize=(8, 5))
    plt.hist(norm_square_center_distances_list, bins=100, edgecolor='black', alpha=0.7)  # Adjust bin count as needed
    plt.xlabel('Norm Power Distance')
    plt.ylabel('Frequency')
    plt.title('Norm Power Distance Distribution')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.savefig(f"{output_dir}/norm_distance_distribution_res{resolution}.png", dpi=300, bbox_inches='tight')
    plt.close()


def plot_demand_histogram(truncated_demand_list, resolution, output_dir):
    """Render the truncated (0 < demand <= 50) demand distribution histogram."""

    plt.figure(figsize=(8, 5))
    plt.hist(truncated_demand_list, bins=100, edgecolor='black', alpha=0.7)  # Adjust bin count as needed
    plt.xlabel('Demand')
    plt.ylabel('Frequency')
    plt.title('Demand Distribution')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.savefig(f"{output_dir}/demand_distribution_res{resolution}.png", dpi=300, bbox_inches='tight')
    plt.close()


def plot_column_size_distribution(column_sizes, output_dir):
    """Render the histogram of generated column sizes."""

    plt.figure(figsize=(8, 5))
    plt.hist(column_sizes, bins=20, edgecolor='black', alpha=0.7)
    plt.xlabel('Size of Columns (Number of Hexagons)')
    plt.ylabel('Frequency')
    plt.title('Distribution of Column Sizes')
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    # Save the figure to a file
    plt.savefig(f"{output_dir}/column_size_distribution.png", dpi=300, bbox_inches='tight')
    plt.close()

#!/usr/bin/env python3
"""Build cell-level equity scores for Chattanooga H3 resolution-8 cells.

This script uses official U.S. Census Bureau sources only:

1. ACS 5-year table-based summary files for:
   - B19013: median household income
   - B25044: tenure by vehicles available
2. TIGER/Line 2024 block group shapefiles.

It computes a cell-level equity score for each H3 cell in the supplied
Chattanooga hex map by:

1. Downloading ACS and block-group geometry data into a local cache.
2. Intersecting block groups with H3 cells.
3. Area-weighting household counts and averaging block-group median income.
4. Computing:
   - no-vehicle score
   - low-income score
   - combined equity score e_i = mu * no_vehicle + (1 - mu) * low_income

Outputs:
  - CSV with one row per H3 cell
  - GeoJSON with the same attributes attached to each cell geometry

By default the script reads the existing Chattanooga resolution-8 hex file from
the experiment codebase and writes outputs into the current folder.
"""

from __future__ import annotations

import argparse
import io
import sys
from pathlib import Path
from typing import Iterable

import geopandas as gpd
import pandas as pd
import requests


ACS_TABLE_URL = (
    "https://www2.census.gov/programs-surveys/acs/summary_file/"
    "{year}/table-based-SF/data/5YRData/acsdt5y{year}-{table}.dat"
)
TIGER_BG_URL = "https://www2.census.gov/geo/tiger/TIGER{year}/BG/tl_{year}_{state}_bg.zip"
TIGER_TRACT_URL = (
    "https://www2.census.gov/geo/tiger/TIGER{year}/TRACT/tl_{year}_{state}_tract.zip"
)

REPO_ROOT = Path(__file__).resolve().parent.parent

DEFAULT_HEX_FILE = REPO_ROOT / "data" / "hex_chatt_res8.geojson"
DEFAULT_CACHE_DIR = REPO_ROOT / "data" / "acs_equity_cache"
DEFAULT_OUTPUT_CSV = (
    REPO_ROOT / "output" / "equity" / "chattanooga_res8_equity_scores.csv"
)
DEFAULT_OUTPUT_GEOJSON = (
    REPO_ROOT / "output" / "equity" / "chattanooga_res8_equity_scores.geojson"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hex-file", type=Path, default=DEFAULT_HEX_FILE)
    parser.add_argument("--year", type=int, default=2024)
    parser.add_argument("--mu", type=float, default=0.5)
    parser.add_argument(
        "--score-mode",
        choices=["combined", "income_only"],
        default="combined",
        help="Use the combined income + no-vehicle score, or an income-only score.",
    )
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_OUTPUT_CSV)
    parser.add_argument("--output-geojson", type=Path, default=DEFAULT_OUTPUT_GEOJSON)
    parser.add_argument(
        "--states",
        nargs="+",
        default=["47", "13"],
        help="State FIPS codes whose block groups may overlap the Chattanooga hex map.",
    )
    return parser.parse_args()


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def download_file(url: str, destination: Path) -> Path:
    ensure_parent(destination)
    if destination.exists():
        return destination

    with requests.get(url, stream=True, timeout=180) as response:
        response.raise_for_status()
        with destination.open("wb") as fh:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    fh.write(chunk)
    return destination


def load_acs_table(cache_dir: Path, year: int, table: str) -> pd.DataFrame:
    table = table.lower()
    url = ACS_TABLE_URL.format(year=year, table=table)
    local_path = cache_dir / f"acsdt5y{year}-{table}.dat"
    download_file(url, local_path)
    return pd.read_csv(local_path, sep="|", dtype={"GEO_ID": "string"})


def load_census_geometries(
    cache_dir: Path, year: int, state_fips: str, geography: str
) -> gpd.GeoDataFrame:
    if geography == "block_group":
        url = TIGER_BG_URL.format(year=year, state=state_fips)
        local_zip = cache_dir / f"tl_{year}_{state_fips}_bg.zip"
    elif geography == "tract":
        url = TIGER_TRACT_URL.format(year=year, state=state_fips)
        local_zip = cache_dir / f"tl_{year}_{state_fips}_tract.zip"
    else:
        raise ValueError(f"Unsupported geography: {geography}")

    download_file(url, local_zip)
    return gpd.read_file(f"zip://{local_zip}")


def normalize_rank(series: pd.Series) -> pd.Series:
    valid = series.dropna().sort_values()
    if valid.empty:
        return pd.Series(index=series.index, dtype="float64")
    if len(valid) == 1:
        out = pd.Series(0.0, index=series.index, dtype="float64")
        out.loc[series.isna()] = pd.NA
        return out

    ranks = series.rank(method="average", ascending=True)
    normalized = (ranks - 1.0) / (len(valid) - 1.0)
    normalized.loc[series.isna()] = pd.NA
    return normalized


def load_and_prepare_sources(
    hex_file: Path,
    cache_dir: Path,
    year: int,
    states: Iterable[str],
    geography: str,
    score_mode: str,
) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    hexes = gpd.read_file(hex_file)
    if "id" not in hexes.columns:
        raise ValueError(f"Expected an 'id' property in {hex_file}.")
    hexes = hexes.rename(columns={"id": "cell_id"})
    hexes = hexes[["cell_id", "geometry"]].copy()

    geom_parts = []
    for state in states:
        geom_parts.append(load_census_geometries(cache_dir, year, state, geography))
    census_geoms = pd.concat(geom_parts, ignore_index=True)
    census_geoms = gpd.GeoDataFrame(census_geoms, geometry="geometry", crs=geom_parts[0].crs)

    minx, miny, maxx, maxy = hexes.total_bounds
    census_geoms = census_geoms.cx[minx:maxx, miny:maxy].copy()

    income = load_acs_table(cache_dir, year, "b19013")
    income = income[["GEO_ID", "B19013_E001"]].copy()
    if geography == "block_group":
        acs_prefix = "1500000US"
    elif geography == "tract":
        acs_prefix = "1400000US"
    else:
        raise ValueError(f"Unsupported geography: {geography}")

    income = income[income["GEO_ID"].str.startswith(acs_prefix, na=False)]
    income["median_income"] = pd.to_numeric(income["B19013_E001"], errors="coerce")
    income["acs_geoid"] = income["GEO_ID"]
    income = income[["acs_geoid", "median_income"]]

    if score_mode == "combined":
        vehicles = load_acs_table(cache_dir, year, "b25044")
        vehicles = vehicles[
            ["GEO_ID", "B25044_E001", "B25044_E003", "B25044_E010"]
        ].copy()
        vehicles = vehicles[vehicles["GEO_ID"].str.startswith(acs_prefix, na=False)]
        vehicles["households_total"] = pd.to_numeric(vehicles["B25044_E001"], errors="coerce")
        vehicles["owner_no_vehicle"] = pd.to_numeric(vehicles["B25044_E003"], errors="coerce")
        vehicles["renter_no_vehicle"] = pd.to_numeric(vehicles["B25044_E010"], errors="coerce")
        vehicles["households_no_vehicle"] = (
            vehicles["owner_no_vehicle"] + vehicles["renter_no_vehicle"]
        )
        vehicles["acs_geoid"] = vehicles["GEO_ID"]
        vehicles = vehicles[["acs_geoid", "households_total", "households_no_vehicle"]]
        attrs = income.merge(vehicles, on="acs_geoid", how="inner")
    else:
        attrs = income.copy()

    census_geoms["acs_geoid"] = acs_prefix + census_geoms["GEOID"].astype("string")
    census_geoms = census_geoms.merge(attrs, on="acs_geoid", how="inner")
    if score_mode == "combined":
        census_geoms = census_geoms.dropna(
            subset=["median_income", "households_total", "households_no_vehicle"]
        ).copy()
        census_geoms = census_geoms[
            (census_geoms["median_income"] > 0) & (census_geoms["households_total"] > 0)
        ].copy()
    else:
        census_geoms = census_geoms.dropna(subset=["median_income"]).copy()
        census_geoms = census_geoms[census_geoms["median_income"] > 0].copy()

    return hexes, census_geoms


def interpolate_scores(
    hexes: gpd.GeoDataFrame,
    census_geoms: gpd.GeoDataFrame,
    mu: float,
    score_source: str,
    score_mode: str,
) -> gpd.GeoDataFrame:
    if not (0.0 <= mu <= 1.0):
        raise ValueError(f"mu must lie in [0, 1], got {mu}.")

    projected_crs = hexes.estimate_utm_crs()
    if projected_crs is None:
        raise RuntimeError("Unable to estimate a projected CRS for area weighting.")

    hexes_proj = hexes.to_crs(projected_crs)
    census_proj = census_geoms.to_crs(projected_crs)

    census_proj["source_area"] = census_proj.geometry.area
    hexes_proj["cell_area"] = hexes_proj.geometry.area

    overlay_columns = ["GEOID", "median_income", "source_area", "geometry"]
    if score_mode == "combined":
        overlay_columns.extend(["households_total", "households_no_vehicle"])

    intersections = gpd.overlay(
        census_proj[overlay_columns],
        hexes_proj[["cell_id", "cell_area", "geometry"]],
        how="intersection",
        keep_geom_type=True,
    )
    intersections["intersection_area"] = intersections.geometry.area

    intersections = intersections[intersections["intersection_area"] > 0].copy()
    intersections["source_weight"] = (
        intersections["intersection_area"] / intersections["source_area"]
    )
    intersections["target_weight"] = intersections["intersection_area"] / intersections["cell_area"]

    intersections["income_weighted"] = intersections["median_income"] * intersections["target_weight"]
    if score_mode == "combined":
        intersections["households_total_aw"] = (
            intersections["households_total"] * intersections["source_weight"]
        )
        intersections["households_no_vehicle_aw"] = (
            intersections["households_no_vehicle"] * intersections["source_weight"]
        )

    agg_spec = {
        "target_weight_sum": ("target_weight", "sum"),
        "income_weighted_sum": ("income_weighted", "sum"),
    }
    if score_mode == "combined":
        agg_spec["households_total_cell"] = ("households_total_aw", "sum")
        agg_spec["households_no_vehicle_cell"] = ("households_no_vehicle_aw", "sum")

    aggregated = intersections.groupby("cell_id", as_index=False).agg(**agg_spec).copy()

    aggregated["median_income_cell"] = (
        aggregated["income_weighted_sum"] / aggregated["target_weight_sum"]
    )
    aggregated["income_rank"] = normalize_rank(aggregated["median_income_cell"])
    aggregated["low_income_score"] = 1.0 - aggregated["income_rank"]
    if score_mode == "combined":
        aggregated["no_vehicle_score"] = (
            aggregated["households_no_vehicle_cell"] / aggregated["households_total_cell"]
        )
        aggregated["vehicle_share"] = 1.0 - aggregated["no_vehicle_score"]
        aggregated["equity_score"] = (
            mu * aggregated["no_vehicle_score"] + (1.0 - mu) * aggregated["low_income_score"]
        )
    else:
        aggregated["households_total_cell"] = pd.NA
        aggregated["households_no_vehicle_cell"] = pd.NA
        aggregated["no_vehicle_score"] = pd.NA
        aggregated["vehicle_share"] = pd.NA
        aggregated["equity_score"] = aggregated["low_income_score"]
    aggregated["mu"] = mu
    aggregated["score_source"] = score_source
    aggregated["score_mode"] = score_mode

    keep_cols = [
        "cell_id",
        "households_total_cell",
        "households_no_vehicle_cell",
        "vehicle_share",
        "no_vehicle_score",
        "median_income_cell",
        "income_rank",
        "low_income_score",
        "equity_score",
        "mu",
        "score_source",
        "score_mode",
    ]
    aggregated = aggregated[keep_cols]

    result = hexes.merge(aggregated, on="cell_id", how="left")
    missing = int(result["equity_score"].isna().sum())
    if missing:
        print(
            f"Warning: {missing} cells did not receive an equity score.",
            file=sys.stderr,
        )
    return result


def fill_missing_from_fallback(
    primary: gpd.GeoDataFrame, fallback: gpd.GeoDataFrame
) -> gpd.GeoDataFrame:
    fill_cols = [
        "households_total_cell",
        "households_no_vehicle_cell",
        "vehicle_share",
        "no_vehicle_score",
        "median_income_cell",
        "income_rank",
        "low_income_score",
        "equity_score",
        "mu",
        "score_source",
        "score_mode",
    ]
    fallback_cols = ["cell_id"] + fill_cols
    fallback = fallback[fallback_cols].drop_duplicates(subset=["cell_id"]).copy()
    merged = primary.merge(
        fallback,
        on="cell_id",
        how="left",
        suffixes=("", "_fallback"),
    )
    for col in fill_cols:
        merged[col] = merged[col].fillna(merged[f"{col}_fallback"])
        merged = merged.drop(columns=f"{col}_fallback")
    return gpd.GeoDataFrame(merged, geometry="geometry", crs=primary.crs)


def fill_missing_with_nearest_neighbor(result: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    missing_mask = result["equity_score"].isna()
    if not missing_mask.any():
        return result

    projected_crs = result.estimate_utm_crs()
    if projected_crs is None:
        return result

    result_proj = result.to_crs(projected_crs)
    scored = result_proj[~missing_mask].copy()
    missing = result_proj[missing_mask][["cell_id", "geometry"]].copy()
    if scored.empty or missing.empty:
        return result

    scored_cols = [
        "households_total_cell",
        "households_no_vehicle_cell",
        "vehicle_share",
        "no_vehicle_score",
        "median_income_cell",
        "income_rank",
        "low_income_score",
        "equity_score",
        "mu",
        "score_mode",
    ]
    nearest = gpd.sjoin_nearest(
        missing,
        scored[["cell_id"] + scored_cols + ["geometry"]],
        how="left",
        lsuffix="missing",
        rsuffix="scored",
        distance_col="nearest_distance",
    )
    nearest = nearest.sort_values(["cell_id_missing", "nearest_distance"])
    nearest = nearest.drop_duplicates(subset=["cell_id_missing"]).copy()
    nearest = nearest.rename(columns={"cell_id_missing": "cell_id"})
    nearest["score_source"] = "nearest_neighbor_fallback"
    fill_cols = ["cell_id"] + scored_cols + ["score_source"]
    nearest = nearest[fill_cols]

    return fill_missing_from_fallback(result, nearest)


def write_outputs(result: gpd.GeoDataFrame, output_csv: Path, output_geojson: Path) -> None:
    ensure_parent(output_csv)
    ensure_parent(output_geojson)

    csv_df = pd.DataFrame(result.drop(columns="geometry"))
    csv_df = csv_df.sort_values("cell_id")
    csv_df.to_csv(output_csv, index=False)

    geojson_df = result.sort_values("cell_id").to_crs("EPSG:4326")
    geojson_df.to_file(output_geojson, driver="GeoJSON")


def build_equity_scores(
    hex_file: Path,
    cache_dir: Path,
    year: int,
    states: Iterable[str],
    mu: float,
    score_mode: str,
    output_csv: Path,
    output_geojson: Path,
) -> gpd.GeoDataFrame:
    """Build cell-level equity scores for an H3 hex map and write the outputs.

    Reusable core of the pipeline (city-agnostic): interpolates ACS block-group
    attributes to hex cells, falls back to tract level and nearest scored
    neighbor for uncovered cells, and writes the CSV/GeoJSON outputs.
    """

    cache_dir.mkdir(parents=True, exist_ok=True)

    print("Loading H3 cells...")
    hexes, block_groups = load_and_prepare_sources(
        hex_file=hex_file,
        cache_dir=cache_dir,
        year=year,
        states=states,
        geography="block_group",
        score_mode=score_mode,
    )
    print(f"Loaded {len(hexes)} H3 cells and {len(block_groups)} intersecting block groups.")

    print("Interpolating ACS attributes to H3 cells...")
    result = interpolate_scores(
        hexes,
        block_groups,
        mu=mu,
        score_source="block_group",
        score_mode=score_mode,
    )

    if result["equity_score"].isna().any():
        print("Filling missing cells from tract-level fallback...")
        _, tracts = load_and_prepare_sources(
            hex_file=hex_file,
            cache_dir=cache_dir,
            year=year,
            states=states,
            geography="tract",
            score_mode=score_mode,
        )
        missing_hexes = result[result["equity_score"].isna()][["cell_id", "geometry"]].copy()
        tract_fallback = interpolate_scores(
            missing_hexes,
            tracts,
            mu=mu,
            score_source="tract_fallback",
            score_mode=score_mode,
        )
        result = fill_missing_from_fallback(result, tract_fallback)

    if result["equity_score"].isna().any():
        print("Filling residual missing cells from nearest scored neighbor...")
        result = fill_missing_with_nearest_neighbor(result)

    print("Writing outputs...")
    write_outputs(result, output_csv, output_geojson)

    scored = result["equity_score"].notna().sum()
    print(f"Done. Wrote {scored} scored cells to:")
    print(f"  CSV:     {output_csv}")
    print(f"  GeoJSON: {output_geojson}")
    return result


def main() -> int:
    """Chattanooga-specific CLI wrapper around `build_equity_scores`."""

    args = parse_args()
    build_equity_scores(
        hex_file=args.hex_file,
        cache_dir=args.cache_dir,
        year=args.year,
        states=args.states,
        mu=args.mu,
        score_mode=args.score_mode,
        output_csv=args.output_csv,
        output_geojson=args.output_geojson,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Shared data-preparation helpers for the experiment entry points.

All functions preserve the exact preprocessing behavior of the original
entry points: H3 hex-grid construction over the geocoded city boundary, OD-demand
filtering/aggregation to hexagons, and loading/normalizing the pre-computed
center-to-center shortest-path distances.
"""

import json

import geopandas as gpd
import h3
import numpy as np
import osmnx as ox
import pandas as pd
from shapely.geometry import Point, Polygon
from tqdm import tqdm

from cg.utils import haversine_distance


CITY_FULL_NAMES = {
    "chatt": "Chattanooga, Tennessee, USA",
    "boston": "Boston, Massachusetts, USA",
    "atlanta": "Atlanta, Georgia, USA",
    "miami": "Miami, Florida, USA",
    "nashville": "Nashville, Tennessee, USA",
    "chicago": "Chicago, Illinois, USA",
    "la": "Los Angeles, California, USA",
}


def get_city_full_name(city):
    if city not in CITY_FULL_NAMES:
        raise ValueError(f"Unknown city: {city}")
    return CITY_FULL_NAMES[city]


def load_city_boundary(city):
    """Geocode the city and return its boundary polygon."""
    return ox.geocode_to_gdf(get_city_full_name(city)).geometry.iloc[0]


def build_hex_grid(G, boundary, resolution):
    """Map the city to the H3 system.

    Returns (hex_gdf, hexagons) where hex_gdf is indexed by hex id, carries
    centroid and nearest-road-network-node columns, and its geometry follows
    the (lon, lat) convention.
    """
    hexagons = h3.geo_to_cells(boundary, resolution)
    hex_polygons = []
    for hexagon in hexagons:
        hex_polygons.append(Polygon(h3.cell_to_boundary(hexagon)))

    # Create a GeoDataFrame from the hexagons
    hex_gdf = gpd.GeoDataFrame(geometry=hex_polygons)

    hex_gdf['id'] = hexagons
    hex_gdf.set_index('id', inplace=True)
    for hexid, row in hex_gdf.iterrows():

        # Calculate the centroid of each hexagon
        hex_gdf.at[hexid, 'centroid_lat'] = row.geometry.centroid.x
        hex_gdf.at[hexid, 'centroid_lon'] = row.geometry.centroid.y

        # Get the nearest node in the road network to the centroid of the hexagon
        center_node = int(ox.distance.nearest_nodes(G, float(row.geometry.centroid.y), float(row.geometry.centroid.x)))
        hex_gdf.at[hexid, 'center_node'] = center_node
        hex_gdf.at[hexid, 'center_node_lat'] = G.nodes[center_node]['y']
        hex_gdf.at[hexid, 'center_node_lon'] = G.nodes[center_node]['x']

    #* Reverse the order of lat and lon in the geometry for geo-operations and visualization (to respect the lon, lat convention)
    hex_gdf.geometry = hex_gdf.geometry.map(lambda geom: Polygon([(y, x) for x, y in geom.exterior.coords]))

    return hex_gdf, hexagons


def aggregate_demand_to_hexagons(raw_demand, boundary, hex_gdf, hexagons):
    """Filter raw OD demand and aggregate it to hexagons.

    Steps (unchanged from the original experiment scripts): drop very short trips
    (<500 m), drop trips outside the city boundary, spatially join origin and
    destination points to hexagons, and build the nested demand dictionary.
    Also annotates hex_gdf in place with 'out_demand' and 'in_demand' columns.
    """

    #* Step 2: Filter demand
    # Step 2-1: Demand that are very short (<500m in distance)
    raw_demand['dist_m'] = haversine_distance(
        raw_demand['origin_lat'], raw_demand['origin_lon'],
        raw_demand['destination_lat'], raw_demand['destination_lon']
    )
    demand_no_short_trip = raw_demand[raw_demand['dist_m'] >= 500].copy()

    # Step 2-2: Demand outside the city boundary
    def is_within_boundary(lat, lon, polygon):
        point = Point(lon, lat)
        return polygon.contains(point)

    tqdm.pandas()
    inbound_demand = demand_no_short_trip[
        demand_no_short_trip.progress_apply(lambda row: is_within_boundary(row['origin_lat'], row['origin_lon'], boundary) and
                                     is_within_boundary(row['destination_lat'], row['destination_lon'], boundary), axis=1)
    ]
    inbound_demand.reset_index(drop=True, inplace=True)

    #* Step 3: Aggregate the demand data to hexagons
    # Step 3-1: Create GeoDataFrame for demand points
    inbound_demand.loc[:, 'origin_geom'] = [Point(lon, lat) for lon, lat in zip(inbound_demand['origin_lon'], inbound_demand['origin_lat'])]
    inbound_demand.loc[:, 'dest_geom'] = [Point(lon, lat) for lon, lat in zip(inbound_demand['destination_lon'], inbound_demand['destination_lat'])]

    demand_gdf_origin = gpd.GeoDataFrame(inbound_demand, geometry='origin_geom', crs="EPSG:4269")[['origin_geom']]
    demand_gdf_dest = gpd.GeoDataFrame(inbound_demand, geometry='dest_geom', crs="EPSG:4269")[['dest_geom']]

    # Step 3-2: Spatial join for origins and destinations, repectively
    joined_origin = gpd.sjoin(demand_gdf_origin, hex_gdf, how="left", predicate='within')
    joined_dest = gpd.sjoin(demand_gdf_dest, hex_gdf, how="left", predicate='within')

    # Step 3-3: Data cleaning, remove demand that are problematic (e.g., located exactly on the boundary)
    #!: The sets of demand points removed due to unsuccessful match with any hexagon vary in different hex resolutions
    #!: Thus, we should always use the total demand points before data cleaning as a benchmark

    ori_index_removed = joined_origin[joined_origin['id'].isna()].index
    dest_index_removed = joined_dest[joined_dest['id'].isna()].index
    index_removed = ori_index_removed.union(dest_index_removed)

    joined_origin.drop(index_removed, inplace=True)
    joined_dest.drop(index_removed, inplace=True)

    # Step 3-4: Construct the demand dictionary
    demand = {i :{j: 0 for j in hexagons} for i in hexagons}
    for ori, dest in zip(joined_origin['id'], joined_dest['id']):
        #? Should we allow self-loop?
        # if ori != dest:
            demand[ori][dest] += 1

    #* Step 4: Add in-and-out-demand columns to hex_gdf for visualization)
    hex_gdf['out_demand'] = 0
    hex_gdf['in_demand'] = 0
    for i in demand:
        hex_gdf.loc[i, 'out_demand'] = sum(demand[i].values())
        for j in demand[i]:
            hex_gdf.loc[j, 'in_demand'] += demand[i][j]

    return demand


def load_center_distances(city, resolution):
    """Load the pre-computed center-to-center shortest-path distances."""
    with open(f'data/center_distances_{city}_res{resolution}.json', 'r') as f:
        return json.load(f)


def normalize_square_distances(center_distances, power=2, norm_max=1):
    """Normalize distances by the max finite distance and raise to `power`."""
    center_distances_list = [v for inner in center_distances.values() for v in inner.values() if not np.isinf(v)]
    max_distance = max(center_distances_list)

    norm_square_center_distances = {
        outer_key: {inner_key: ((norm_max * value / max_distance)**power if not np.isinf(value) else np.inf)
                    for inner_key, value in inner_dict.items()}
        for outer_key, inner_dict in center_distances.items()
    }
    return norm_square_center_distances

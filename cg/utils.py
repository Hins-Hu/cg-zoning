"""Shared utilities for the CG and CliqueGen experiments.

Holds the road-network-to-block-graph construction (construct_block_graph),
shortest-path distance helpers, the demand evaluators (evaluate_zoning and
the deduplicating calculate_total_demand_served used for all reported
coverage numbers), the haversine distance, and the Gurobi WLS license loader
(get_grb_license, credentials from .env or environment variables).
"""

import itertools

import numpy as np
import networkx as nx
import os
from rtree import index  # For efficient spatial adjacency search
import geopandas as gpd
from shapely.geometry import Point
from gurobipy import Env


def get_grb_license():
    """Build a Gurobi WLS ``Env`` from credentials held in the environment.

    Secrets are never hard-coded in source. The required variables are:

        GRB_WLSACCESSID, GRB_WLSSECRET, GRB_LICENSEID

    Copy ``.env.example`` to ``.env`` and fill in your own WLS license (the
    ``.env`` file is git-ignored), or export the variables in your shell.
    See the README for details.
    """

    # Load a local .env if python-dotenv is installed; otherwise fall back to
    # variables already exported in the environment.
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass

    access_id = os.environ.get("GRB_WLSACCESSID")
    secret = os.environ.get("GRB_WLSSECRET")
    license_id = os.environ.get("GRB_LICENSEID")

    if not (access_id and secret and license_id):
        raise RuntimeError(
            "Gurobi WLS credentials not found. Copy .env.example to .env and set "
            "GRB_WLSACCESSID / GRB_WLSSECRET / GRB_LICENSEID, or export them in "
            "your shell. See the README."
        )

    params = {
        "WLSACCESSID": access_id,
        "WLSSECRET": secret,
        "LICENSEID": int(license_id),
    }
    return Env(params=params)


def compute_shortest_path_distances(H, power = 2):
    
    """
    Compute shortest path distances raised to the power.
    Parameters:
    -----------
    H : networkx.DiGraph
        Directed graph.
    power : int
        Power to raise the shortest path distances.

    Returns:
    --------
    cost_dict : dict
        Dictionary of shortest path distances raised to the power
    """
    
    
    if H is not None:
        
        shortest_distances = dict(nx.all_pairs_dijkstra_path_length(H, weight='weight'))
        cost_dict = {key: {inner_key: value**power for inner_key, value in inner_dict.items()} 
                     for key, inner_dict in shortest_distances.items()}
    
    return cost_dict


def construct_block_graph(blocks, G):
    
    """ Construct a directed block graph H from the given blocks and road network G.

    INPUT:
        blocks: GeoDataFrame containing the following columns -
            'id', 'geometry', 'centroid_lat', 'centroid_lon', 'center_node', 'center_node_lat', 'center_node_lon'
        G: NetworkX graph representing the road network
    
    OUTPUT:
        H: NetworkX directed graph representing the block graph with edges weighted by \\
            shortest path distances between two center nodes in G
    """
    
    
    # Initialize directed block graph H
    H = nx.DiGraph()

    # Create nodes
    #* `center_node` is the node in G that is closest to the centroid of the block
    #* `nodes` is a set of nodes in G that are within the block
    for block_id, row in blocks.iterrows():
        H.add_node(block_id, centroid=(row['center_node_lat'], row['center_node_lon']), 
                geometry=row['geometry'], center_node=int(row['center_node']), nodes=set())

    # Build a spatial index for fast adjacency lookups
    spatial_idx = index.Index()
    for i, (block_id, row) in enumerate(blocks.iterrows()):
        spatial_idx.insert(i, row['geometry'].bounds, obj=block_id)

    # Use Rtree to efficiently construct the edge set
    #* This is only for visualization. We won't need this for the shortest path distance computation in the next step. 
    block_id_2_node = blocks['center_node'].to_dict()
    for i, (block_id_1, row1) in enumerate(blocks.iterrows()):
        possible_neighbors = spatial_idx.intersection(row1['geometry'].bounds, objects=True)
        for item in possible_neighbors:
            block_id_2 = item.object
            row2 = blocks.loc[block_id_2]
            if row1['center_node'] != row2['center_node'] and row1['geometry'].touches(blocks.loc[block_id_2, 'geometry']):
                node1, node2 = block_id_2_node[block_id_1], block_id_2_node[block_id_2]
                
                # Compute shortest path distance in G
                if nx.has_path(G, node1, node2):
                    weight_12 = nx.shortest_path_length(G, source=node1, target=node2, weight='weight')
                    H.add_edge(block_id_1, block_id_2, weight=weight_12)
                
                if nx.has_path(G, node2, node1):
                    weight_21 = nx.shortest_path_length(G, source=node2, target=node1, weight='weight')
                    H.add_edge(block_id_2, block_id_1, weight=weight_21)

    # Remove isolated blocks
    #TODO: Issue 1: The connectivity of blocks only depends on their center nodes, but that does not mean
    #TODO:       nodes other than the center one are not connected across blocks. We can't just rule out the chance of serving
    #TODO:       the demand in the isolated blocks    
    
    #TODO： Issue 2: We don't even need to remove isolated blocks, because an isolated block, if the demand aggregation level is high,
    #TODO:       can still be designed as a zone and demand-insde the block can be handled by a self-loop.
    # isolated_blocks = [block for block in list(H.nodes) if H.degree(block) == 0]
    # H.remove_nodes_from(isolated_blocks)
    
    
    # For each node in the block graph, map the node to a set of nodes in G within the block
    nodes_gdf = gpd.GeoDataFrame(
        {'node_id': list(G.nodes)},
        geometry=[Point(G.nodes[n]['x'], G.nodes[n]['y']) for n in G.nodes]
    )
    nodes_gdf.set_crs(blocks.crs, inplace=True)
    nodes_within_blocks = gpd.sjoin(nodes_gdf, blocks, predicate='within', how='right')
    block_2_nodes = nodes_within_blocks.groupby('id')['node_id'].apply(lambda x: set(x.dropna().astype(int))).to_dict()
    
    for block in H.nodes:
        H.nodes[block]['nodes'] = block_2_nodes[block] | set([H.nodes[block]['center_node']])
        
    return H


def evaluate_zoning(zones, demand):
    """
    Evaluate the total demand within the selected zones.

    Note: pairs that appear in multiple (overlapping) zones are counted once
    per zone. Use `calculate_total_demand_served` for the deduplicated
    evaluation reported in the paper.

    Input:
    - zones: List of selected zones, where each zone is a set of cell indices (e.g., hex id)
    - demand: Nested dictionary containing pairwise demand between cells
    Output:
    - total_demand: Intra-zone demand served
    """

    total_demand = 0
    for zone in zones:
        for i in zone:
            for j in zone:
                total_demand += demand[i][j]
    return total_demand


def calculate_total_demand_served(selected_zones, demand):
    """Total demand served by the selected zones, without double counting.

    Shared evaluator used by both the CG and the CliqueGen experiments: pairs
    that appear in multiple zones are counted once, and in-cell (self-loop)
    demand is added once per covered cell.
    """

    # Do not double count pairs existing in multiple zones
    valid_pairs = set()
    all_cells = set()

    for zone in selected_zones:
        for pair in itertools.permutations(zone, 2):
            valid_pairs.add(pair)
        all_cells.update(zone)

    total_demand = 0
    for i, j in valid_pairs:
        total_demand += demand[i][j]

    # Add in-cell demand (self-loops) without double counting
    for i in all_cells:
        total_demand += demand[i][i]

    return total_demand


def haversine_distance(lat1, lon1, lat2, lon2):
    R = 6371000  # Radius of Earth in meters
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2)**2
    return 2 * R * np.arcsin(np.sqrt(a))

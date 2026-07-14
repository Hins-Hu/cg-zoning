"""CliqueGen baseline algorithms (prior work, used for comparison only).

clique_generator_on_map enumerates all candidate zones (cliques of mutually
close cells under the single-zone budget, with convex-hull extension), and
solve_ILP selects the optimal subset under the global budget via a
max-coverage ILP. Driven by experiments/run_cliquegen.py for Table 1 and
Figure 3.
"""

import itertools
from shapely.geometry import Point, MultiPoint
import gurobipy as gp
from gurobipy import GRB, quicksum
from collections import defaultdict
from copy import deepcopy


# Check whether two nodes are close enough based on zone cost
def two_nodes_close(pair, distances, alpha, beta, budget):
    cost_forward = alpha * distances[pair[0]][pair[1]] + beta
    cost_backward = alpha * distances[pair[1]][pair[0]] + beta
    return (cost_forward <= budget) and (cost_backward <= budget)


# Check whether a node is close to a clique (i.e., a set of nodes close enough)
def is_node_close_to_clique(node, clique, distances, alpha, beta, budget):
    for n in clique:
        if not two_nodes_close((node, n), distances, alpha, beta, budget):
            return False
    return True


#*[Hins] I just realized that this does not exclude all cliques that are inside a convex hull of another clique
#*[Hins] because the order of visiting clique is arbitrary

#*[Hins] Another observation, when distances matrix is not symmetric (e.g. one-way roads exist),
#*[Hins] a node inside a convex hull of a clique geographically may not be added to that clique
def convex_hull_extend_on_map(clique, nodes, pos, pairwise_map, visited_cliques):
    
    # a tag
    is_extended = False
    
    # pos is calculated before hand
    points = []
    for node in clique:
        points.append(pos[node])

    # create the convex hull
    convex_hull = MultiPoint(points).convex_hull
    # create the set that will have all the points that have to be grouped together
    extend_set = deepcopy(clique)
    # checks for all the possible trips that could be encapsulated
    for node in set(nodes) - clique:
        
        # Filter out trips that are not sharable
        is_node_extendable = True
        for item in clique:
            if not pairwise_map[tuple(sorted((node, item)))]:
                is_node_extendable = False
                break
        if not is_node_extendable:
            continue

        # this means that the trip is encapsulated by the hull
        if convex_hull.contains(pos[node]):
            extend_set.add(node)
    if extend_set != clique:
        is_extended = True

    return is_extended, extend_set
    

def clique_generator_on_map(H, distances, alpha, beta, budget, connectivity_threshold, time_limit=None):

    # Initialization
    import time as time_module
    start_time = time_module.time()
    time_limit_reached = False
    
    nodes = list(H.nodes())
    shared_map = defaultdict(list)  # Will store tuples of (clique, diameter)
    shared_map[1] = [({n}, 0) for n in nodes]  # Single nodes have diameter 0
    visited_cliques = defaultdict(int)
    card = 2
    max_card = 2
    
    # Pre-computation for efficiency
    pos = {n: Point(H.nodes[n]['centroid']) for n in nodes}
    pairwise_map = defaultdict(int)
    for pair in itertools.combinations(nodes, 2):
        pair = tuple(sorted(pair))
        pairwise_map[pair] = two_nodes_close(pair, distances, alpha, beta, budget)
    
    while True:
        # Check time limit
        if time_limit is not None and (time_module.time() - start_time) > time_limit:
            print(f"Time limit of {time_limit}s reached at cardinality {card}")
            time_limit_reached = True
            break
            
        # Termination condition
        if card > max_card + 1:
            break

        prev_list = shared_map[card - 1]
        for clique, prev_diameter in prev_list:
            # Check time limit inside the loop for finer granularity
            if time_limit is not None and (time_module.time() - start_time) > time_limit:
                print(f"Time limit of {time_limit}s reached during cardinality {card} processing")
                time_limit_reached = True
                break
                
            for node in set(nodes) - clique:
                
                # Check if the new clique is visited already
                clique_key = tuple(sorted(clique | {node}))
                if visited_cliques[clique_key] == 1:
                    continue

                # Check if the new clique is valid when a new node is added
                if is_node_close_to_clique(node, clique, distances, alpha, beta, budget):
                    
                    # Compute new diameter incrementally
                    new_diameter = prev_diameter
                    for existing_node in clique:
                        new_diameter = max(new_diameter, distances[node][existing_node], distances[existing_node][node])
                        
                    is_extended, extended_clique = convex_hull_extend_on_map(clique | {node}, nodes, pos, pairwise_map, visited_cliques)
                    if is_extended:
                        extended_key = tuple(sorted(extended_clique))
                        if not visited_cliques[extended_key]:
                            # Compute diameter for extended clique
                            extended_diameter = new_diameter
                            for n1 in extended_clique - clique - {node}:
                                for n2 in clique | {node}:
                                    extended_diameter = max(extended_diameter, distances[n1][n2], distances[n2][n1])
                            
                            new_card = len(extended_clique)
                            shared_map[new_card].append((extended_clique, extended_diameter))
                            visited_cliques[extended_key] = 1
                            max_card = max(max_card, new_card)
                    else:
                        shared_map[card].append((clique | {node}, new_diameter))
                        clique_key = tuple(sorted(clique | {node}))
                        visited_cliques[clique_key] = 1
                        max_card = max(max_card, card)
                
                else:
                    # We also mark the invalid cliques as visited to avoid rechecking
                    clique_key = tuple(sorted(clique | {node}))
                    visited_cliques[clique_key] = 1
        
        # Break out of outer loop if time limit reached
        if time_limit_reached:
            break
                    
        print(f"Cardinality {card} has {len(shared_map[card])} cliques")
        print(f"Cardinality {card} complete")
        
        # Increase the cardinality
        card += 1
        
    # Extract the final list of cliques and compute their costs using tracked diameters
    clique_list = []
    clique_costs = {}
    for cliques_with_diameters in shared_map.values():
        for clique, diameter in cliques_with_diameters:
            clique_key = tuple(sorted(clique))
            clique_list.append(clique_key)
            clique_costs[clique_key] = alpha * diameter + beta
    
    #!Debugging
    seen = set()
    count = 0
    for clique in clique_list:
        key = tuple(sorted(clique))
        if key in seen:
            count += 1
        else:
            seen.add(key)
    print(f"{count} duplicates found")


    return clique_list, clique_costs, max_card, time_limit_reached

def solve_ILP(H, clique_list, clique_costs, demand, budget):
    
    # Initialization
    V = list(H.nodes())
    
    # Create index map to avoid long variable names
    clique_to_idx = {clique: idx for idx, clique in enumerate(clique_list)}
        
    # Pre-computation to extract the useful pairs of nodes
    valid_pairs = set()
    pair_2_clique = defaultdict(list)
    for clique in clique_list:
        for pair in itertools.permutations(clique, 2):
            valid_pairs.add(pair)
            pair_2_clique[pair].append(clique)
    
    # Gurobi license is read from the environment via cg.utils.get_grb_license
    # (see README / .env.example). Never hard-code license secrets here.
    from cg.utils import get_grb_license
    env = get_grb_license()
    
    # Add variables
    model = gp.Model(env=env)
    x = {}
    for clique in clique_list:
        x[clique] = model.addVar(vtype = GRB.BINARY, name=f"x_{clique_to_idx[clique]}")
    w = {}
    for i, j in valid_pairs:
        w[i, j] = model.addVar(vtype=GRB.BINARY, name=f"w_{i}_{j}")
    
    model.setObjective(quicksum(demand[i][j] * w[i, j] for i, j in valid_pairs), GRB.MAXIMIZE)
    
    # Constraints: linking x and w
    for i, j in valid_pairs:
        model.addConstr(
            w[i, j] <= quicksum(x[clique] for clique in pair_2_clique[(i, j)]),
            name=f"link_{i}_{j}"
        )

    # Constraint: Budget constraint (replaces max number of zones)
    model.addConstr(
        gp.quicksum(clique_costs[clique] * x[clique] for clique in clique_list) <= budget,
        name="budget_constraint"
    )
    
    # Optimization
    model.optimize()    
    if model.status == (GRB.OPTIMAL or GRB.SUBOPTIMAL):
        print("The max coverage ILP was solved to optimality.")
        selected_zones = [clique for clique in clique_list if x[clique].X > 0.5]
        print("Selected candidate zones:", selected_zones)

    else:
        print("Fail to solve the max coverage ILP.")
        
    return selected_zones
    
    


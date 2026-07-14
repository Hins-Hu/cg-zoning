"""Column Generation core for the Micro-Transit Zoning Problem.

column_generation drives the restricted master LP / pricing loop and the
final master ILP. Pricing can be solved exactly as an ILP (solve_pricing) or
an IQP (solve_pricing_QP), or approximately with the greedy heuristics
(pricing_heuristic for a best-pair start, pricing_heuristic_random for R
random restarts, Algorithm 1 in the paper). Gurobi models are built with the
WLS environment from cg.utils.get_grb_license.
"""

from collections import defaultdict
from gurobipy import Model, GRB, quicksum
import time
import numpy as np
from typing import List, Dict, DefaultDict, Tuple, Union, Optional, Any
from cg import utils


def pricing_heuristic_random(V, distances, lamb, pi, time_limit = 120, alpha=1, beta=1, num_runs = 1, single_zone_budget=None):
    """
    Inputs:
     - V: a list of nodes
     - distances: a nested dictionary where distances[i][j] returns the shortest path distance between nodes i and j
     - lamb: A scalar of the dual variable associated with the budget constraint
     - pi: A dictionary where pi[(i, j)] returns the dual variable for a pair of nodes (all required pairs exist)
     - time_limit: A positive integer representing the time limit for the pricing problem
     - alpha: The slope of the linear cost
     - beta: The intercept of the linear cost
     - num_runs: The number of runs for the random initialization
     - single_zone_budget: A positive integer representing the budget for the pricing to get a zone
    
    Raises:
     - ValueError: If no profitable pair is found
     
    Returns:
     - list_zone: A list of zones
     - list_reduced_cost: The reduced costs of zones
     - list_cost: The total costs of the zones
     - list_node_map: A mapping of zones to zero-one vectors of size |V|
    """
    
    # Column containers
    list_zone, list_reduced_cost, list_cost, list_node_map = [], [], [], []
    
    # Start timer outside the loop to respect total time limit
    t = time.time()
    
    for run_id in range(num_runs):
        
        # Check overall timeout before starting a new run
        if time.time() - t > time_limit:
            print(f"Timeout. Terminating the pricing heuristic algorithm.")
            break
        
        #* Step 1: Initialization
        # Find a random pair
        while True:
            # Error handling
            if run_id == 0 and time.time() - t > time_limit:
                raise ValueError("If you see this message, it means there is something wrong. The first pair should always be found.")

            i = np.random.choice(V)
            j = np.random.choice(V)
            if i != j and alpha * max(distances[i][j], distances[j][i]) + beta <= single_zone_budget:
                break
        i, j = str(i), str(j)
        S = {i, j}
        D = max(distances[i][j], distances[j][i])
        
        #* Step 2: Node Addition
        while True:
            
            # Timeout for pricing
            if time.time() - t > time_limit:
                print("Timeout. Terminating the pricing heuristic algorithm.")
                break
            
            max_delta = -float('inf')
            best_node = None
            new_D = D
            
            # Evaluate each node not in S
            for i in (set(V) - S):
                
                # Calculate diameter
                current_D = max(max(distances[i][j], distances[j][i]) for j in S)
                current_D = max(current_D, D)
                if single_zone_budget is not None and alpha * current_D + beta > single_zone_budget:
                    continue
                
                # Calculate added profit (sum of pi for new connections)
                added_profit = sum(pi[(i, j)] + pi[(j, i)] for j in S)
                
                # Calculate net gain (delta)
                delta = added_profit - lamb * alpha * (current_D - D)
                
                if delta > max_delta:
                    max_delta = delta
                    best_node = i
                    new_D = current_D
            
            # Break if no improvement found
            if max_delta <= 0:
                break
                
            # Add the best node
            S.add(best_node)
            D = new_D
        
        # Calculate cost and reduced cost
        cost = alpha * D + beta
        reduced_cost = sum(pi[(i, j)] for i in S for j in S if i != j) - lamb * cost
        
        # Check the reduced cost and add to the list of columns
        eps_4_degeneration = 1e-2
        if reduced_cost > eps_4_degeneration:
            list_zone.append(S)
            list_reduced_cost.append(reduced_cost)
            list_cost.append(cost)
            curr_node_map = {node: 1 if node in S else 0 for node in V}
            list_node_map.append(curr_node_map)
    
    # Keep the log for failure
    if len(list_zone) == 0:
        print("Pricing heuristic fails to find the seed pair with positive reduced cost.")
    
    return list_zone, list_reduced_cost, list_cost, list_node_map


def pricing_heuristic(V, distances, lamb, pi, time_limit = 120, alpha=1, beta=1, single_zone_budget=None):
    """
    Inputs:
     - V: a list of nodes
     - distances: a nested dictionary where distances[i][j] returns the shortest path distance between nodes i and j
     - lamb: A scalar of the dual variable associated with the budget constraint
     - pi: A dictionary where pi[(i, j)] returns the dual variable for a pair of nodes (all required pairs exist)
     - time_limit: A positive integer representing the time limit for the pricing problem
     - alpha: The slope of the linear cost
     - beta: The intercept of the linear cost
     - single_zone_budget: A positive integer representing the budget for the pricing to get a zone
    
    Raises:
     - ValueError: If no profitable pair is found
     
    Returns:
     - list_zone: A list of zones
     - list_reduced_cost: The reduced costs of zones
     - list_cost: The total costs of the zones
     - list_node_map: A mapping of zones to zero-one vectors of size |V|
    """
    
    #* Step 1: Initialization
    max_obj = -float('inf')
    best_pair = None
    t = time.time()
    
    # Find the pair with highest initial objective
    for i in V:
        for j in V:
            if i == j or alpha * distances[i][j] + beta > single_zone_budget:
                continue
            else:
                obj = pi[(i, j)] - lamb * (alpha * distances[i][j] + beta)
                if obj > max_obj:
                    max_obj = obj
                    best_pair = (i, j)

    # Check if a valid pair was found
    if best_pair is None:
        print("Pricing heuristic: No feasible pair exists within budget (infeasible problem).")
        return [], [], [], []
    
    # Assign the best pair to S
    i, j = best_pair
    S = {i, j}
    D = distances[i][j]
    
    #* Step 2: Node Addition
    while True:
        
        # Timeout for pricing
        if time.time() - t > time_limit:
            print("Timeout. Terminating the pricing heuristic algorithm.")
            break
        
        max_delta = -float('inf')
        best_node = None
        new_D = D
        
        # Evaluate each node not in S
        for i in (set(V) - S):
            
            # Calculate diameter
            current_D = max(D, max(distances[i][j] for j in S))
            if single_zone_budget is not None and alpha * current_D + beta > single_zone_budget:
                continue
            
            # Calculate added profit (sum of pi for new connections)
            added_profit = sum(pi[(i, j)] + pi[(j, i)] for j in S)
            
            # Calculate net gain (delta)
            delta = added_profit - lamb * alpha * (current_D - D)
            
            if delta > max_delta:
                max_delta = delta
                best_node = i
                new_D = current_D
        
        # Break if no improvement found
        if max_delta <= 0:
            break
            
        # Add the best node
        S.add(best_node)
        D = new_D
    
    # Calculate cost and reduced cost
    cost = alpha * D + beta
    reduced_cost = sum(pi[(i, j)] for i in S for j in S if i != j) - lamb * cost
    
    eps_4_degeneration = 1e-2
    if reduced_cost <= eps_4_degeneration:
        print("Pricing heuristic: No column with positive reduced cost found (optimality signal).")
        return [], [], [], []
    
    else:
        list_zone = [S]
        list_reduced_cost = [reduced_cost]
        list_cost = [cost]
        list_node_map = [{node: 1 if node in S else 0 for node in V}]
        
        return list_zone, list_reduced_cost, list_cost, list_node_map
    

#TODO: Not updated yet according to `solving_pricing`
def solve_pricing_QP(V, distances, lamb, pi, time_limit = 120, alpha=1, beta=1, single_zone_budget=None):
    
    # Initialization for columns returned
    list_zone, list_reduced_cost, list_cost, list_node_map = [], [], [], []

    # Create a model
    env = utils.get_grb_license()
    pricing = Model("pricing_QP", env = env)

    # The decision variables
    z = pricing.addVars(V, vtype=GRB.BINARY, name="z")
    diameter = pricing.addVar(vtype=GRB.CONTINUOUS, name="diameter", lb=0)
    
    # Set the objective
    pricing.setObjective(
        quicksum(pi[(i, j)] * z[i] * z[j] for i in V for j in V if i != j) - lamb * (alpha * diameter + beta),
        GRB.MAXIMIZE
    )
    
    # The constraints
    for i in V:
        for j in V:
            if i == j:
                continue
            # Skip pairs with infinite distance or that exceed the budget
            if np.isinf(distances[i][j]) or (single_zone_budget is not None and alpha * distances[i][j] + beta > single_zone_budget):
                # If distance is infinite or exceeds budget, prevent both nodes from being selected together
                pricing.addConstr(z[i] + z[j] <= 1, f"z_{i}_z_{j}_infeasible")
            else:
                pricing.addConstr(diameter >= distances[i][j] * z[i] * z[j], f"diameter_geq_shortest_distance_{i}_{j}")
    
    if single_zone_budget is not None:
        pricing.addConstr(alpha * diameter + beta <= single_zone_budget, "budget")
    
    # Gurobi parameters
    pricing.setParam('TimeLimit', time_limit)
    
    # Optimize the model
    pricing.optimize()
    
    # Solution extraction
    if pricing.SolCount > 0:
        
        reduced_cost = pricing.ObjVal
        node_map = {i:1 if z[i].X > 0.5 else 0 for i in V}
        selected_nodes = {i for i in V if z[i].X > 0.5}
        
        longest = 0
        for i in selected_nodes:
            for j in selected_nodes:
                if longest < distances[i][j]:
                    longest = distances[i][j]
        cost = alpha * longest + beta
        
        # Only add to list if reduced cost is positive
        eps_4_degeneration = 1e-2
        if reduced_cost > eps_4_degeneration:
            list_zone.append(selected_nodes)
            list_reduced_cost.append(reduced_cost)
            list_cost.append(cost)
            list_node_map.append(node_map)
        
        if pricing.status == GRB.OPTIMAL:
            print("The pricing problem is solved to optimality.")
        else:
            print("A feasible sub-optimal solution is found for the pricing problem.")
        
    else:
        print("No feasible solution found for the pricing problem.")

    print(f"Number of columns (with positive reduced cost) returned: {len(list_zone)}")
    return list_zone, list_reduced_cost, list_cost, list_node_map, pricing.status == GRB.OPTIMAL
    

def solve_pricing(V, distances, lamb, pi, time_limit = 120, alpha=1, beta=1,
                single_zone_budget=None,
                num_columns=1):
    
    """
    lamb: A scalar of the dual variable associated with the budget constraint
    pi: A dictionary where pi[(i, j)] returns the dual variable for a pair of nodes
    """
    
    # Initialization for columns returned
    list_zone, list_reduced_cost, list_cost, list_node_map = [], [], [], []

    # Create a model
    env = utils.get_grb_license()
    pricing = Model("pricing", env = env)

    # The decision variables
    # y = pricing.addVars(V, V, vtype=GRB.BINARY, name="y")
    y = pricing.addVars(
        ((i, j) for i in V for j in V if i != j and alpha * distances[i][j] + beta <= single_zone_budget),
        vtype=GRB.BINARY,
        name="y"
    )
    z = pricing.addVars(V, vtype=GRB.BINARY, name="z")
    diameter = pricing.addVar(vtype=GRB.CONTINUOUS, name="diameter", lb=0)

    # Set the objective
    pricing.setObjective(
        quicksum(pi[(i, j)] * y[i, j] for (i, j) in y.keys()) - lamb * (alpha * diameter + beta),
        GRB.MAXIMIZE
    )

    # The constraints
    for i in V:
        for j in V:
            if i == j:
                continue
            if alpha * distances[i][j] + beta > single_zone_budget:
                pricing.addConstr(z[i] + z[j] <= 1, f"z_{i} + z_{j}_leq_1")
            else:
                pricing.addConstr(y[i, j] <= z[i], f"y_{i}_{j}_leq_z_{i}")
                pricing.addConstr(y[i, j] <= z[j], f"y_{i}_{j}_leq_z_{j}")
                pricing.addConstr(y[i, j] >= z[i] + z[j] - 1, f"y_{i}_{j}_geq_z_{i}_{j}")
                pricing.addConstr(diameter >= distances[i][j] * y[i, j], f"diameter_geq_shortest_distance_{i}_{j}")

    # Gurobi parameters
    pricing.setParam('TimeLimit', time_limit)
    pricing.setParam('MIPFocus', 1) # Focus on finding feasible solutions
    # pricing.setParam('MIPFocus', 3) # Focus on improving the best feasible solution

    # Optimize the model
    pricing.optimize()
    
    # Solution extraction
    if pricing.SolCount > 0:
        
        num_columns = min(num_columns, pricing.SolCount)
        for sol_idx in range(num_columns):
            pricing.setParam('SolutionNumber', sol_idx)
            eps_4_degeneration = 1e-2
            if pricing.PoolObjVal < eps_4_degeneration:
                continue
            
            list_reduced_cost.append(pricing.PoolObjVal)
            list_node_map.append({i:1 if z[i].Xn > 0.5 else 0 for i in V})
            list_zone.append({i for i in V if z[i].Xn > 0.5})
        
            longest = 0
            for i in list_zone[-1]:
                for j in list_zone[-1]:
                    if longest < distances[i][j]:
                        longest = distances[i][j]
            list_cost.append(alpha * longest + beta)

        if pricing.status == GRB.OPTIMAL:
            print("The pricing problem is solved to optimality.")
        else:
            print("A feasible sub-optimal solution is found for the pricing problem.")
        
    else:
        print("No feasible solution found for the pricing problem.")

    print(f"Number of columns (with positive reduced cost) returned: {len(list_zone)}")
    return list_zone, list_reduced_cost, list_cost, list_node_map, pricing.status == GRB.OPTIMAL


def column_generation(
    V: List[any],
    budget: float,
    demand: Dict[any, Dict[any, float]],
    distances: Dict[any, Dict[any, float]],
    objective_demand: Dict[any, Dict[any, float]] | None = None,
    alpha: float = 1,
    beta: float = 1,
    cg_time_limit: int = 600,
    pricing_time_limit: int = 120,
    num_columns: int = 1,
    pricing_method: str = "exact_ilp",
    num_random_runs: int = 5,
    single_zone_budget: float = 2,
):
#TODO: Add a function signature for the output
    
    """
    V: a list of nodes
    budget: a positive integer
    demand: a nested dictionary where demand[i][j] returns the demand level between nodes i and j
    distances: a nested dictionary where distances[i][j] returns the shortest path distance between nodes i and j
    cg_time_limit: a positive integer representing the time limit for the column generation
    pricing_time_limit: a positive integer representing the time limit for the pricing problem
    single_zone_budget: a positive integer representing the budget for the pricing to get the a single zone
    """
    
    # Initialization
    #TODO: we may need to use a different strategy for the initial zone
    iter = 0
    t = time.time()
    if objective_demand is None:
        objective_demand = demand

    demand_flat = {(i, j): objective_demand[i][j] for i in objective_demand for j in objective_demand[i]}
    
    # Initialize time tracking variables
    total_pricing_time = 0
    num_pricing_solved = 0
    if pricing_method == "best_init" or pricing_method == "exact_ilp" or pricing_method == "exact_qp":
        list_zone, list_reduced_cost, list_cost, list_node_map = pricing_heuristic(V, distances, pi = demand_flat, lamb = 1,
                                                                    time_limit = pricing_time_limit,
                                                                    alpha=alpha, beta=beta,
                                                                    single_zone_budget = single_zone_budget)
    elif pricing_method == "random_init":
        list_zone, list_reduced_cost, list_cost, list_node_map = pricing_heuristic_random(V, distances, pi = demand_flat, lamb = 1,
                                                                    time_limit = pricing_time_limit,
                                                                    alpha=alpha, beta=beta,
                                                                    num_runs=num_random_runs,
                                                                    single_zone_budget = single_zone_budget)
    else:
        raise ValueError(f"Invalid pricing method: {pricing_method}")
    
    # Check if initial column was found
    if len(list_zone) == 0:
        raise ValueError("Failed to find initial column with positive reduced cost. Cannot start column generation.")
        
    columns = {iter: [list_zone[0], list_reduced_cost[0], list_cost[0], list_node_map[0]]}
    duals = defaultdict(dict)
    master_LP_objs = defaultdict(int)
    
    env = utils.get_grb_license()
    master_LP = Model("master_LP", env=env)

    #* Uncomment the following to use a specific LP solver
    # master_LP.setParam("Method", -1)

    # Variables    
    x = master_LP.addVars(columns.keys(), vtype=GRB.CONTINUOUS, lb = 0, ub = 1, name=f"x_{iter}")
    w = master_LP.addVars(V, V, vtype=GRB.CONTINUOUS, lb=0, ub = 1, name=f"w")
    for i in V:
        master_LP.remove(w[i, i])

    # Set the objective
    master_LP.setObjective(
        quicksum(objective_demand[i][j] * w[i, j] for i in V for j in V if i != j),
        GRB.MAXIMIZE,
    )
    
    # Add the constraints
    budget_constr = master_LP.addConstr(quicksum(columns[k][2] * x[k] for k in columns) <= budget, name=f"budget_constr_{iter}")
    link_constrs = {}
    for i in V:
        for j in V:
            if i == j:
                continue
            link_constrs[(i, j)] = master_LP.addConstr(
                w[i, j]
                <= quicksum(columns[k][3][i] * columns[k][3][j] * x[k] for k in columns),
                name=f"link_{i}_{j}_iter_{iter}",
            )
    
    # The main loop of column generation
    while True:
        
        # Timeout for CG
        if time.time() - t > cg_time_limit:
            print("Timeout. Terminating the column generation.")
            break
        
        # Solve the master LP
        master_LP.optimize()
        
        # Check degeneracy for each master LP
        num_degenerate_vars = sum(
            1 for v in master_LP.getVars() if v.VBasis == 0 and v.X == 0
        )
        print(f"Number of degenerate variables: {num_degenerate_vars}")
        
        
        # Record the master objectives
        master_LP_objs[iter] = master_LP.ObjVal
        
        # Extract the duals
        #* Set stable_weight to a positive value between 0 and 1 to stabilize the duals
        stable_weight = 0
        dual = {}
        tmp_lamb = budget_constr.Pi
        tmp_pi = {(i, j): link_constrs[(i, j)].Pi for (i, j) in link_constrs}
        if iter > 1:
            last_lamb = duals[iter-1]['lamb']
            dual["lamb"] = stable_weight * last_lamb + (1 - stable_weight) * tmp_lamb
            last_pi = duals[iter-1]['pi']
            dual["pi"] = {key: stable_weight * last_pi.get(key) + (1 - stable_weight) * tmp_pi.get(key) for key in tmp_pi}    
        else:
            dual['lamb'] = tmp_lamb
            dual["pi"] = tmp_pi
        duals[iter] = dual
            
        # Print the master LP optimality
        print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
        print(f"Iteration {iter}: Objective = {master_LP.ObjVal}")
        print(f"lambda = {dual['lamb']}" )
        print(f"sum_of_pi = {sum(dual['pi'].values())}")
        print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
            
        # Solve the pricing problem
        pricing_start_time = time.time()
        if pricing_method == "exact_ilp":
            list_zone, list_reduced_cost, list_cost, list_node_map, is_optimal = solve_pricing(V, distances, lamb = dual["lamb"], pi = dual["pi"],
                                                                        time_limit = pricing_time_limit,
                                                                        alpha=alpha, beta=beta,
                                                                        num_columns=num_columns,
                                                                        single_zone_budget=single_zone_budget)
        elif pricing_method == "best_init":
             list_zone, list_reduced_cost, list_cost, list_node_map = pricing_heuristic(V, distances, lamb = dual["lamb"], pi = dual["pi"],
                                                                    time_limit = pricing_time_limit,
                                                                    alpha=alpha, beta=beta,
                                                                    single_zone_budget=single_zone_budget)
        
        elif pricing_method == "random_init":
            list_zone, list_reduced_cost, list_cost, list_node_map = pricing_heuristic_random(V, distances, lamb = dual["lamb"], pi = dual["pi"],
                                                                    time_limit = pricing_time_limit,
                                                                    alpha=alpha, beta=beta,
                                                                    num_runs=num_random_runs,
                                                                    single_zone_budget=single_zone_budget)
        elif pricing_method == "exact_qp":
            list_zone, list_reduced_cost, list_cost, list_node_map, is_optimal = solve_pricing_QP(V, distances, lamb = dual["lamb"], pi = dual["pi"],
                                                                        time_limit = pricing_time_limit,
                                                                        alpha=alpha, beta=beta,
                                                                        single_zone_budget=single_zone_budget)
        else:
            raise ValueError(f"Invalid pricing method: {pricing_method}")
        
        pricing_time = time.time() - pricing_start_time
        
        # Track pricing time only if solution was found (not timed out)
        if len(list_zone) > 0:
            total_pricing_time += pricing_time
            num_pricing_solved += 1
        
        print("======================================================")
        if len(list_zone) == 0:
            print("The best column: []")
        else:
            print(f"The best column: {list_zone[0]}")
        print("======================================================")
        
        # Logging for the reasons of terminating column generation
        if (pricing_method == "best_init" or pricing_method == "random_init") and len(list_zone) == 0:
            print("No more zone with positive reduced cost is found by heuristic. Terminating the column generation.")
            break
        
        if (pricing_method == "exact_ilp" or pricing_method == "exact_qp") and len(list_zone) == 0 and not is_optimal:
            #?: Should we add a heuristic for backup?
            print("No more zone with positive reduced cost is found and optimality is NOT attained. Terminating the column generation.")
            break
        
        if (pricing_method == "exact_ilp" or pricing_method == "exact_qp") and len(list_zone) == 0 and is_optimal:
            print("No more zone with positive reduced cost is found at optimality. Complete column generation with proven optimality.")
            break
        
        # Add new columns to the master LP
        column_counter = len(columns)
        for idx, (zone, reduced_cost, cost, node_map) in enumerate(zip(list_zone, list_reduced_cost, list_cost, list_node_map)):
            column_id = column_counter + idx
            columns[column_id] = [zone, reduced_cost, cost, node_map]
            x[column_id] = master_LP.addVar(vtype=GRB.CONTINUOUS, lb=0, ub=1, name=f"x_{column_id}")
        
        # logging
        print(f"Number of columns added so far: {len(columns)}")
        
        master_LP.remove(budget_constr)
        budget_constr = master_LP.addConstr(quicksum(columns[i][2] * x[i] for i in columns) <= budget, name=f"budget_constr_{iter+1}")
        for i in V:
            for j in V:
                if i == j:
                    continue
                master_LP.remove(link_constrs[(i, j)])
                link_constrs[(i, j)] = master_LP.addConstr(
                    w[i, j]
                    <= quicksum(columns[k][3][i] * columns[k][3][j] * x[k] for k in columns),
                    name=f"link_{i}_{j}_iter_{iter+1}",
                )
        master_LP.update()
        
        # Update the iteration counter
        iter += 1
    
    # Solve the master ILP when the column generation is complete
    master_ilp = Model('master_ilp', env = env)
    x_ilp = master_ilp.addVars(columns.keys(), vtype=GRB.BINARY, name=f"x")
    w_ilp = master_ilp.addVars(V, V, vtype=GRB.BINARY, name=f"w")
    for i in V:
        master_ilp.remove(w_ilp[i, i])
    
    master_ilp.setObjective(
        quicksum(objective_demand[i][j] * w_ilp[i, j] for i in V for j in V if i != j),
        GRB.MAXIMIZE,
    )
    
    budget_constr = master_ilp.addConstr(quicksum(columns[i][2] * x_ilp[i] for i in columns) <= budget, name=f"budget_constr")
    link_constrs = {}
    for i in V:
        for j in V:
            if i == j:
                continue            
            link_constrs[(i, j)] = master_ilp.addConstr(w_ilp[i, j] <= quicksum(columns[k][3][i] * columns[k][3][j] * x_ilp[k] for k in columns), name=f"link_{i}_{j}")
    
    # Set time limit for master ILP
    #TODO: set it as a parameter
    master_ilp_time_limit = 600
    master_ilp.setParam('TimeLimit', master_ilp_time_limit)
    
    master_ilp.optimize()
    
    if master_ilp.status in [GRB.OPTIMAL, GRB.SUBOPTIMAL]:
        zones = []
        for i in x_ilp:
            if x_ilp[i].X > 0.5:
                zones.append(columns[i][0])

        print("The master ILP was solved to optimality.")
    else:
        print("No feasible solution found for the master ILP.")
    
    # Calculate and print average pricing time
    if num_pricing_solved > 0:
        avg_pricing_time = total_pricing_time / num_pricing_solved
    else:
        avg_pricing_time = 0
    
    print(f"\nAverage pricing time: {avg_pricing_time:.2f} seconds")
    print(f"Total pricing problems solved: {num_pricing_solved}")
    
    return zones, columns, duals, master_LP_objs, avg_pricing_time

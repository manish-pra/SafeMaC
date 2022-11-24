from copy import deepcopy

import networkx as nx
import numpy as np
import torch


def greedy_algorithm_opti_cov(density, graph, n_soln, disk_size, cov_mat):
    """Compute greedy solution on covariance of F (_cov). (_opti) The code is optimized to not scale linearly with number of agents

    Args:
        density (torch.Tensor Nx1): posterior mean of density
        graph (nx.DiGraph()): dynamics graph on which greedy solution needs to be evaluated e.g., pessi, opti
        n_soln (int): number of greedy solutions, typically number of agents
        disk_size (int): maximum number of connecting edges that an agent can travel from its current location
        cov_mat (torch.Tensor NxN): Covariance matrix

    Returns:
        idx_x_curr (list): list of agent index
        M_dist (list): list of coverage gain at each location over the whole domain
        ____  (list): list is marginal gain for each agent
    """
    # Greedy algorithm initializations
    idx_x_curr = []
    M_dist = []
    record_margin_gain = []
    non_covered_density = torch.empty_like(density).copy_(density)
    node_coverage_gain = {}

    # 1) Compute marginal gain at each node (cell) of the domain
    for node in graph.nodes:
        marginal_gain = coverage_oracle_cov(
            node, non_covered_density, graph, disk_size, cov_mat
        )
        node_coverage_gain[node] = marginal_gain

    # 2) Compute greedy solution, n_soln times
    for k in range(n_soln):
        # 2.1) Greedily pick the node and save it's location, marginal gain and remainin gain distribution over whole domain
        idx = max(node_coverage_gain, key=node_coverage_gain.get)
        idx_x_curr.append(idx)
        record_margin_gain.append(node_coverage_gain[idx])
        M_dist.append(node_coverage_gain)

        # 2.2) Recompute marinal gain, given 'k' agents has already been picked. This method is optimized to not recompute the unaffected locations
        if k < n_soln - 1:

            # 2.2.1) Mark covered location as 0 (since they do not provide new information)
            non_covered_density[
                list(
                    nx.single_source_shortest_path_length(
                        graph, idx_x_curr[-1], cutoff=disk_size
                    )
                )
            ] = 0.0

            # 2.2.2) Compute affected locations coordinates. All the locations upto 2* radius of last picked agent is affected
            affected_locs = list(
                nx.single_source_shortest_path_length(
                    graph, idx_x_curr[-1], cutoff=2 * disk_size
                )
            )

            # 2.2.3) Recompute marinal gain, given 'k' agents has already been picked.
            for node in affected_locs:
                marginal_gain = coverage_oracle_cov(
                    node, non_covered_density, graph, disk_size, cov_mat
                )
                node_coverage_gain[node] = marginal_gain

    return idx_x_curr, M_dist, torch.sum(torch.stack(record_margin_gain))


def coverage_oracle_cov(idx_x_curr, density, graph, disk_size, cov_mat):
    """ This method computes upper bound of the coverage utilising covariance matrix of density, as apposed to sum of ucb of density

    Args:
        idx_x_curr (list): node of current agent whose coverage upper bound is querried
        density (torch.Tensor Nx1): posterior mean of density
        graph (nx.DiGraph()): dynamics graph for getting list of covered nodes
        disk_size (int): maximum number of connecting edges that an agent can travel from its current location
        cov_mat (torch.Tensor NxN): Covariance matrix

    Returns:
        Fs (torch.Tensor 1x1): upper bound of the coverage function at the node
    """
    # 1) Get a list of nodes being covered
    covarage_area_idx = list(
        nx.single_source_shortest_path_length(
            graph, idx_x_curr, cutoff=disk_size)
    )

    # 2) \sum \mu + \sqrt (\sum \sigma_ii + \sigma_{ij})
    Fs = torch.sum(density[covarage_area_idx]) + torch.sqrt(
        torch.sum(cov_mat[covarage_area_idx][:, covarage_area_idx])
    )
    return Fs


def greedy_algorithm_opti(density, graph, n_soln, disk_size):
    """Compute greedy solution on sum of UCB. (_opti) The code is optimized to not scale linearly with number of agents

    Args:
        density (torch.Tensor Nx1): Upper confidence bound of density for whole domain
        graph (nx.DiGraph()): dynamics graph on which greedy solution needs to be evaluated e.g., pessi, opti
        n_soln (int): number of greedy solutions, typically number of agents
        disk_size (int): maximum number of connecting edges that an agent can travel from its current location

    Returns:
        idx_x_curr (list): list of agent index
        M_dist (list): list of coverage gain at each location over the whole domain
        ____  (list): list is marginal gain for each agent
    """
    # Greedy algorithm initializations
    idx_x_curr = []
    M_dist = []
    record_margin_gain = []
    non_covered_density = torch.empty_like(density).copy_(density)
    node_coverage_gain = {}

    # 1) Compute marginal gain at each node (cell) of the domain
    for node in graph.nodes:
        marginal_gain = coverage_oracle(
            node, non_covered_density, graph, disk_size)
        node_coverage_gain[node] = marginal_gain

    # 2) Compute greedy solution, n_soln times
    for k in range(n_soln):
        idx = max(node_coverage_gain, key=node_coverage_gain.get)
        idx_x_curr.append(idx)
        record_margin_gain.append(node_coverage_gain[idx])
        M_dist.append(node_coverage_gain)

        # 2.2) Recompute marinal gain, given 'k' agents has already been picked. This method is optimized to not recompute the unaffected locations
        if k < n_soln - 1:

            # 2.2.1) Mark covered location as 0 (since they do not provide new information)
            non_covered_density[
                list(
                    nx.single_source_shortest_path_length(
                        graph, idx_x_curr[-1], cutoff=disk_size
                    )
                )
            ] = 0.0

            # 2.2.2) Compute affected locations coordinates. All the locations upto 2* radius of last picked agent is affected
            affected_locs = list(
                nx.single_source_shortest_path_length(
                    graph, idx_x_curr[-1], cutoff=2 * disk_size
                )
            )

            # 2.2.3) Recompute marinal gain, given 'k' agents has already been picked.
            for node in affected_locs:
                marginal_gain = coverage_oracle(
                    node, non_covered_density, graph, disk_size
                )
                node_coverage_gain[node] = marginal_gain

    return idx_x_curr, M_dist, torch.sum(torch.stack(record_margin_gain))


def greedy_algorithm(density, graph, n_soln, disk_size):
    """Compute greedy solution on sum of UCB.

    Args:
        density (torch.Tensor Nx1): Upper confidence bound of density for whole domain
        graph (nx.DiGraph()): dynamics graph on which greedy solution needs to be evaluated e.g., pessi, opti
        n_soln (int): number of greedy solutions, typically number of agents
        disk_size (int): maximum number of connecting edges that an agent can travel from its current location

    Returns:
        idx_x_curr (list): list of agent index
        M_dist (list): list of coverage gain at each location over the whole domain
        ____  (list): list is marginal gain for each agent
    """
    # Greedy algorithm initializations
    idx_x_curr = []
    M_dist = []
    record_margin_gain = []
    max_marginal_gain = 0  # if n_sol=0, we still need to pass this value
    non_covered_density = torch.empty_like(density).copy_(density)

    # 1) Compute greedy solution, n_soln times
    for k in range(n_soln):
        # DON'T KNOW WHAT I MEANT HERE: increase the size of the curr S to include next element
        # 1.1) Initialize
        idx_x_curr.append(0)
        M_dist.append(density)
        max_marginal_gain = -np.inf

        # 1.2) Compute marginal gain at each node (cell) of the domain and pick the node which give maximum marginal gain
        for node in graph.nodes:
            idx_x_curr[-1] = node  # filling in the last due to append
            marginal_gain = coverage_oracle(
                node, non_covered_density, graph, disk_size)
            M_dist[-1][node] = marginal_gain
            if marginal_gain > max_marginal_gain:
                max_marginal_gain = marginal_gain
                best_pos_k = node

        # 1.3) Save the greedy solution for kth agent and record marginal gain
        idx_x_curr[-1] = best_pos_k
        record_margin_gain.append(max_marginal_gain)

        # 1.4) Mark covered location as 0 (since they do not provide new information)
        non_covered_density[
            list(
                nx.single_source_shortest_path_length(
                    graph, idx_x_curr[-1], cutoff=disk_size
                )
            )
        ] = 0.0

    return idx_x_curr, M_dist, torch.sum(torch.stack(record_margin_gain))


def coverage_oracle(idx_x_curr, density, graph, disk_size):
    """_summary_ This method computes upper bound of the coverage using sum of ucb of density

    Args:
        idx_x_curr (list): node of current agent whose coverage upper bound is querried
        density (torch.Tensor Nx1): Upper confidence bound of density for whole domain
        graph (nx.DiGraph()): dynamics graph for getting list of covered nodes
        disk_size (int): maximum number of connecting edges that an agent can travel from its current location

    Returns:
        Fs (torch.Tensor 1x1): upper bound of the coverage function at the node
    """
    # 1) Get a list of nodes being covered
    covarage_area_idx = list(
        nx.single_source_shortest_path_length(
            graph, idx_x_curr, cutoff=disk_size)
    )

    # 2) \sum \mu + \sum \sigma_ii
    Fs = torch.sum(density[covarage_area_idx])
    return Fs


def apply_goose(
    pessimistic_graph,
    optimistic_graph,
    grid_V,
    agent_param,
    common_param,
    Cx_model,
    xi_star,
):
    """_summary_: GoOSE algorithm as per https://arxiv.org/pdf/1910.13726.pdf

    Args:
        pessimistic_graph (_type_): _description_
        optimistic_graph (_type_): _description_
        grid_V (_type_): _description_
        agent_param (_type_): _description_
        common_param (_type_): _description_
        Cx_model (_type_): _description_
        xi_star (_type_): _description_

    Returns:
        _type_: _description_
    """
    # 4) Do safe expansion of the set
    # 4.1)Pick set of points from pessi, which can potentially reduce covariance(where we get some information)
    pessi_loc = grid_V[list(pessimistic_graph.nodes)]
    Sp_lower_Cx, Sp_upper_Cx = Cx_model.posterior(
        pessi_loc).mvn.confidence_region()
    Sp_lower_Cx, Sp_upper_Cx = Sp_lower_Cx.detach(), Sp_upper_Cx.detach()
    Sp_lower_Cx = (
        Sp_lower_Cx * (1 + agent_param["Cx_beta"]) / 2
        + Sp_upper_Cx * (1 - agent_param["Cx_beta"]) / 2
    )
    Sp_upper_Cx = (
        Sp_upper_Cx * (1 + agent_param["Cx_beta"]) / 2
        + Sp_lower_Cx * (1 - agent_param["Cx_beta"]) / 2
    )
    Wx_eps_t = (Sp_upper_Cx - Sp_lower_Cx) > common_param[
        "epsilon"
    ] / 2  # *params["agent"]["Cx_beta"]

    # 4.2) Pick some points based on some heuristics example distance metric
    # bool_opti = torch.where((V >= S_opti[agent_key].Xleft) & (
    #     V <= S_opti[agent_key].Xright), True, False)
    # bool_priority = bool_opti * ~bool_pessi

    priority_nodes = list(set(optimistic_graph.nodes) -
                          set(pessimistic_graph.nodes))
    A_priority = 1 / \
        (torch.sum(torch.abs(grid_V[priority_nodes] - xi_star), 1) + 1)
    A_x = grid_V[priority_nodes]
    # 4.3) Potential immediate expander set
    x_priority, idx_priority = A_priority.sort(descending=True)

    # fails if x_priority is an empty tensor
    print(x_priority.shape)
    found_a_expander = False
    for i in range(x_priority.shape[0]):
        G_expander = Sp_upper_Cx - agent_param["Lc"] * torch.norm(
            torch.abs(pessi_loc - A_x[idx_priority[i]]), dim=1
        )
        # print("idx_priority",idx_priority[i])
        if torch.any((G_expander > 0) * Wx_eps_t) == True:
            # print(G_expander)
            found_a_expander = True
            break
    if not found_a_expander:
        print("Fully explored, Expansion not possible")
        return xi_star, True  # no expansion possible, yet to think
    w_t = (Sp_upper_Cx - Sp_lower_Cx) * (G_expander > 0) * Wx_eps_t
    query_pt = pessi_loc[w_t.argmax()]
    # query_pt = V[bool_pessi][G_expander.argmax()]
    print("Goose pt", query_pt, "current uncertainity", w_t.max())
    # current bug: O goal is on extreme left 10, still coming on left of its pessimistic set, If Lc is low, then the func can't change rapidly, this implies point far away would also get impacted by sampling in pessi set. so it was all good, but Lc limitation
    fully_explored_status = False  # inverse of expander status
    return query_pt, fully_explored_status


# def greedy_algorithm(density, V, n_soln):
#     # Greedy algorithm
#     S_curr = []
#     M_dist = []
#     max_obj = 0  # id n_sol=0, we still need to pass this value
#     for k in range(n_soln):
#         # increase the size of the curr S to include next element
#         S_curr.append([0])
#         M_dist.append(np.random.random((V.shape[0], 1)))
#         max_obj = -np.inf
#         for i in range(V.shape[0]):
#             S_curr[-1] = [i]  # filling in the last due to append
#             obj_i = objective_oracle_given(S_curr, density)
#             M_dist[-1][i] = obj_i.detach().numpy()
#             if obj_i > max_obj:
#                 # print("The changed obj is for", k, "agent from", max_obj,
#                 #       "to", obj_ij, "Loc from", S_curr[-1], "to", [i, j])
#                 max_obj = obj_i

#                 best_pos_k = [i]
#         S_curr[-1] = best_pos_k

#     idx_xn_star = S_curr
#     xn_star = V[[idx_xn_star]].reshape(-1)
#     yn_star = density[[idx_xn_star]].reshape(-1)
#     return xn_star, yn_star, M_dist, max_obj


# def objective_oracle_given(S_curr, density):
#     Fs = 0
#     Rad_idx = 5  # 1,2,3,4,5  ..curr.. 5,4,3,2,1 #
#     covarage_area = density > np.inf  # intialize with false
#     for agent_idx in S_curr:
#         disk_left = np.max((agent_idx[0]-Rad_idx, 0))
#         disk_right = np.min((agent_idx[0]+Rad_idx, density.shape[0]-1))
#         covarage_area[disk_left:disk_right+1] = torch.ones(
#             disk_right+1-disk_left) < 2
#     Fs = torch.sum(covarage_area*density)
#     return Fs


def objective_oracle_given_pos(S_curr, density, V):
    Rad = 5 * 0.12
    Fs = 0
    agents_disk = []
    for agent_loc in S_curr:
        agent_loc = agent_loc[0]
        agent_left = agent_loc - Rad
        agent_right = agent_loc + Rad
        left_condn = torch.logical_or(
            V > agent_left, torch.isclose(V, agent_left))
        right_condn = torch.logical_or(
            V < agent_right, torch.isclose(V, agent_right))
        agents_disk.append(torch.logical_and(left_condn, right_condn))

    covarage_area = V > float("inf")  # intialize with false
    for agent_cover in agents_disk:
        covarage_area = torch.logical_or(covarage_area, agent_cover)

    Fs = torch.sum(covarage_area * density)
    return Fs


def d_func(x, y):
    return np.abs((x[0] - y[0]))


def get_nearest_agent(v, S):
    agent_index = 10
    min_dist = np.inf
    for k in S:
        temp = d_func(v, k)
        if temp < min_dist:
            min_dist = temp
            agent_index = k
    return agent_index, min_dist

from os import remove
import torch

import matplotlib.pyplot as plt
import matplotlib.animation as manimation
import numpy as np

from copy import copy, deepcopy
import pickle

import networkx.algorithms.isomorphism as iso
import networkx as nx

import itertools


def get_idx(V, positions):
    idx = []
    for position in positions:
        idx.append(torch.abs(torch.Tensor(V) - position).argmin().item())
    return idx


def idxfromloc(grid_V, loc):
    diff = grid_V - loc
    idx = torch.arange(grid_V.shape[0])[
        torch.isclose(diff, torch.zeros(2)).all(dim=1)
    ].item()
    return idx


def form_batches(players, init_safe):
    associate_dict, players = get_optimistic_intersection(players, init_safe)

    pessi_associate_dict, players = get_pessimistic_union(players, init_safe)
    return associate_dict, pessi_associate_dict, players


def get_associated_submodular_goal(players, associate_dict):
    xn_star_mat = []
    print("associate dict", associate_dict)
    exploit = {}
    for key in associate_dict:
        k = len(associate_dict[key])  # number of elements in the dict
        union_nodes = [
            list(players[agent].pessimistic_graph.nodes)
            for agent in associate_dict[key]
        ]
        union_nodes += [list(players[key].optimistic_graph.nodes)]
        union_nodes = list(set().union(*union_nodes))
        union_graph = players[key].base_graph.subgraph(union_nodes)
        for agent in associate_dict[key]:
            players[agent].update_union_graph(union_graph)
        xn_star, acq_density, M_dist, Fx_obj, exploit[key] = players[key].get_next_goal(
            k
        )
        xn_star_mat.append(xn_star)
    return xn_star_mat, acq_density, M_dist


def Update_disc_bound_goal(players, associate_dict, xn_star_mat):
    for key, xn_star in zip(associate_dict, xn_star_mat):
        for agent, xi_star in zip(associate_dict[key], xn_star):
            # Share the associate dict
            players[agent].get_expected_disc_loc(xi_star)
            players[agent].update_disc_boundary(xi_star)

    covered_nodes = []  # outside of optimistc set agent can't cover
    for key, player in enumerate(players):
        player.set_condi_disc_nodes(list(covered_nodes))
        covered_nodes = covered_nodes + player.full_disc_nodes

    list_meas_loc = []
    for key, xn_star in zip(associate_dict, xn_star_mat):
        for agent, xi_star in zip(associate_dict[key], xn_star):
            players[agent].set_others_meas_loc(
                list_meas_loc
            )  # to effi. calculate new loc
            players[agent].set_submodular_goal(xi_star)
            list_meas_loc.append(players[agent].planned_measure_loc)


def submodular_optimization(players, init_safe, params):
    if params["agent"]["use_goose"]:
        associate_dict, pessi_associate_dict, players = form_batches(players, init_safe)
    else:
        associate_dict = {}
        associate_dict[0] = []
        for idx in range(params["env"]["n_players"]):
            associate_dict[0].append(idx)
        pessi_associate_dict = associate_dict.copy()

    if params["agent"]["sol_domain"] == "pessi":
        xn_star_mat, acq_density, M_dist = get_associated_submodular_goal(
            players, pessi_associate_dict
        )
        Update_disc_bound_goal(players, pessi_associate_dict, xn_star_mat)
    else:
        xn_star_mat, acq_density, M_dist = get_associated_submodular_goal(
            players, associate_dict
        )
        Update_disc_bound_goal(players, associate_dict, xn_star_mat)

    return associate_dict, pessi_associate_dict, acq_density, M_dist


def get_pessimistic_union(players, init_safe):
    pessi_associate_dict = {}
    n = len(players)
    graph_set_dict = {}
    for agent_iter, player in enumerate(players):
        graph_set_dict[agent_iter] = set(player.pessimistic_graph)

    union = set.union(*[graph_set_dict[i] for i in range(n)])
    union_graph = players[agent_iter].base_graph.subgraph(np.asarray(list(union)))
    skip_list = []
    for agent_iter, agent_init in enumerate(init_safe["idx"]):
        if agent_iter not in skip_list:
            pessi_associate_dict[agent_iter] = []
            pessi_associate_dict[agent_iter].append(agent_iter)
            skip_list.append(agent_iter)
            edges = nx.algorithms.traversal.breadth_first_search.bfs_edges(
                union_graph, agent_init.item()
            )
            connected_nodes = [agent_init.item()] + [v for u, v in edges]
            players[agent_iter].pessimistic_graph = players[
                agent_iter
            ].base_graph.subgraph(np.asarray(list(connected_nodes)))

            for follower_iter in range(agent_iter + 1, n):
                if follower_iter not in skip_list:
                    if (
                        init_safe["idx"][follower_iter].item()
                        in players[agent_iter].pessimistic_graph.nodes()
                    ):
                        pessi_associate_dict[agent_iter].append(follower_iter)
                        players[follower_iter].pessimistic_graph = players[
                            agent_iter
                        ].pessimistic_graph.copy()
                        skip_list.append(follower_iter)

    return pessi_associate_dict, players


def get_optimistic_intersection(players, init_safe):
    # all_safe_nodes =
    # all_unsafe_nodes =
    unsafe_edges_set = set()

    old_safe_graph = players[0].centralized_safe_graph.copy()  # edges
    reachable_nodes = []

    for agent_key, player in enumerate(players):
        # optimistic_nodes = set.union(
        #     set(player.optimistic_graph.nodes), set(init_safe['idx'].tolist()))
        optimistic_nodes = set.union(
            set(player.optimistic_graph.nodes), {init_safe["idx"][agent_key].item()}
        )
        opti_complementary_nodes = (
            set(old_safe_graph.nodes) - optimistic_nodes
        )  # add init node here
        opti_complementary_graph = old_safe_graph.subgraph(
            np.asarray(list(opti_complementary_nodes))
        )
        opti_complementary_edges = opti_complementary_graph.edges
        unsafe_edges = set(old_safe_graph.edges) - set.union(
            set(player.optimistic_graph.edges), set(opti_complementary_edges)
        )
        unsafe_edges_set = set.union(unsafe_edges_set, unsafe_edges)
        reachable_nodes = reachable_nodes + list(player.optimistic_graph.nodes)
    if len(unsafe_edges_set) != 0:
        stop = 1
    all_safe_edges = set(old_safe_graph.edges) - unsafe_edges_set
    reachable_nodes = list(set(reachable_nodes))
    unreachable_nodes = list(set(players[0].base_graph.nodes) - set(reachable_nodes))

    new_safe_graph = old_safe_graph.edge_subgraph(list(all_safe_edges))
    optimistically_safe_nodes = list(set(new_safe_graph.nodes) - set(unreachable_nodes))
    # list(set(old_safe_graph) - set(optimistically_safe_nodes))
    definitely_unsafe_nodes = list(
        set(players[0].base_graph.nodes) - set(optimistically_safe_nodes)
    )
    print("definitely_unsafe_nodes", definitely_unsafe_nodes)
    for init_node, player in zip(init_safe["idx"], players):
        edges = nx.algorithms.traversal.breadth_first_search.bfs_edges(
            new_safe_graph, init_node.item()
        )
        connected_nodes = [init_node.item()] + [v for u, v in edges]
        player.optimistic_graph = new_safe_graph.subgraph(list(connected_nodes))
        player.update_centralized_unit(
            optimistically_safe_nodes,
            definitely_unsafe_nodes,
            new_safe_graph,
            unsafe_edges_set,
            unreachable_nodes,
        )

    # associate_dict logic
    associate_dict = {}
    remaining_agents = [i for i in range(len(init_safe["idx"]))]
    removing_list = []
    for key, player in enumerate(players):
        if key in remaining_agents:
            associate_dict[key] = []
            # associate_dict[key].append(key)
        for agent in remaining_agents:
            if init_safe["idx"][agent].item() in player.optimistic_graph.nodes:
                associate_dict[key].append(agent)
                removing_list.append(agent)
        remaining_agents = list(set(remaining_agents) - set(removing_list))

    # find all the edges in the optimistic set
    # can be disconnected graph as well, but don't worry has been taken into account in the above algo

    return associate_dict, players


# def get_optimistic_partition(players, init_safe):

#     opti_asso_dict = {}
#     n = len(players)
#     graph_set_list = []
#     rm_dict = {}
#     for agent_iter, player in enumerate(players):
#         graph_set_list.append(set(player.optimistic_graph))
#         rm_dict[agent_iter] = set(player.optimistic_graph)

#     rm_list = graph_set_list[:]

#     rm_idx = [i for i in range(n)]
#     not_found_niche = [i for i in range(n)]
#     rm_record = set()
#     i = -1
#     while len(rm_idx) - i > 2:
#         i += 1
#         combi_n_i = list(itertools.combinations(
#             rm_idx, len(rm_idx)-(i)))
#         # print(i, rm_idx, combi_n_i)
#         for combi_idx, combi in enumerate(combi_n_i):
#             intersect = set.intersection(*[rm_dict[i] for i in combi])
#             if len(intersect) != 0:
#                 # print(combi, intersect, rm_idx)
#                 for agent_iter in rm_idx:
#                     sub_inter = rm_dict[agent_iter] - \
#                         intersect  # subtracted intersection
#                     if init_safe["idx"][agent_iter].item() in intersect:
#                         key = i*10 + combi_idx + 100
#                         if key not in opti_asso_dict:
#                             opti_asso_dict[key] = {}
#                             opti_asso_dict[key]["idx"] = []
#                         opti_asso_dict[key]["idx"].append(
#                             agent_iter)  # record agent index
#                         opti_asso_dict[key]["graph"] = intersect
#                         not_found_niche.remove(agent_iter)

#                     rm_dict[agent_iter] = rm_dict[agent_iter] - intersect
#                     if len(rm_dict[agent_iter]) == 0:
#                         rm_record.add(agent_iter)
#         for rm_iter in reversed(sorted(rm_record)):
#             rm_idx.remove(rm_iter)  # remove from index
#             # remove the set
#             # rm_list.remove(rm_list[rm_iter])
#         rm_record = set()

#     for agent in not_found_niche:
#         opti_asso_dict[agent] = {}
#         opti_asso_dict[agent]["idx"] = [agent]
#         opti_asso_dict[agent]["graph"] = rm_dict[agent]

#     associate_dict = {}
#     for key in opti_asso_dict:
#         associate_dict[opti_asso_dict[key]["idx"]
#                        [0]] = opti_asso_dict[key]["idx"]
#         for agent_iter in opti_asso_dict[key]["idx"]:
#             players[agent_iter].optimistic_graph = players[agent_iter].optimistic_graph.subgraph(
#                 np.asarray(list(opti_asso_dict[key]["graph"])))  # can be disconnected graph as well, but don't worry has been taken into account in the above algo

#     return associate_dict, players

# def expansion_operator(V,  True_Constraint, init_set, thresh, Lc):
#     S_po_mat = []
#     S_po_prev = copy(init_set)
#     S_po = copy(S_po_prev)
#     S_po_mat.append(S_po_prev)
#     termin_condn = False
#     bound_left = True_Constraint[get_idx(V, [S_po_prev.Xleft])[0]].detach()
#     bound_right = True_Constraint[get_idx(V, [S_po_prev.Xright])[0]].detach()
#     set_bound = SafeSet(bound_left, bound_right, V, 0.12)
#     while not termin_condn:
#         bound_left = True_Constraint[get_idx(V,
#                                              [S_po_prev.Xleft])[0]].detach()
#         bound_right = True_Constraint[get_idx(V,
#                                               [S_po_prev.Xright])[0]].detach()
#         set_bound.Update(bound_left, bound_right)
#         # print((set_bound.Xleft-constraint)/Lc, (set_bound.Xright-constraint)/Lc)

#         S_po_left = S_po_prev.Xleft
#         for steps in range(100):
#             if set_bound.Xleft - Lc*(0.12)*steps < thresh:
#                 if steps == 0:
#                     break
#                 S_po_left = max(S_po_prev.Xleft - (0.12)
#                                 * (steps-1), V.min())
#                 break
#         # if V[min(V.shape[0]-1, get_idx(V, [S_po_prev.Xleft - (set_bound.Xleft-thresh)/Lc])[0])][0] < S_po_prev.Xleft:
#         #     S_po_left = V[get_idx(V, [
#         #         S_po_prev.Xleft - (set_bound.Xleft-thresh)/Lc])[0]][0]

#         S_po_right = S_po_prev.Xright
#         for steps in range(100):
#             if set_bound.Xright - Lc*(0.12)*steps < thresh:
#                 if steps == 0:
#                     break
#                 S_po_right = min(S_po_prev.Xright + (0.12)
#                                  * (steps-1), V.max())
#                 break
#         # if V[get_idx(V, [S_po_prev.Xright + (set_bound.Xright-thresh)/Lc])[0]][0] > S_po_prev.Xright:
#         #     S_po_right = V[get_idx(V, [
#         #         S_po_prev.Xright + (set_bound.Xright-thresh)/Lc])[0]][0]
#         S_po.Update(S_po_left.reshape(-1), S_po_right.reshape(-1))
#         termin_condn = ((S_po_prev.Xleft == S_po.Xleft)
#                         and (S_po_prev.Xright == S_po.Xright))
#         # print((S_po_prev.Xleft == S_po.Xleft), (S_po_prev.Xright == S_po.Xright))
#         # print(termin_condn, S_po, S_po_prev)
#         S_po_prev = copy(S_po)
#         S_po_mat.append(S_po_prev)

#     return S_po


def get_frame_writer():
    # FFMpegWriter = manimation.writers['ffmpeg']
    metadata = dict(title="Movie Test", artist="Matplotlib", comment="Movie support!")
    writer = manimation.FFMpegWriter(
        fps=3, codec="libx264", metadata=metadata
    )  # libx264 (good quality), mpeg4
    return writer


def SafelyExplore(
    players,
    params,
    associate_dict,
    pessi_associate_dict,
    iter,
    visu,
    init_safe,
    env,
    writer,
    fig,
    pt0,
    bool_IsGoalSafe,
):
    for agent_key in torch.arange(params["env"]["n_players"])[
        ~bool_IsGoalSafe
    ].tolist():
        # This is the goal
        xi_star = players[agent_key].get_next_to_go_loc()

        # obs_idx = idxfromloc(env.grid_V, xi_star)
        # if obs_idx in players[agent_key].pessimistic_graph.nodes or obs_idx not in players[agent_key].optimistic_graph.nodes:
        #     bool_IsGoalSafe[agent_key] = True

        reached_pt = players[agent_key].get_goose_goal(xi_star)
        players[agent_key].update_current_location(reached_pt)
        # xn_reached_dict["disc"][agent_key] = players[agent_key].disc_boundary

        # 3.1) Train and update after reaching the location
        TrainAndUpdateConstraint(reached_pt, agent_key, players, params, env)

        # 3.2) Update GP's of all the agents, has an effect on visu only
        for player, safe in zip(players, init_safe["idx"]):
            player.update_Cx_gp_with_current_data()

        print("xn reached point", reached_pt)
        # TODO: Think if it make sense to update all agents all sets with the data already communicated
        # 3.3) Update Visualization
        pt0 = UpdateSafeVisu(
            agent_key,
            players,
            visu,
            env,
            writer,
            associate_dict,
            pessi_associate_dict,
            fig,
            pt0,
        )
        # torch.all(torch.isclose(reached_pt, xi_star)):
    return players, associate_dict, pessi_associate_dict, pt0


def TrainAndUpdateConstraint(query_pt, agent_key, players, params, env):
    # 1) Fit a model on the available data based
    train = {}
    train["Cx_X"] = query_pt.reshape(-1, params["common"]["dim"])

    if params["agent"]["obs_model"] == "full_disc":
        disc_nodes = players[agent_key].get_expected_disc(
            idxfromloc(env.grid_V, train["Cx_X"])
        )
        train["Cx_Y"], train["Cx_X"] = env.get_disk_constraint_observation(disc_nodes)
    else:
        train["Cx_Y"] = env.get_constraint_observation(train["Cx_X"])

    players[agent_key].update_Cx_gp(train["Cx_X"], train["Cx_Y"])
    for i in range(params["env"]["n_players"]):
        if i is not agent_key:
            players[i].communicate_constraint([train["Cx_X"]], [train["Cx_Y"]])


def TrainAndUpdateDensity(query_pt, agent_key, players, params, env):
    # 1) Fit a model on the available data based
    train = {}
    train["Fx_X"] = query_pt.reshape(-1, params["common"]["dim"])

    if params["agent"]["obs_model"] == "full_disc":
        disc_nodes = players[agent_key].get_expected_disc(
            idxfromloc(env.grid_V, train["Cx_X"])
        )
        train["Fx_Y"], train["Fx_X"] = env.get_disk_density_observation(disc_nodes)
    else:
        train["Fx_Y"] = env.get_density_observation(train["Fx_X"])

    players[agent_key].update_Fx_gp(train["Fx_X"], train["Fx_Y"])
    players[agent_key].planned_disk_center_at_last_meas = train["Fx_X"][-1].reshape(
        -1, players[agent_key].env_dim
    )
    for i in range(params["env"]["n_players"]):
        if i is not agent_key:
            players[i].communicate_density([train["Fx_X"]], [train["Fx_Y"]])


def AllAgentsExpandOperator(S_pessi_in, agent_key, players, S_opti, S_pessi, lines, V):
    S_opti[agent_key], S_pessi[agent_key], lines[agent_key] = players[
        agent_key
    ].get_expanded_sets(
        V[S_opti[agent_key].StateInSet].view(-1, 1), S_pessi_in[agent_key]
    )
    return S_opti[agent_key], S_pessi[agent_key], lines


def UpdateSafeVisu(
    agent_key,
    players,
    visu,
    env,
    writer,
    associate_dict,
    pessi_associate_dict,
    f_handle,
    pt0,
):
    opti_boundary = {}
    for key in associate_dict:
        G = players[key].optimistic_graph
        if env.Ny == 1:
            opti_boundary[key] = [x for x in G.nodes() if (G.out_degree(x) <= 1)]
        else:
            opti_boundary[key] = [x for x in G.nodes() if (G.out_degree(x) <= 3)]

    pessi_boundary = {}
    for key in pessi_associate_dict:  # make a pessimistic associate dict here
        G = players[key].pessimistic_graph
        if env.Ny == 1:
            pessi_boundary[key] = [x for x in G.nodes() if (G.out_degree(x) <= 1)]
        else:
            pessi_boundary[key] = [x for x in G.nodes() if (G.out_degree(x) <= 3)]

    visu.CxVisuUpdate(
        players[agent_key].Cx_model,
        players[agent_key].current_location,
        pessi_boundary,
        opti_boundary,
        players[agent_key].get_Cx_data(),
        agent_key,
        players[agent_key].all_unsafe_nodes,
        players[agent_key].unsafe_edges_set,
        players[agent_key].unreachable_nodes,
    )  # All playes have communivated hence we can call this function with any agent key
    if pt0 != None:
        for t in pt0:
            t[0].remove()
    if env.Ny == 1:
        pt0 = visu.plot1Dsafe_GP(f_handle)
    else:
        pt0 = visu.plot_safe_GP(f_handle)
    # plt.savefig("curr-iter-plot.png")
    writer.grab_frame()

    return pt0


def UpdateCoverageVisu(
    agent_key, players, visu, env, acq_density, M_dist, writer, f_handle, pt1
):
    current_goal = {}
    # xn_planned_dict["measure"][agent_key]
    # it's the next goal, obs point
    current_goal["Fx_X"] = players[agent_key].get_next_to_go_loc()
    current_goal["Fx_Y"] = env.get_density_observation(current_goal["Fx_X"])[0]
    visu.FxUpdate(
        players[agent_key].Fx_model,
        current_goal,
        players[agent_key].disc_boundary,
        acq_density,
        M_dist,
        players[agent_key].get_Fx_data(),
        agent_key,
    )
    if pt1 != None:
        for t in pt1:
            t[0].remove()

    if env.Ny == 1:
        pt1 = visu.plot1Dobj_GP(f_handle)
    else:
        pt1 = visu.plot_Fx(f_handle)

    # print("current goal", [xn_planned_dict[agent_key] for agent_key in xn_planned_dict])
    # plt.savefig("curr-iter-plot.png")

    writer.grab_frame()
    return pt1


# take intersection of all agents ahowing true
# Check if agents lies in the intersected optimistic set
# If yes, then move forward with the next element.
# If the new element has a new player with true, intersect it will all and check for safe.
# Do it untill all the players are done.

# if connected then return intersection of optimistic set and union of pessi,
# else a saparate opti and pessi set with agent corresponds


def merge_Sopti(S_opti, params, V, init_safe):
    # 2) Solve the submodular problem and get a next point to go xi* in pessimistic safe set
    # if connected then return intersection of optimistic set and union of pessi,
    # else a saparate opti and pessi set with agent corresponds
    S_opti_copy = []
    for t in S_opti:
        S_opti_copy.append(copy(t))
    bool_opti_at_V_prev = (
        torch.ones(params["env"]["n_players"]) < 1
    )  # [False,False,False]
    # #######################################################################
    # FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF
    # FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF
    # FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF
    for i in range(V.shape[0]):
        opti_at_v = [
            S_opti_copy[k].StateInSet[i] for k in range(params["env"]["n_players"])
        ]
        bool_opti_at_V = torch.BoolTensor(opti_at_v).reshape(-1)
        if not (bool_opti_at_V == bool_opti_at_V_prev).all():
            z = V > -1000  # TTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTT
            # Take the intersection of all opti set that has
            t = [
                deepcopy(k.StateInSet) for k, res in zip(S_opti, bool_opti_at_V) if res
            ]
            for temp in t:
                z = torch.logical_and(z, temp)
                # z *= temp   # intersection of all optimistic sets
            # ##IMP z is always a connected set: temp in t always has one common element.
            for j, res in enumerate(bool_opti_at_V):
                if (
                    res == True
                ):  # if the current element is in the optimistic set of the agent
                    # left, right = (z*V)[0], (z*V)[-1]
                    left, right = V[z][0], V[z][-1]
                    if left <= init_safe[j][0] and init_safe[j][0] <= right:
                        S_opti_copy[j].UpdateByTensor(z)
                        # save the unique index of the agent, to know if they lie in same opti set
                        S_opti_copy[j].idx = i
                    else:
                        a = 1
                        # TODO: might have to change bool_opti_at_v to false for this j
            # if len(t) == params["env"]["n_players"] and a == 0: # Let it run fully
            #     break
            # if bool_opti_at_V is same and no new agent is added then no need to
            # enter the loop untill
        bool_opti_at_V_prev = bool_opti_at_V

    return S_opti_copy


def merge_S_pessi(S_pessi, params, V):
    S_pessi_copy = []
    for t in S_pessi:
        S_pessi_copy.append(copy(t))
    bool_pessi_at_V_prev = torch.ones(params["env"]["n_players"]) < 1
    for i in range(V.shape[0]):
        pessi_at_v = [
            S_pessi_copy[k].StateInSet[i] for k in range(params["env"]["n_players"])
        ]
        bool_pessi_at_V = torch.BoolTensor(pessi_at_v).reshape(-1)
        if not (bool_pessi_at_V == bool_pessi_at_V_prev).all():
            z = V < -1000
            # Take the intersection of all pessi set that has
            t = [k.StateInSet for k, res in zip(S_pessi, bool_pessi_at_V) if res]
            for temp in t:
                z = torch.logical_or(z, temp)
                # z = z + temp
            for j, res in enumerate(bool_pessi_at_V):
                if (
                    res == True
                ):  # if the current element is in the pessimistic set of the agent
                    S_pessi_copy[j].UpdateByTensor(z)
                    S_pessi_copy[j].idx = i
                    # S_pessi set stay the same no change
            # if len(t) == params["env"]["n_players"] and a == 0: # Let it run fully
            #     break
            # if bool_pessi_at_V is same and no new agent is added then no need to
            # enter the loop untill
        bool_pessi_at_V_prev = bool_pessi_at_V

    return S_pessi_copy


# def goose(players, ):
#     # keep on expanding untill the point xi_star is in pessimistic set or outside of the safe set
#     count = 0
#     status = True
#     # Exploration is complete if optimistic set is larger than pessimistic in all sense,
#     converged = not len(set(players[agent_key].optimistic_graph.nodes) - set(
#         players[agent_key].pessimistic_graph.nodes))
#     # check if xi_star is in pessimistic set or not?
#     target_in_pessi = (env.idxfromloc(
#         xi_star) in players[agent_key].pessimistic_graph.nodes)
#     while count < 1 and (not target_in_pessi):
#         count += 1
#         print("count", count)
#         reached_pt, status = apply_goose(
#             players[agent_key].pessimistic_graph, players[agent_key].optimistic_graph, grid_V, agent_key, params, players[agent_key].Cx_model, xi_star)
#     if status == True:
#         players[agent_key].update_current_location(reached_pt)
#         xn_reached_dict[agent_key] = reached_pt
#         # May be try making update simultaneous instead of sequential
#         TrainAndUpdate(
#             reached_pt, agent_key, players, params, grid_V, env)

#         # S_opti[agent_key], S_pessi[agent_key], lines = AllAgentsExpandOperator(
#         #     S_pessi, agent_key, players, S_opti, S_pessi, lines, V)  # make lines also wrt agent key
#         for player, safe in zip(players, init_safe["idx"]):
#             player.update_Cx_gp_with_current_data()
#             player.update_Fx_gp_with_current_data()
#     else:
#         # S_opti[agent_key] = S_pessi[agent_key]
#         xn_reached_dict[agent_key] = players[agent_key].current_location


def apply_goose_old(S_pessi, S_opti, V, agent_key, params, Cx_model, xi_star):
    # 4) Do safe expansion of the set
    # 4.1)Pick set of points from pessi, which can potentially reduce covariance(where we get some information)
    bool_pessi = torch.where(
        (V >= S_pessi[agent_key].Xleft) & (V <= S_pessi[agent_key].Xright), True, False
    )
    Sp_lower_Cx, Sp_upper_Cx = (
        Cx_model[agent_key].posterior(V[bool_pessi]).mvn.confidence_region()
    )
    Sp_lower_Cx = (
        Sp_lower_Cx * (1 + params["agent"]["Cx_beta"]) / 2
        + Sp_upper_Cx * (1 - params["agent"]["Cx_beta"]) / 2
    )
    Sp_upper_Cx = (
        Sp_upper_Cx * (1 + params["agent"]["Cx_beta"]) / 2
        + Sp_lower_Cx * (1 - params["agent"]["Cx_beta"]) / 2
    )
    Wx_eps_t = (Sp_upper_Cx - Sp_lower_Cx) > params["common"][
        "epsilon"
    ]  # *params["agent"]["Cx_beta"]

    # 4.2) Pick some points based on some heuristics example distance metric
    bool_opti = torch.where(
        (V >= S_opti[agent_key].Xleft) & (V <= S_opti[agent_key].Xright), True, False
    )
    bool_priority = bool_opti * ~bool_pessi
    A_priority = 1 / (torch.abs(V[bool_priority] - xi_star) + 1)
    A_x = V[bool_priority]
    # 4.3) Potential immediate expander set
    x_priority, idx_priority = A_priority.sort(descending=True)

    # fails if x_priority is an empty tensor
    print(x_priority.shape)
    found_a_expander = False
    for i in range(x_priority.shape[0]):
        G_expander = Sp_upper_Cx - params["agent"]["Lc"] * torch.abs(
            V[bool_pessi] - A_x[idx_priority[i]]
        )
        # print("idx_priority",idx_priority[i])
        if torch.any((G_expander > 0) * Wx_eps_t) == True:
            # print(G_expander)
            found_a_expander = True
            break
    if not found_a_expander:
        print("Expansion not possible")
        return xi_star, False  # no expansion possible, yet to think
    w_t = (Sp_upper_Cx - Sp_lower_Cx) * (G_expander > 0) * Wx_eps_t
    query_pt = V[bool_pessi][w_t.argmax()]
    # query_pt = V[bool_pessi][G_expander.argmax()]
    print(
        "Goose pt", query_pt, "for agent ", agent_key, "current uncertainity", w_t.max()
    )
    # current bug: O goal is on extreme left 10, still coming on left of its pessimistic set, If Lc is low, then the func can't change rapidly, this implies point far away would also get impacted by sampling in pessi set. so it was all good, but Lc limitation
    expander_status = True
    return query_pt, expander_status


def save_data_plots(
    list_FxIX_rho_opti,
    list_FtildexIX_rho_opti,
    list_FxIX_lcb_pessi,
    list_FxlIX_lcb_pessi,
    list_FxlIX_pessi_rho_Rbar0,
    list_sum_max_density_sigma,
    list_FxIX_rho_Rbar_eps,
    list_FxIX_rho_Rbar0,
    opt_Fx_obj,
    exploit_record,
    nodes,
    samples,
    normalization_factor,
    save_path,
):
    data_dict = {}

    mat_FxIX_rho_opti = torch.vstack(list_FxIX_rho_opti)
    mat_FtildexIX_rho_opti = torch.vstack(list_FtildexIX_rho_opti)
    mat_FxlIX_lcb_pessi = torch.vstack(list_FxlIX_lcb_pessi)
    mat_FxlIX_pessi_rho_Rbar0 = torch.vstack(list_FxlIX_pessi_rho_Rbar0)
    mat_FxIX_rho_Rbar_eps = torch.vstack(list_FxIX_rho_Rbar_eps)
    mat_FxIX_rho_Rbar0 = torch.vstack(list_FxIX_rho_Rbar0)
    mat_FxIX_lcb_pessi = torch.vstack(list_FxIX_lcb_pessi)
    std = torch.std(mat_FxIX_rho_opti, 1)
    mean = torch.mean(mat_FxIX_rho_opti, 1)

    data_dict["opt_Fx"] = opt_Fx_obj.detach()
    data_dict["mat_FxIX_rho_opti"] = mat_FxIX_rho_opti
    data_dict["mat_FtildexIX_rho_opti"] = mat_FtildexIX_rho_opti
    data_dict["mat_FxlIX_lcb_pessi"] = mat_FxlIX_lcb_pessi.detach()
    data_dict["mat_FxIX_lcb_pessi"] = mat_FxIX_lcb_pessi
    data_dict["mat_FxlIX_pessi_rho_Rbar0"] = mat_FxlIX_pessi_rho_Rbar0
    data_dict["mat_FxIX_rho_Rbar_eps"] = mat_FxIX_rho_Rbar_eps
    data_dict["mat_FxIX_rho_Rbar0"] = mat_FxIX_rho_Rbar0
    data_dict["exploit_record"] = exploit_record
    data_dict["mean"] = mean
    data_dict["std"] = std
    data_dict["nodes"] = nodes
    data_dict["samples"] = samples
    data_dict["list_sum_max_density_sigma"] = (
        torch.stack(list_sum_max_density_sigma).detach().numpy()
    )
    data_dict["normalization_factor"] = normalization_factor

    take_greedy_lcb = data_dict["mat_FxIX_lcb_pessi"] < data_dict["mat_FxlIX_lcb_pessi"]
    picking_values = data_dict["mat_FxlIX_lcb_pessi"] * take_greedy_lcb + data_dict[
        "mat_FxIX_lcb_pessi"
    ] * (~take_greedy_lcb)
    true_obj_rbar0 = data_dict[
        "mat_FxlIX_pessi_rho_Rbar0"
    ] * take_greedy_lcb + data_dict["mat_FxIX_rho_Rbar0"] * (~take_greedy_lcb)
    obj_at_curr_max = true_obj_rbar0[0]
    curr_max = picking_values[0]
    resulting_mat = true_obj_rbar0.clone()
    for idx, val in enumerate(picking_values):
        if val > curr_max:
            curr_max = picking_values[idx].clone()
            obj_at_curr_max = true_obj_rbar0[idx].clone()
            resulting_mat[idx] = true_obj_rbar0[idx].clone()
        else:
            resulting_mat[idx] = obj_at_curr_max.clone()
    data_dict["mat_recommendation_rho_Rbar0"] = resulting_mat

    tildex_mean = torch.mean(mat_FtildexIX_rho_opti, 1)

    a_file = open(save_path + "/data.pkl", "wb")
    pickle.dump(data_dict, a_file)
    a_file.close()
    plt.close()
    plt.errorbar(np.arange(mean.shape[0]), mean, yerr=std, label="Fx_t")
    plt.grid(True)
    plt.legend()
    plt.savefig(save_path + "/Fx.png")

    plt.close()
    plt.plot(
        np.arange(data_dict["list_sum_max_density_sigma"].shape[0]),
        data_dict["list_sum_max_density_sigma"],
        label="max-sigma",
    )
    plt.grid(True)
    plt.legend()
    plt.savefig(save_path + "/sigma.png")
    return data_dict

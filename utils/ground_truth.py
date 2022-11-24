from utils.datatypes import SafeSet
from utils.central_graph import CentralGraph, expansion_operator
from utils.agent_helper import greedy_algorithm, greedy_algorithm_opti, coverage_oracle
from utils.helper import idxfromloc
import time
import torch
import networkx as nx
import numpy as np


class GroundTruth:
    def __init__(self, env, params) -> None:
        self.opt_goal = {}
        self.opt_val = torch.zeros(1)
        self.true_associate_dict = {}
        self.optimal_feasible_boundary = {}
        self.optimal_graphs_eps = []
        self.optimal_graphs0 = []
        self.env_data = env.env_data
        self.true_density = env.get_true_objective_func()
        self.true_constraint_function = env.get_true_safety_func()
        self.init_safe = env.get_safe_init()
        self.params = params
        self.grid_V = env.grid_V

    def __load_opt_graphs(self):
        if "s_opt" in self.env_data:
            print("Loading from env data")
            for agent_i in self.env.env_data["s_opt"]:
                self.optimal_graphs_eps.append(self.env.env_data["s_opt"][agent_i])
        else:
            print("Calculating opt set, env sopt not available")
            optimal_graph = CentralGraph(env_params=self.params["env"])
            Safe = []
            for i in range(self.params["env"]["n_players"]):
                # Safe.append(SafeSet(self.init_safe[i], self.init_safe[i], V))
                if self.params["agent"]["use_goose"]:
                    out = expansion_operator(
                        optimal_graph.graph,
                        self.true_constraint_function
                        - self.params["common"]["epsilon"],
                        self.init_safe["idx"][i],
                        self.params["common"]["constraint"],
                        self.params["agent"]["Lc"],
                    )
                else:
                    out = optimal_graph.graph.copy()
                self.optimal_graphs_eps.append(out)
        Rbar_eps_nodes_union = set.union(
            *[
                set(self.optimal_graphs_eps[i].nodes)
                for i in range(self.params["env"]["n_players"])
            ]
        )
        self.normalization_Rbar_eps = torch.sum(
            self.true_density[list(Rbar_eps_nodes_union)]
        )

    def compute_normalization_factor(self):
        print("Calculating normalization factor")
        optimal_graph = CentralGraph(env_params=self.params["env"])
        for i in range(self.params["env"]["n_players"]):
            # Safe.append(SafeSet(self.init_safe[i], self.init_safe[i], V))
            if self.params["agent"]["use_goose"]:
                out = expansion_operator(
                    optimal_graph.graph,
                    self.true_constraint_function,
                    self.init_safe["idx"][i],
                    self.params["common"]["constraint"],
                    self.params["agent"]["Lc"],
                )
            else:
                out = optimal_graph.graph.copy()
            self.optimal_graphs0.append(out)

        Rbar0_nodes_union = set.union(
            *[
                set(self.optimal_graphs0[i].nodes)
                for i in range(self.params["env"]["n_players"])
            ]
        )
        self.normalization_Rbar0 = torch.sum(self.true_density[list(Rbar0_nodes_union)])

    def __compute_true_asso_dict(self):
        # 1) Computation of optimal solution
        skip_list = []
        start = time.time()
        for lead_idx, lead_graph in enumerate(self.optimal_graphs_eps):
            if lead_idx not in skip_list:
                skip_list.append(lead_idx)
                # if not (set_num in true_associate_dict):
                self.true_associate_dict[lead_idx] = []
                self.true_associate_dict[lead_idx].append(lead_idx)
                print(self.true_associate_dict)
                for agent_idx in range(lead_idx + 1, len(self.optimal_graphs_eps)):
                    if agent_idx not in skip_list:
                        G = nx.intersection(
                            lead_graph, self.optimal_graphs_eps[agent_idx]
                        )
                        if len(G.nodes) > 0:
                            skip_list.append(agent_idx)
                            self.true_associate_dict[lead_idx].append(agent_idx)
                    # if nx.is_isomorphic(lead_graph, self.optimal_graphs_eps[agent_idx]):
        end = time.time()
        print("Associate time:", end - start)

    def __compute_opt_soln(self):
        self.opt_goal["Fx_X"] = torch.empty(0)
        self.opt_goal["Fx_Y"] = torch.empty(0)

        for key in self.true_associate_dict:
            k = len(self.true_associate_dict[key])
            start = time.time()
            idx_x_curr, dist_gain, opt_Fx_obj = greedy_algorithm_opti(
                self.true_density.clone(),
                self.optimal_graphs_eps[key],
                k,
                self.params["common"]["disk_size"],
            )
            end = time.time()
            print("time:", end - start)
            opt_goal_x, opt_goal_y = (
                self.grid_V[idx_x_curr],
                self.true_density[idx_x_curr],
            )
            self.opt_goal["Fx_X"] = torch.cat([self.opt_goal["Fx_X"], opt_goal_x])
            self.opt_goal["Fx_Y"] = torch.cat([self.opt_goal["Fx_Y"], opt_goal_y])
            self.opt_val += opt_Fx_obj

    def __compute_opt_feas_boundary(self):
        for key in self.true_associate_dict:
            G = self.optimal_graphs_eps[key]
            self.optimal_feasible_boundary[key] = [
                x for x in G.nodes() if (G.out_degree(x) == 3 or G.out_degree(x) == 2)
            ]

    def compute_optimal_location_by_expansion(self):
        self.__load_opt_graphs()
        self.__compute_true_asso_dict()
        self.__compute_opt_soln()
        self.__compute_opt_feas_boundary()

    def multi_coverage_oracle(self, idx_x_list, density, graph, disk_size, key):
        total_Fs = 0
        non_covered_density = density.clone()
        for idx_x in idx_x_list:
            if (
                idx_x not in graph.nodes
            ):  # The logic is not valid for 2d, ideally give large penalty of being unsafe if this comes up
                if idx_x in self.optimal_graphs0[key].nodes:
                    coverage_area_idx = set(
                        nx.single_source_shortest_path_length(
                            self.optimal_graphs0[key], idx_x, cutoff=disk_size
                        )
                    )
                    diff_nodes = set(self.optimal_graphs0[key].nodes) - set(graph.nodes)
                    common_nodes = coverage_area_idx - diff_nodes
                    Fs = torch.sum(density[list(common_nodes)])
                else:
                    Fs = 0
                # # Happens only with goose
                # nodes = list(graph.nodes)
                # closest_idx = np.abs(np.stack(nodes) - idx_x).argmin()
                # Fs = coverage_oracle(
                #     nodes[closest_idx], non_covered_density, graph, disk_size - (np.abs(nodes[closest_idx] - idx_x)))
                # non_covered_density[list(nx.single_source_shortest_path_length(
                #     graph, nodes[closest_idx], cutoff=disk_size - (np.abs(nodes[closest_idx] - idx_x))))] = 0.0
            else:
                Fs = coverage_oracle(idx_x, non_covered_density, graph, disk_size)
                non_covered_density[
                    list(
                        nx.single_source_shortest_path_length(
                            graph, idx_x, cutoff=disk_size
                        )
                    )
                ] = 0.0
            total_Fs += Fs
        return total_Fs

    def compute_cover_xIX_rho_Rbar(self, players, graph_type):
        sum_Fx_obj = torch.zeros(1)
        for key in self.true_associate_dict:
            # xn_buddies are the agents which are in same constrain set
            xn_buddies = [
                players[agent_idx].planned_disk_center
                for agent_idx in self.true_associate_dict[key]
            ]  # regret of measure pt and regret of planned pt. Only problem is that planned pt can also be outside of safe opt and then calculation can be an issue
            # idx_star = [
            #     players[self.true_associate_dict[key][0]].get_idx(xn_buddies)]
            # Fx_obj_at_diff_density = objective_oracle_given(
            #     idx_star, self.true_density.view(-1, 1)[final_feasible_Set[self.true_associate_dict[key][0]]])
            idx_xn_buddies = []
            for xn in xn_buddies:
                idx_xn_buddies.append(idxfromloc(self.grid_V, xn))
            if graph_type == "eps":
                Fx_obj_at_diff_density = self.multi_coverage_oracle(
                    idx_xn_buddies,
                    self.true_density,
                    self.optimal_graphs_eps[key],
                    self.params["common"]["disk_size"],
                    key,
                )
            else:
                Fx_obj_at_diff_density = self.multi_coverage_oracle(
                    idx_xn_buddies,
                    self.true_density,
                    self.optimal_graphs0[key],
                    self.params["common"]["disk_size"],
                    key,
                )
            sum_Fx_obj += Fx_obj_at_diff_density
        return sum_Fx_obj

    def compute_cover_xIX_lcb_pessi(self, players):
        sum_Fx_obj = torch.zeros(1)
        pessimistic_graphs = []
        for player in players:
            pessimistic_graphs.append(player.pessimistic_graph)
        for key in self.true_associate_dict:
            # xn_buddies are the agents which are in same constrain set
            xn_buddies = [
                players[agent_idx].planned_disk_center_at_last_meas
                for agent_idx in self.true_associate_dict[key]
            ]  # regret of measure pt and regret of planned pt. Only problem is that planned pt can also be outside of safe opt and then calculation can be an issue
            # idx_star = [
            #     players[self.true_associate_dict[key][0]].get_idx(xn_buddies)]
            # Fx_obj_at_diff_density = objective_oracle_given(
            #     idx_star, self.true_density.view(-1, 1)[final_feasible_Set[self.true_associate_dict[key][0]]])
            idx_xn_buddies = []
            for xn in xn_buddies:
                idx_xn_buddies.append(idxfromloc(self.grid_V, xn))
            Fx_obj_at_diff_density = self.multi_coverage_oracle(
                idx_xn_buddies,
                players[key].get_lcb_density(),
                pessimistic_graphs[key],
                self.params["common"]["disk_size"],
                key,
            )
            sum_Fx_obj += Fx_obj_at_diff_density
        return sum_Fx_obj.detach()

    def compute_cover_xlIX_rho_pessi(self, players, pessi_associate_dict):
        """This method is for the purpose of recommendation

        Args:
            players (_type_): _description_
            associate_dict (_type_): _description_

        Returns:
            _type_: _description_
        """

        sum_Fx_estimate_lcb = torch.zeros(1)
        sum_Fx_obj = torch.zeros(1)
        pessimistic_graphs = []
        for player in players:
            pessimistic_graphs.append(player.pessimistic_graph)

        for key in pessi_associate_dict:
            k = len(pessi_associate_dict[key])
            idx_xn_star, acq_density, M_dist, Fx_obj = players[
                key
            ].get_lcb_pessi_coverage_point(k)
            sum_Fx_estimate_lcb += Fx_obj
            for agent, idx_goal in zip(pessi_associate_dict[key], idx_xn_star):
                players[agent].set_greedy_lcb_pessi_goal = idx_goal

        for key in self.true_associate_dict:
            # xn_buddies are the agents which are in same constrain set
            idx_xn_buddies = [
                players[agent_idx].set_greedy_lcb_pessi_goal
                for agent_idx in self.true_associate_dict[key]
            ]
            Fx_obj_at_diff_density = self.multi_coverage_oracle(
                idx_xn_buddies,
                self.true_density,
                self.optimal_graphs0[key],
                self.params["common"]["disk_size"],
                key,
            )
            sum_Fx_obj += Fx_obj_at_diff_density
        return sum_Fx_obj, sum_Fx_estimate_lcb

    def compute_cover_xIX_rho_opti(self, players, associate_dict):
        sum_Fx_obj = torch.zeros(1)
        optimistic_graphs = []
        for player in players:
            optimistic_graphs.append(player.union_graph)
        for key in associate_dict:
            # xn_buddies are the agents which are in same constrain set
            xn_buddies = [
                players[agent_idx].get_recommendation_pt()
                for agent_idx in associate_dict[key]
            ]  # regret of measure pt and regret of planned pt. Only problem is that planned pt can also be outside of safe opt and then calculation can be an issue
            # idx_star = [
            #     players[associate_dict[key][0]].get_idx(xn_buddies)]
            # Fx_obj_at_diff_density = objective_oracle_given(
            #     idx_star, self.true_density.view(-1, 1)[final_feasible_Set[associate_dict[key][0]]])
            idx_xn_buddies = []
            for xn in xn_buddies:
                idx_xn_buddies.append(idxfromloc(self.grid_V, xn))

            Fx_obj_at_diff_density = self.multi_coverage_oracle(
                idx_xn_buddies,
                self.true_density,
                optimistic_graphs[key],
                self.params["common"]["disk_size"],
                key,
            )
            sum_Fx_obj += Fx_obj_at_diff_density
        return sum_Fx_obj

    def compute_cover_tildexIX_rho_opti(self, players, opti_associate_dict):
        sum_Fx_obj = torch.zeros(1)
        optimistic_graphs = []
        for player in players:
            optimistic_graphs.append(player.union_graph)
        # print('union length', len(player.union_graph))
        for key, player in enumerate(players):
            for asso_key in opti_associate_dict:
                if key in opti_associate_dict[asso_key]:
                    bool_condi_agents = (
                        torch.Tensor(opti_associate_dict[asso_key]) < key
                    )
                    condi_agents = torch.IntTensor(opti_associate_dict[asso_key])[
                        bool_condi_agents
                    ].tolist()
                    # xn_buddies are the agents which are in same constrain set
                    xn_buddies = [
                        players[agent_idx].get_recommendation_pt()
                        for agent_idx in condi_agents
                    ]  # regret of measure pt and regret of planned pt. Only problem is that planned pt can also be outside of safe opt and then calculation can be an issue
                    idx_xn_buddies = []
                    for xn in xn_buddies:
                        idx_xn_buddies.append(idxfromloc(self.grid_V, xn))
            idx_max_gain, max_gain = self.compute_opt_greedy_given_others(
                key, idx_xn_buddies, optimistic_graphs
            )
            sum_Fx_obj += max_gain
        return sum_Fx_obj

    def compute_opt_Fx_at_t(self, players):
        sum_Fx_obj = torch.zeros(1)
        for key, player in enumerate(players):
            for asso_key in self.true_associate_dict:
                if key in self.true_associate_dict[asso_key]:
                    bool_condi_agents = (
                        torch.Tensor(self.true_associate_dict[asso_key]) < key
                    )
                    condi_agents = torch.IntTensor(self.true_associate_dict[asso_key])[
                        bool_condi_agents
                    ].tolist()
                    # xn_buddies are the agents which are in same constrain set
                    xn_buddies = [
                        players[agent_idx].get_recommendation_pt()
                        for agent_idx in condi_agents
                    ]  # regret of measure pt and regret of planned pt. Only problem is that planned pt can also be outside of safe opt and then calculation can be an issue
                    idx_xn_buddies = []
                    for xn in xn_buddies:
                        idx_xn_buddies.append(idxfromloc(self.grid_V, xn))
            idx_max_gain, max_gain = self.compute_opt_greedy_given_others(
                key, idx_xn_buddies, self.optimal_graphs_eps
            )
            sum_Fx_obj += max_gain
        return sum_Fx_obj

    def compute_opt_greedy_given_others(self, key, others, optimistic_graphs):
        non_covered_density = torch.empty_like(self.true_density).copy_(
            self.true_density
        )

        for idx_x_curr in others:
            # The logic is not valid for 2d, ideally give large penalty of being unsafe if this comes up
            if idx_x_curr in optimistic_graphs[key].nodes:
                non_covered_density[
                    list(
                        nx.single_source_shortest_path_length(
                            optimistic_graphs[key],
                            idx_x_curr,
                            cutoff=self.params["common"]["disk_size"],
                        )
                    )
                ] = 0.0

        idx_max_gain, M_dist, max_gain = greedy_algorithm(
            non_covered_density,
            optimistic_graphs[key],
            1,
            self.params["common"]["disk_size"],
        )

        return idx_max_gain, max_gain

    # def compute_optimal_location(self.true_density, self.true_constraint_function, self.init_safe, params, V):
    #     # 1) Computation of optimal solution
    #     opt_goal = {}
    #     # TODO: rename opt_set to feasible set
    #     opt_set = self.true_constraint_function - \
    #         0.1 > params["common"]["constraint"]

    #     # 1.1) Compute the final disjoint set
    #     disjoint_sets = []
    #     j = -1
    #     i = 0
    #     while i < opt_set.size()[0]:
    #         if opt_set[i] == True:
    #             disjoint_sets.append(torch.ones(opt_set.size()[0]) > 2)
    #             j += 1
    #             while i < opt_set.size()[0] and (opt_set[i] == True):
    #                 disjoint_sets[j][i] = True
    #                 i = i+1
    #         i += 1

    #     opt_goal["Fx_X"] = torch.empty(0)
    #     opt_goal["Fx_Y"] = torch.empty(0)

    #     # TODO: Take greedy algorithm outside of player class
    #     true_associate_dict = {}
    #     final_feasible_Set = []
    #     opt_val = torch.zeros(1)
    #     for set_num, single_opt_set in enumerate(disjoint_sets):
    #         k = 0  # number of agent in a single disjoint optimal set
    #         for agent_num, loc in enumerate(self.init_safe):
    #             # ###### NOT TESTED changed here < to <= > to >=
    #             if (V[single_opt_set][0] <= loc[0] and loc[-1] <= V[single_opt_set][-1]):
    #                 k += 1
    #                 final_feasible_Set.append(single_opt_set)
    #                 # associate set_num1: agent1, agent2  .... set_num2: agent3...
    #                 if not (set_num in true_associate_dict):
    #                     true_associate_dict[set_num] = []
    #                 true_associate_dict[set_num].append(agent_num)
    #         if k > 0:  # if there is no agent initialized in the safe set then no need to compute greedy for it
    #             true_objective = self.true_density[single_opt_set]
    #             opt_goal_x, opt_goal_y, _, opt_Fx_obj = greedy_algorithm(
    #                 true_objective, V[single_opt_set], k, params["common"]["disk_size"])
    #             opt_goal["Fx_X"] = torch.cat([opt_goal["Fx_X"], opt_goal_x])
    #             opt_goal["Fx_Y"] = torch.cat([opt_goal["Fx_Y"], opt_goal_y])
    #             opt_val += opt_Fx_obj

    #     print(opt_goal["Fx_X"])
    #     feasible_set_type = []
    #     for feasible_set in final_feasible_Set:
    #         feasible_set_type.append(
    #             SafeSet(V[feasible_set].min(), V[feasible_set].max(), V))
    #     return opt_goal, opt_val, final_feasible_Set, true_associate_dict, feasible_set_type

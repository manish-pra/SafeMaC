# import plotly
# import warnings
# warnings.filterwarnings('ignore')
# plt.rcParams['figure.figsize'] = [12, 6]
from copy import copy
from dataclasses import dataclass

import networkx as nx
import numpy as np
# import matplotlib.pyplot as plt
# import os
import torch
from botorch import fit_gpytorch_model
from botorch.models import SingleTaskGP
from gpytorch.kernels import (LinearKernel, MaternKernel,
                              PiecewisePolynomialKernel, PolynomialKernel,
                              RBFKernel, ScaleKernel)
from gpytorch.mlls.exact_marginal_log_likelihood import \
    ExactMarginalLogLikelihood

from utils.agent_helper import (apply_goose, coverage_oracle, greedy_algorithm,
                                greedy_algorithm_opti,
                                greedy_algorithm_opti_cov)
from utils.central_graph import (CentralGraph, diag_grid_world_graph,
                                 expansion_operator, grid_world_graph)
from utils.datatypes import SafeSet
from utils.helper import idxfromloc

# @dataclass
# class Set:
#     left: float
#     right: float


class Agent(object):
    def __init__(
        self,
        my_key,
        X_train,
        Cx_Y_train,
        Fx_Y_train,
        agent_param,
        common_param,
        grid_V,
        env_params,
    ) -> None:
        self.my_key = my_key
        self.env_dim = common_param["dim"]
        self.Fx_X_train = X_train.reshape(-1, self.env_dim)
        self.Cx_X_train = X_train.reshape(-1, self.env_dim)
        self.Fx_Y_train = Fx_Y_train.reshape(-1, 1)
        self.Cx_Y_train = Cx_Y_train.reshape(-1, 1)
        self.disk_size = common_param["disk_size"]
        self.obs_model = agent_param["obs_model"]
        self.mean_shift_val = agent_param["mean_shift_val"]
        self.explore_exploit_strategy = agent_param["explore_exploit_strategy"]
        self.converged = False
        self.opti = grid_V
        self.grid_V = grid_V
        self.grid_V_prev = grid_V
        self.pessi = X_train
        self.agent_param = agent_param
        self.common_param = common_param
        self.Cx_beta = agent_param["Cx_beta"]
        self.Fx_beta = agent_param["Fx_beta"]
        self.Fx_lengthscale = agent_param["Fx_lengthscale"]
        self.Fx_noise = agent_param["Fx_noise"]
        self.Cx_lengthscale = agent_param["Cx_lengthscale"]
        self.Cx_noise = agent_param["Cx_noise"]
        self.constraint = common_param["constraint"]
        self.Lc = agent_param["Lc"]
        self.epsilon = common_param["epsilon"]
        self.step_size = env_params["step_size"]
        self.Nx = env_params["shape"]["x"]
        self.Ny = env_params["shape"]["y"]
        self.use_goose = agent_param["use_goose"]
        if env_params["cov_module"] == "Sq_exp":
            self.Cx_covar_module = ScaleKernel(
                base_kernel=RBFKernel(),
            )  # ard_num_dims=self.env_dim
            self.Fx_covar_module = ScaleKernel(
                base_kernel=RBFKernel(),
            )  # ard_num_dims=self.env_dim
        elif env_params["cov_module"] == "Matern":
            self.Cx_covar_module = ScaleKernel(
                base_kernel=MaternKernel(nu=2.5),
            )  # ard_num_dims=self.env_dim
            self.Fx_covar_module = ScaleKernel(
                base_kernel=MaternKernel(nu=2.5),
            )  # ard_num_dims=self.env_dim
        else:
            self.Cx_covar_module = ScaleKernel(
                base_kernel=PiecewisePolynomialKernel()
            )  # ard_num_dims=self.env_dim
            self.Fx_covar_module = ScaleKernel(
                base_kernel=PiecewisePolynomialKernel()
            )  # ard_num_dims=self.env_dim

        self.base_graph = grid_world_graph((self.Nx, self.Ny))
        self.diag_graph = diag_grid_world_graph((self.Nx, self.Ny))
        self.optimistic_graph = grid_world_graph((self.Nx, self.Ny))
        self.pessimistic_graph = nx.empty_graph(n=0, create_using=nx.DiGraph())
        self.union_graph = grid_world_graph((self.Nx, self.Ny))
        self.centralized_safe_graph = grid_world_graph((self.Nx, self.Ny))

        self.Fx_model = self.__update_Fx()
        self.Cx_model = self.__update_Cx()
        self.planned_disk_center = self.Fx_X_train
        self.all_safe_nodes = self.base_graph.nodes
        self.all_unsafe_nodes = []
        self.max_constraint_sigma_goal = None
        self.set_greedy_lcb_pessi_goal = None
        self.planned_disk_center_at_last_meas = X_train.reshape(
            -1, self.env_dim)
        self.record = {}
        self.record["lower_Fx"] = []
        self.record["upper_Fx"] = []

    def update_current_location(self, loc):
        """_summary_ Record current location of the agent

        Args:
            loc (torch.Tensor 1x1): Location of the agent
        """
        self.current_location = loc

    def get_recommendation_pt(self):
        if not self.agent_param["Two_stage"]:
            return self.planned_disk_center
        else:
            return self.planned_disk_center
            # PtsToexp = list(set(self.optimistic_graph.nodes) -
            #                 set(self.pessimistic_graph.nodes))
            # if len(PtsToexp) == 0:
            #     return self.planned_disk_center
            # else:
            #     self.set_goal_max_constraint_sigma_under_disc(PtsToexp)
            # return self.max_constraint_sigma_goal

    def update_disc_boundary(self, loc):
        # disc_nodes = self.get_expected_disc(idxfromloc(self.grid_V, loc))
        G = self.base_graph.subgraph(self.full_disc_nodes).copy()
        disc_bound_nodes = [x for x in G.nodes() if (G.out_degree(x) <= 3)]
        G1 = self.diag_graph.subgraph(disc_bound_nodes).copy()
        self.disc_boundary = list(nx.simple_cycles(G1))
        if len(G1.nodes) == 1:
            self.disc_boundary = list(G1.nodes())

    def communicate_constraint(self, X_set, Cx_set):
        for newX, newY in zip(X_set, Cx_set):
            self.__update_Cx_set(newX, newY)

    def communicate_density(self, X_set, Fx_set):
        for newX, newY in zip(X_set, Fx_set):
            self.__update_Fx_set(newX, newY)

    def update_Cx_gp(self, newX, newY):
        self.__update_Cx_set(newX, newY)
        self.__update_Cx()
        return self.Cx_model

    def update_Cx_gp_with_current_data(self):
        self.__update_Cx()
        return self.Cx_model

    def update_Fx_gp(self, newX, newY):
        self.__update_Fx_set(newX, newY)
        self.__update_Fx()
        return self.Fx_model

    def update_Fx_gp_with_current_data(self):
        self.__update_Fx()
        return self.Fx_model

    def __update_Cx_set(self, newX, newY):
        newX = newX.reshape(-1, self.env_dim)
        newY = newY.reshape(-1, 1)
        self.Cx_X_train = torch.cat(
            [self.Cx_X_train, newX]).reshape(-1, self.env_dim)
        self.Cx_Y_train = torch.cat([self.Cx_Y_train, newY]).reshape(-1, 1)

    def __update_Cx(self):
        self.Cx_model = SingleTaskGP(self.Cx_X_train, self.Cx_Y_train)
        # 1.2482120543718338
        self.Cx_model.covar_module.base_kernel.lengthscale = self.Cx_lengthscale
        self.Cx_model.likelihood.noise = self.Cx_noise
        # mll = ExactMarginalLogLikelihood(
        #     self.Cx_model.likelihood, self.Cx_model)
        # fit_gpytorch_model(mll)
        return self.Cx_model

    def __update_Fx_set(self, newX, newY):
        newX = newX.reshape(-1, self.env_dim)
        newY = newY.reshape(-1, 1)
        self.Fx_X_train = torch.cat(
            [self.Fx_X_train, newX]).reshape(-1, self.env_dim)
        self.Fx_Y_train = torch.cat([self.Fx_Y_train, newY]).reshape(-1, 1)

    def __update_Fx(self):
        Fx_Y_train = self.__mean_corrected(self.Fx_Y_train)
        self.Fx_model = SingleTaskGP(self.Fx_X_train, Fx_Y_train)
        self.Fx_model.covar_module.base_kernel.lengthscale = self.Fx_lengthscale
        self.Fx_model.likelihood.noise = self.Fx_noise
        # mll = ExactMarginalLogLikelihood(model.likelihood, model)
        # fit_gpytorch_model(mll)
        return self.Fx_model

    def __predict_Fx(self, newX):
        newX = newX.reshape(-1, self.env_dim)
        newY = self.Fx_model.posterior(newX).mean
        Fx_Y_train = self.__mean_corrected(self.Fx_Y_train)
        Fx_X_train = torch.cat([self.Fx_X_train, newX]
                               ).reshape(-1, self.env_dim)
        Fx_Y_train = torch.cat([Fx_Y_train, newY]).reshape(-1, 1)
        Fx_model = SingleTaskGP(Fx_X_train, Fx_Y_train)
        Fx_model.covar_module.base_kernel.lengthscale = self.Fx_lengthscale
        Fx_model.likelihood.noise = self.Fx_noise
        return Fx_model

    def __mean_corrected(self, variable):
        return variable - self.mean_shift_val

    def get_Fx_bounds(self, V):
        lower_Fx, upper_Fx = self.Fx_model.posterior(V).mvn.confidence_region()
        lower_Fx, upper_Fx = scale_with_beta(lower_Fx, upper_Fx, self.Fx_beta)
        self.lower_Fx = torch.max(self.lower_Fx, lower_Fx)
        self.upper_Fx = torch.min(self.upper_Fx, upper_Fx)
        return self.lower_Fx, self.upper_Fx

    def save_posterior_normalization_const(
        self,
    ):
        lower_Fx, upper_Fx = self.Fx_model.posterior(
            self.grid_V
        ).mvn.confidence_region()
        lower_Fx, upper_Fx = scale_with_beta(lower_Fx, upper_Fx, self.Fx_beta)
        # acq_density = self.Fx_beta*upper_Fx  # acq without mean shift
        diff = upper_Fx - lower_Fx
        self.posterior_normalization_const = diff.max().detach()

    def UpdateConvergence(self, converged):
        self.converged = converged

    def update_union_graph(self, union_graph):
        self.union_graph = union_graph

    def update_optimistic_graph(self, upper_bound, init_node, thresh, Lc):
        # remove Lc*0.12 from all the adjacent cells
        # outer_nodes = list(set(self.optimistic_graph.nodes))  # influence shrinkage
        outer_nodes = list(
            set(self.optimistic_graph.nodes) -
            set(self.pessimistic_graph.nodes)
        )
        upper_bound[outer_nodes] = upper_bound[outer_nodes]
        # self.Lc*self.step_size
        self.optimistic_graph = expansion_operator(
            self.optimistic_graph, upper_bound, init_node, thresh, Lc
        )
        print("Nodes in optimistic graph:", len(self.optimistic_graph.nodes))
        # Lc*step_size is imp since this is the best we can do to create expander set. If this is not satisfied we may not be able to expand
        return True

    def update_pessimistic_graph(self, lower_bound, init_node, thresh, Lc):
        total_safe_nodes = torch.arange(0, lower_bound.shape[0])[
            lower_bound > thresh]
        total_safe_nodes = torch.unique(
            torch.cat([total_safe_nodes, init_node.reshape(-1)])
        )
        total_safe_graph = self.base_graph.subgraph(total_safe_nodes.numpy())
        edges = nx.algorithms.traversal.breadth_first_search.bfs_edges(
            total_safe_graph, init_node.item()
        )  # to remove non connected areas
        connected_nodes = [init_node.item()] + [v for u, v in edges]
        self.pessimistic_graph = update_graph(
            self.pessimistic_graph, self.base_graph, nodes_to_add=connected_nodes
        )
        print("Nodes in pesimistic graph:", len(self.pessimistic_graph.nodes))
        return True

    def update_centralized_unit(
        self,
        all_safe_nodes,
        all_unsafe_nodes,
        centralized_safe_graph,
        unsafe_edges_set,
        unreachable_nodes,
    ):
        self.all_safe_nodes = all_safe_nodes
        self.all_unsafe_nodes = all_unsafe_nodes
        self.centralized_safe_graph = centralized_safe_graph
        self.unsafe_edges_set = unsafe_edges_set
        self.unreachable_nodes = unreachable_nodes

    def get_meas_loc(self, planned_disk_center):
        if self.agent_param["recommend"] == "UCB":
            condi_meas_disc = self.condi_disc_nodes
        elif self.agent_param["recommend"] == "Fcov_UCB":
            condi_meas_disc = self.condi_disc_nodes
        elif self.agent_param["recommend"] == "Hallucinate":
            condi_meas_disc = self.condi_disc_nodes
        elif self.agent_param["recommend"] == "LCB":
            # recomendation is based on LCB, but identify the point that maximizes UCB
            observed_posterior = self.Fx_model.posterior(self.grid_V)
            lower, upper = observed_posterior.mvn.confidence_region()
            non_covered_density = torch.empty_like(upper + self.mean_shift_val).copy_(
                upper + self.mean_shift_val
            )
            non_covered_density[self.prior_covered_nodes] = 0.0
            # non_covered_density[list(nx.single_source_shortest_path_length(
            #     self.optimistic_graph, planned_disk_center, cutoff=self.common_param["disk_size"]))] = 0.0

            idx_max_gain, M_dist, max_gain = greedy_algorithm(
                non_covered_density,
                self.optimistic_graph,
                1,
                self.common_param["disk_size"],
            )
            condi_meas_disc = list(
                set(self.get_expected_disc(idx_max_gain[0]))
                - set(self.prior_covered_nodes)
            )
        if condi_meas_disc:
            if (
                len(self.list_others_meas_loc) == 0
                or self.agent_param["obs_model"] == "disc_max_pt"
            ):
                w, _ = self.get_max_sigma_under_disc(condi_meas_disc)
            else:  # disc_max_cond_pt
                w, _ = self.get_max_conditioned_sigma_under_disc(
                    condi_meas_disc)
            idx = w.argmax().item()
            self.max_density_sigma = w.max()
            return self.grid_V[condi_meas_disc][idx], self.max_density_sigma.detach()
        else:  # if the area is covered by another agent
            return planned_disk_center, torch.zeros(1)[0]

    def set_others_meas_loc(self, list_meas_loc):
        self.list_others_meas_loc = list_meas_loc.copy()

    def set_submodular_goal(self, xi_star):
        self.planned_disk_center = xi_star
        meas_loc, max_density_sigma = self.get_meas_loc(xi_star)
        self.planned_measure_loc = meas_loc
        if self.agent_param["obs_model"] == "disc_center":
            self.planned_measure_loc = self.planned_disk_center
        # self.planned_disc_boundary = xn_planned_dict["disc"]
        self.max_density_sigma = max_density_sigma

    def get_next_to_go_loc(self):
        if not self.agent_param["Two_stage"]:
            return self.planned_measure_loc
        else:
            if self.max_constraint_sigma_goal == None:
                PtsToexp = list(
                    set(self.optimistic_graph.nodes) -
                    set(self.pessimistic_graph.nodes)
                )
                self.max_constraint_goal_idx = (
                    self.set_goal_max_constraint_sigma_under_disc(PtsToexp)
                )
                return self.max_constraint_sigma_goal
            else:
                PtsToexp = list(
                    set(self.optimistic_graph.nodes) -
                    set(self.pessimistic_graph.nodes)
                )
                if self.max_constraint_goal_idx in PtsToexp:
                    return self.max_constraint_sigma_goal
                else:
                    PtsToexp = list(
                        set(self.optimistic_graph.nodes)
                        - set(self.pessimistic_graph.nodes)
                    )
                    if len(PtsToexp) == 0:  # fully explored
                        return self.planned_measure_loc
                    else:
                        self.max_constraint_goal_idx = (
                            self.set_goal_max_constraint_sigma_under_disc(
                                PtsToexp)
                        )
                        return self.max_constraint_sigma_goal

    def update_next_to_go_loc(self, loc):
        self.planned_measure_loc = loc

    def get_goose_goal(self, xi_star):
        # If not true then the agent will directly reach the goal point
        reached_pt = xi_star
        if self.use_goose:
            # keep on expanding untill the point xi_star is in pessimistic set or outside of the safe set

            # while count < 1 and (not target_in_pessi):
            # count += 1
            # print("count", count)
            target_in_pessi = (
                idxfromloc(
                    self.grid_V, xi_star) in self.pessimistic_graph.nodes
            )
            if not target_in_pessi:
                reached_pt, fully_explored = apply_goose(
                    self.pessimistic_graph,
                    self.optimistic_graph,
                    self.grid_V,
                    self.agent_param,
                    self.common_param,
                    self.Cx_model,
                    xi_star,
                )
                if fully_explored:
                    nodes_to_remove = list(
                        set(self.optimistic_graph) -
                        set(self.pessimistic_graph)
                    )
                    self.optimistic_graph = update_graph(
                        self.optimistic_graph,
                        self.base_graph,
                        nodes_to_remove=np.stack(nodes_to_remove),
                    )
                    # TODO: Do not collect more sample just pass since thresh is already over
                    # TODO: Shrink the optimistic and pessimistic graphs
                    reached_pt = self.current_location
        return reached_pt

    def get_expected_disc_loc(self, loc):
        disc_nodes = self.get_expected_disc(idxfromloc(self.grid_V, loc))
        return disc_nodes

    def get_expected_disc(self, loc_idx):
        if self.use_goose:  # TODO: think what else can be done here.
            disc_nodes = list(
                nx.single_source_shortest_path_length(
                    self.union_graph, loc_idx, cutoff=self.disk_size
                )
            )
        else:
            disc_nodes = list(
                nx.single_source_shortest_path_length(
                    self.base_graph, loc_idx, cutoff=self.disk_size
                )
            )
        self.full_disc_nodes = disc_nodes
        return disc_nodes

    def set_condi_disc_nodes(self, prior_covered_nodes):
        self.prior_covered_nodes = prior_covered_nodes
        self.condi_disc_nodes = list(
            set(self.full_disc_nodes) - set(prior_covered_nodes)
        )

    def get_max_sigma_under_disc(self, disc_nodes):
        observed_posterior = self.Fx_model.posterior(self.grid_V[disc_nodes])
        lower, upper = observed_posterior.mvn.confidence_region()
        lower, upper = scale_with_beta(lower, upper, self.Fx_beta)
        w = upper - lower
        return w, disc_nodes

    def get_max_sigma(self):
        observed_posterior = self.Fx_model.posterior(self.grid_V)
        lower, upper = observed_posterior.mvn.confidence_region()
        lower, upper = scale_with_beta(lower, upper, self.Fx_beta)
        w = upper - lower
        return w.max().detach()

    def get_max_conditioned_sigma_under_disc(self, disc_nodes):
        """_summary_ This operation is referred as hallucination for picking measurmeent location.
        Based on the locations of the other agents till i-1th idx, we evalue location for ith agent
        Args:
            disc_nodes (list): Nodes covered by the agents dics at current location

        Returns:
            w (torch.Tensor Nx1): N is number of nodes in the disc, w is width of the confidence interval
        """
        # 1) Predict a new model by hallucinating measurements of i-1 agents
        model_Fx = self.__predict_Fx(torch.vstack(self.list_others_meas_loc))
        observed_posterior = model_Fx.posterior(self.grid_V[disc_nodes])

        # 2) Compute new confidence bounds
        lower, upper = observed_posterior.mvn.confidence_region()
        lower, upper = scale_with_beta(lower, upper, self.Fx_beta)
        w = upper - lower
        return w, disc_nodes

    def hallucination(self, acq_density, n_soln):
        idx_x_curr = []
        dist_gain = []
        opt_Fx_obj = 0
        for i in range(n_soln):
            idx_x_curr_agent, dist_gain_agent, opt_Fx_obj_agent = greedy_algorithm_opti(
                acq_density.clone(), self.base_graph, 1, self.disk_size
            )
            idx_x_curr.append(idx_x_curr_agent[0])
            dist_gain.append(dist_gain_agent[0])
            opt_Fx_obj += opt_Fx_obj_agent
            if i < n_soln - 1:
                model_Fx = self.__predict_Fx(self.grid_V[idx_x_curr])
                lower_Fx, upper_Fx = model_Fx.posterior(
                    self.grid_V
                ).mvn.confidence_region()
                lower_Fx, upper_Fx = scale_with_beta(
                    lower_Fx, upper_Fx, self.Fx_beta)
                acq_density = upper_Fx + self.mean_shift_val
                for j in range(i + 1):
                    acq_density[
                        list(
                            nx.single_source_shortest_path_length(
                                self.base_graph,
                                idx_x_curr[-(j + 1)],
                                cutoff=self.disk_size,
                            )
                        )
                    ] = 0.0
        return idx_x_curr, dist_gain, opt_Fx_obj

    def set_goal_max_constraint_sigma_under_disc(self, disc_nodes):
        observed_posterior = self.Cx_model.posterior(self.grid_V[disc_nodes])
        lower, upper = observed_posterior.mvn.confidence_region()
        lower, upper = scale_with_beta(lower, upper, self.Cx_beta)
        w = upper - lower
        idx = w.argmax().item()
        self.max_constraint_sigma_goal = self.grid_V[disc_nodes][idx]
        return disc_nodes[idx]

    def get_max_uncertain_under_disc(self, loc_idx):
        disc_nodes = self.get_expected_disc(loc_idx)
        w, disc_nodes = self.get_max_sigma_under_disc(disc_nodes)
        idx = w.argmax().item()
        return self.grid_V[disc_nodes][idx], w.max().item(), disc_nodes

    def get_measurement_pt(self, loc_idx):
        """_summary_ NOT IN USE ANYMORE

        Args:
            loc_idx (int): index representing node number

        Returns:
            torch.Tensor 2x1: return co-ordinate of measurement location
        """
        if self.obs_model == "disc_max_pt":
            return self.get_max_uncertain_under_disc(loc_idx)[0]
        elif self.obs_model == "disc_center":
            return self.grid_V[loc_idx]
        elif self.obs_model == "Bernoulli":
            w, disc_nodes = self.get_max_sigma_under_disc(loc_idx)
            ratio = w / self.posterior_normalization_const
            p = torch.Tensor(
                [w.max().detach() / self.posterior_normalization_const])
            val = torch.bernoulli(p).item()
            if val == 1:
                return self.get_max_uncertain_under_disc(loc_idx)[0]
            else:
                return self.grid_V[loc_idx]

    def get_next_goal(self, n_soln):
        if self.explore_exploit_strategy == 0:  # bernaulli
            xn_star, acq_density, M_dist, Fx_obj, exploit = self.get_goal_bernoulli(
                self.grid_V, n_soln
            )
        elif self.explore_exploit_strategy == 1:  # pure exploitation
            xn_star, acq_density, M_dist, Fx_obj = self.get_coverage_point(
                n_soln)
            exploit = True
        elif self.explore_exploit_strategy == 2:  # 2 stage algorithm
            xn_star, acq_density, M_dist, Fx_obj = self.get_coverage_point(
                n_soln)
            xn_star, acq_density, M_dist = self.get_2maxCI_points(
                self.grid_V, n_soln)
            exploit = False
            if self.converged:
                xn_star, acq_density, M_dist, Fx_obj = self.get_coverage_point(
                    n_soln)
                exploit = True

        return xn_star, acq_density, M_dist, Fx_obj, exploit

    def get_goal_bernoulli(self, V, n_soln):
        lower_Fx, upper_Fx = self.Fx_model.posterior(V).mvn.confidence_region()
        # acq_density = self.Fx_beta*upper_Fx  # acq without mean shift
        lower_Fx, upper_Fx = scale_with_beta(lower_Fx, upper_Fx, self.Fx_beta)
        diff = upper_Fx - lower_Fx

        # Exploration phase
        p = torch.Tensor(
            [diff.max().detach() / self.posterior_normalization_const])
        val = torch.bernoulli(p).item()
        print("In bernoulli", val)
        exploit = True
        if val == 1:
            xn_star, acq_density, M_dist, Fx_obj = self.get_coverage_point(
                V, n_soln)
            xn_star, acq_density, M_dist = self.get_2maxCI_points(V, n_soln)
            exploit = False
            print("Exploration inside", p, val)

        else:
            xn_star, acq_density, M_dist, Fx_obj = self.get_coverage_point(
                V, n_soln)
            exploit = True
            print("Exploitation inside", p, val,
                  torch.bernoulli(p * torch.ones([4])))

        return xn_star, acq_density, M_dist, Fx_obj, exploit

    def get_maxCI_point(self, V):
        # 2.1) Get the density function \mu to optimize
        lower_Fx, upper_Fx = self.Fx_model.posterior(V).mvn.confidence_region()
        # acq_density = self.Fx_beta*upper_Fx  # acq without mean shift
        lower_Fx, upper_Fx = scale_with_beta(lower_Fx, upper_Fx, self.Fx_beta)
        diff = upper_Fx - lower_Fx
        xn_star = torch.Tensor(
            [V[diff.argmax()], V[diff.argmax()]]).reshape(-1)
        acq_density = diff
        return xn_star, acq_density, self.V

    def get_uncertain_points(self, V, model_Fx):
        # 2.1) Get the density function \mu to optimize
        lower_Fx, upper_Fx = model_Fx.posterior(V).mvn.confidence_region()
        # acq_density = self.Fx_beta*upper_Fx  # acq without mean shift
        lower_Fx, upper_Fx = scale_with_beta(lower_Fx, upper_Fx, self.Fx_beta)
        diff = upper_Fx - lower_Fx
        x1_star = V[diff.argmax()]
        return x1_star

    def get_2maxCI_points(self, V, n_soln):
        model_Fx = self.Fx_model
        xn_star = torch.empty(0)
        for _ in range(n_soln):
            x1_star = self.get_uncertain_points(V, model_Fx)
            xn_star = torch.cat([xn_star, x1_star.reshape(-1)])
            model_Fx = self.__predict_Fx(xn_star)

        lower_Fx, upper_Fx = self.Fx_model.posterior(V).mvn.confidence_region()
        # acq_density = self.Fx_beta*upper_Fx  # acq without mean shift
        lower_Fx, upper_Fx = scale_with_beta(lower_Fx, upper_Fx, self.Fx_beta)
        acq_density = (upper_Fx + lower_Fx) / 2 + self.mean_shift_val
        return xn_star, acq_density.detach(), self.V

        # # 2.1) Get the density function \mu to optimize
        # lower_Fx, upper_Fx = self.Fx_model.posterior(V).mvn.confidence_region()
        # # acq_density = self.Fx_beta*upper_Fx  # acq without mean shift
        # lower_Fx = lower_Fx*(1+self.Fx_beta)/2 + upper_Fx*(1-self.Fx_beta)/2
        # upper_Fx = upper_Fx*(1+self.Fx_beta)/2 + lower_Fx*(1-self.Fx_beta)/2
        # diff = upper_Fx - lower_Fx
        # x1_star = V[diff.argmax()]
        # Fx_model = self.__predict_Fx(x1_star)

        # lower_Fx, upper_Fx = Fx_model.posterior(V).mvn.confidence_region()
        # # acq_density = self.Fx_beta*upper_Fx  # acq without mean shift
        # lower_Fx = lower_Fx*(1+self.Fx_beta)/2 + upper_Fx*(1-self.Fx_beta)/2
        # upper_Fx = upper_Fx*(1+self.Fx_beta)/2 + lower_Fx*(1-self.Fx_beta)/2
        # diff = upper_Fx - lower_Fx
        # xn_star = torch.Tensor([x1_star, V[diff.argmax()]]).reshape(-1)
        # acq_density = diff
        # return xn_star, acq_density, self.V

    def get_lcb_density(self):
        # 2.1) Get the density function \mu to optimize
        lower_Fx, upper_Fx = self.Fx_model.posterior(
            self.grid_V
        ).mvn.confidence_region()
        # acq_density = self.Fx_beta*upper_Fx  # acq without mean shift
        lower_Fx, upper_Fx = scale_with_beta(lower_Fx, upper_Fx, self.Fx_beta)
        acq_density = lower_Fx + self.mean_shift_val
        return acq_density

    def get_lcb_pessi_coverage_point(self, n_soln):
        # 2.1) Get the density function \mu to optimize
        lower_Fx, upper_Fx = self.Fx_model.posterior(
            self.grid_V
        ).mvn.confidence_region()
        # acq_density = self.Fx_beta*upper_Fx  # acq without mean shift
        lower_Fx, upper_Fx = scale_with_beta(lower_Fx, upper_Fx, self.Fx_beta)
        acq_density = lower_Fx + self.mean_shift_val
        idx_x_curr, dist_gain, opt_Fx_obj = greedy_algorithm_opti(
            acq_density.clone(), self.pessimistic_graph, n_soln, self.disk_size
        )
        return idx_x_curr, acq_density, dist_gain, opt_Fx_obj.detach()

    def get_coverage_point(self, n_soln):
        # 2.1) Get the density function \mu to optimize
        # mvn.confidence_region() return \mu + 2* \sigma. Hence \beta=4. since \sqrt(\beta)=2.
        lower_Fx, upper_Fx = self.Fx_model.posterior(
            self.grid_V
        ).mvn.confidence_region()
        # acq_density = self.Fx_beta*upper_Fx  # acq without mean shift
        lower_Fx, upper_Fx = scale_with_beta(lower_Fx, upper_Fx, self.Fx_beta)
        self.record["upper_Fx"].append(upper_Fx)
        self.record["lower_Fx"].append(lower_Fx)
        if (
            self.agent_param["recommend"] == "UCB"
            or self.agent_param["recommend"] == "Hallucinate"
        ):
            acq_density = upper_Fx + self.mean_shift_val
        elif self.agent_param["recommend"] == "Fcov_UCB":
            # multiplying with 2 due to confidence_region. Then scaling with Fx_beta
            mat = self.Fx_model.posterior(
                self.grid_V).mvn.covariance_matrix * ((self.Fx_beta)**2)
            acq_density = (upper_Fx + lower_Fx) / 2 + self.mean_shift_val
        elif self.agent_param["recommend"] == "LCB":
            acq_density = lower_Fx + self.mean_shift_val

        # acq_density = (upper_Fx + lower_Fx)/2 + self.mean_shift_val
        # acq_density = upper_Fx + self.mean_shift_val
        # acq_density = self.Fx_model.posterior(
        #     self.grid_V).sample().reshape(-1) + self.mean_shift_val
        # 2.2) Use greedy algorithm to get new index to visit
        if self.use_goose:
            if self.agent_param["sol_domain"] == "pessi":
                idx_x_curr, dist_gain, opt_Fx_obj = greedy_algorithm_opti(
                    acq_density.clone(), self.pessimistic_graph, n_soln, self.disk_size
                )
            else:
                idx_x_curr, dist_gain, opt_Fx_obj = greedy_algorithm_opti(
                    acq_density.clone(), self.union_graph, n_soln, self.disk_size
                )
        else:
            if self.agent_param["recommend"] == "Fcov_UCB":  # acq is mean
                idx_x_curr, dist_gain, opt_Fx_obj = greedy_algorithm_opti_cov(
                    acq_density.clone(), self.base_graph, n_soln, self.disk_size, mat
                )
            elif self.agent_param["recommend"] == "Hallucinate":
                idx_x_curr, dist_gain, opt_Fx_obj = self.hallucination(
                    acq_density.clone(), n_soln
                )
            else:
                idx_x_curr, dist_gain, opt_Fx_obj = greedy_algorithm_opti(
                    acq_density.clone(), self.base_graph, n_soln, self.disk_size
                )

        return self.grid_V[idx_x_curr], acq_density, dist_gain, opt_Fx_obj.detach()

    def get_F_of_x_with_CI(self, V, n_soln, exploit, n_samples):
        ret_Fx_obj = []
        if exploit:
            for i in range(n_samples):
                acq_density = (
                    self.Fx_model.posterior(V).sample().reshape(-1)
                    + self.mean_shift_val
                )
                ret_Fx_obj.append(greedy_algorithm(acq_density, V, n_soln)[3])
        else:
            xn_star, acq_density, M_dist = self.get_2maxCI_points(V, n_soln)
            for i in range(n_samples):
                idx_star = [self.get_idx(xn_star)]
                ret_Fx_obj.append(
                    objective_oracle_given(idx_star, acq_density))

        return torch.stack(ret_Fx_obj)

    # def get_F_of_x_for_fix_pts(self, X_fix):
    def get_Cx_bounds(self, grid_V):
        V_lower_Cx, V_upper_Cx = self.Cx_model.posterior(
            grid_V).mvn.confidence_region()
        V_lower_Cx = V_lower_Cx.detach()
        V_upper_Cx = V_upper_Cx.detach()
        V_lower_Cx, V_upper_Cx = scale_with_beta(
            V_lower_Cx, V_upper_Cx, self.Cx_beta)
        # front_shift_idx = int((grid_V[0] - self.V_prev[0])/0.12 + 0.01)
        # rear_shift_idx = int((self.V_prev[-1]-grid_V[-1])/0.12 + 0.01)
        # n = self.V_lower_Cx.shape[0]
        # temp_lower_Cx = self.V_lower_Cx[front_shift_idx:
        #                                 n-1*rear_shift_idx]
        # temp_upper_Cx = self.V_upper_Cx[front_shift_idx:
        #                                 n-1*rear_shift_idx]
        # delta_w = (temp_upper_Cx - temp_lower_Cx) - (V_upper_Cx-V_lower_Cx)
        # # print(self.Cx_X_train.shape, self.Cx_X_train)
        # # print("W",  delta_w)
        # # self.V_lower_Cx = torch.max(
        # #     temp_lower_Cx, V_lower_Cx)  # element wise max
        # # self.V_upper_Cx = torch.min(
        # #     temp_upper_Cx, V_upper_Cx)  # element wise min
        # self.V_lower_Cx = V_lower_Cx
        # self.V_upper_Cx = V_upper_Cx
        # self.V_prev = grid_V
        return V_lower_Cx, V_upper_Cx

    def update_graph(self, Safe):
        V_lower_Cx, V_upper_Cx = self.get_Cx_bounds(self.grid_V)

        # Order matters here
        self.update_pessimistic_graph(
            V_lower_Cx, Safe, self.constraint, self.Lc)

        self.update_optimistic_graph(
            V_upper_Cx - self.epsilon, Safe, self.constraint, self.Lc
        )

        return True

    def get_idx(self, positions):
        idx = []
        for position in positions:
            idx.append(torch.abs(torch.Tensor(
                self.V) - position).argmin().item())
        return idx

    def expansion_operator(self, V, V_bound_Cx, init_set, pessi):
        S_po_mat = []
        S_po_prev = copy(init_set)
        S_po = copy(S_po_prev)
        S_po_mat.append(S_po_prev)
        termin_condn = False
        bound_left = V_bound_Cx[self.get_idx([S_po_prev.Xleft])[0]].detach()
        bound_right = V_bound_Cx[self.get_idx([S_po_prev.Xright])[0]].detach()
        set_bound = SafeSet(bound_left, bound_right, V, 0.12)
        while not termin_condn:
            bound_left = V_bound_Cx[self.get_idx(
                [S_po_prev.Xleft])[0]].detach()
            bound_right = V_bound_Cx[self.get_idx(
                [S_po_prev.Xright])[0]].detach()
            set_bound.Update(bound_left, bound_right)
            # print((set_bound.Xleft-self.constraint)/self.Lc, (set_bound.Xright-self.constraint)/self.Lc)

            S_po_left = S_po_prev.Xleft
            for steps in range(100):
                if set_bound.Xleft - self.Lc * (0.12) * steps < self.constraint:
                    if steps == 0:
                        break
                    S_po_left = max(S_po_prev.Xleft - (0.12)
                                    * (steps - 1), V.min())
                    break
            # if V[min(V.shape[0]-1, self.get_idx([S_po_prev.Xleft - (set_bound.Xleft-self.constraint)/self.Lc])[0])][0] < S_po_prev.Xleft:
            #     S_po_left = V[self.get_idx([
            #         S_po_prev.Xleft - (set_bound.Xleft-self.constraint)/self.Lc])[0]][0]
            # print("in while")
            S_po_right = S_po_prev.Xright
            for steps in range(100):
                if set_bound.Xright - self.Lc * (0.12) * steps < self.constraint:
                    if steps == 0:
                        break
                    S_po_right = min(S_po_prev.Xright + (0.12)
                                     * (steps - 1), V.max())
                    break
            # if V[self.get_idx([S_po_prev.Xright + (set_bound.Xright-self.constraint)/self.Lc])[0]][0] > S_po_prev.Xright:
            #     S_po_right = V[self.get_idx([
            #         S_po_prev.Xright + (set_bound.Xright-self.constraint)/self.Lc])[0]][0]
            S_po.Update(S_po_left, S_po_right)
            termin_condn = (S_po_prev.Xleft == S_po.Xleft) and (
                S_po_prev.Xright == S_po.Xright
            )
            # print((S_po_prev.Xleft == S_po.Xleft), (S_po_prev.Xright == S_po.Xright))
            # print(termin_condn, S_po, S_po_prev)
            S_po_prev = copy(S_po)
            S_po_mat.append(S_po_prev)
            lines = {}
            lines["left"] = self.extract_bound_lines(
                S_po_mat, V_bound_Cx, "left")
            lines["right"] = self.extract_bound_lines(
                S_po_mat, V_bound_Cx, "right")
        return S_po, lines

    def extract_bound_lines(self, S_po_mat, V_bound_Cx, side):
        points = {}
        if side == "left":
            points["X"] = [k.Xleft.detach().numpy()
                           for k in S_po_mat for _ in range(2)]
        else:
            points["X"] = [
                k.Xright.detach().numpy() for k in S_po_mat for _ in range(2)
            ]
        points["Y"] = []
        for n, k in enumerate(V_bound_Cx[self.get_idx(points["X"])].detach().numpy()):
            if n % 2 == 0:
                k = self.constraint
            points["Y"].append(k)
        return points

    def get_Cx_data(self):
        data = {}
        data["Cx_X"] = self.Cx_X_train
        data["Cx_Y"] = self.Cx_Y_train
        data["loc"] = self.current_location
        return data

    def get_Fx_data(self):
        data = {}
        data["Fx_X"] = self.Fx_X_train
        data["Fx_Y"] = self.Fx_Y_train
        data["loc"] = self.current_location
        return data


def scale_with_beta(lower, upper, beta):
    """_summary_ Scale confidence intervals as per beta

    Args:
        lower (torch.Tensor Nx1): Lower confidence bound, note that gpytorch .confidence region have a inbuilt factor of 2
        upper (torch.Tensor Nx1): Upper confidence bound
        beta (int): Scaling coefficient. Note it is \sqrt(\beta) as per original bounds

    Returns:
        lower (torch.Tensor Nx1): Scaled lower confidence bound
        upper (torch.Tensor Nx1): Scaled upper confidence bound

    """
    temp = lower * (1 + beta) / 2 + upper * (1 - beta) / 2
    upper = upper * (1 + beta) / 2 + lower * (1 - beta) / 2
    lower = temp
    return lower, upper


def update_graph(G, base_G, nodes_to_remove=None, nodes_to_add=None):
    """
    Updates nodes of a given graph using connectivity structure of base graph.

    Parameters
    ----------
    G: nx.Graph
        Graph to update
    base_G: nx.Graph
        Base graph that gives connectivity structure
    nodes_to_remove: ndarray
        array of nodes to remove from G
    nodes_to_add: ndarray
        array of nodes to add to G

    Returns
    -------
    G: nx.Graph
        Updated graph
    """
    if nodes_to_add is not None and len(nodes_to_add) > 0:
        nodes = np.unique(
            np.hstack((np.asarray(list(G.nodes)), np.asarray(nodes_to_add)))
        )
        nodes = nodes.astype(np.int64)
        G = base_G.subgraph(nodes).copy()

    if nodes_to_remove is not None and nodes_to_remove.size > 0:
        for n in nodes_to_remove:
            G.remove_node(n)
            G.remove_edges_from(base_G.edges(n))

    return G


if __name__ == "__main__":
    # Initialization:
    S0 = [70, 71, 72]
    X_train = torch.Tensor([i for i in S0]).reshape(-1, 1)
    Fx_Y_train = torch.Tensor([i for i in S0]).reshape(-1, 1)
    Cx_Y_train = torch.Tensor([i for i in S0]).reshape(-1, 1)
    p1 = Agent(
        X_train,
        Cx_Y_train,
        Fx_Y_train,
        beta=3,
        mean_shift_val=2,
        constraint=0.5,
        eps=1e-2,
        explore_exploit_strategy=1,
        init_safe=S0,
        V=S0,
        Lc=0.5,
    )

    print(p1.update_Cx_gp(X_train, Cx_Y_train))
    print(p1.update_Fx_gp(X_train, Fx_Y_train))

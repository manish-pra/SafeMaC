import numpy as np
import torch
import matplotlib
import matplotlib.pyplot as plt

matplotlib.use("Agg")  # for debug mode to plot and save fig


class Visu:
    def __init__(
        self,
        f_handle,
        constraint,
        grid_V,
        safe_boundary,
        true_constraint_function,
        true_objective_func,
        opt_goal,
        optimal_feasible_boundary,
        agent_param,
        env_params,
        common_params,
    ) -> None:
        self.constraint = constraint
        self.f_handle = f_handle
        self.grid_V = grid_V
        self.common_params = common_params
        self.use_goose = agent_param["use_goose"]
        self.Nx = env_params["shape"]["x"]
        self.Ny = env_params["shape"]["y"]
        self.step_size = env_params["step_size"]
        self.num_players = env_params["n_players"]
        self.safe_boundary = safe_boundary
        # safety_function(test_x.numpy().reshape(-1, 1))
        self.true_constraint_function = true_constraint_function
        self.true_objective_func = true_objective_func
        self.opt_goal = opt_goal
        self.Cx_beta = agent_param["Cx_beta"]
        self.Fx_beta = agent_param["Fx_beta"]
        self.mean_shift_val = agent_param["mean_shift_val"]
        self.agent_current_loc = {}
        self.agent_current_goal = {}  # Dict of all the agent current goal
        self.discs_nodes = {}
        self.optimal_feasible_boundary = optimal_feasible_boundary
        self.prev_w = torch.zeros([101])
        self.x = self.grid_V.transpose(0, 1).reshape(-1, self.Nx, self.Ny)[0]
        self.y = self.grid_V.transpose(0, 1).reshape(-1, self.Nx, self.Ny)[1]
        self.tr_constraint = self.true_constraint_function.reshape(self.Nx, self.Ny)
        self.tr_density = self.true_objective_func.reshape(self.Nx, self.Ny)
        if self.Ny != 1:
            self.plot_contour_env()
        self.plot_unsafe_debug = False

    def fmt(self, x):
        s = f"{x:.1f}"
        if s.endswith("0"):
            s = f"{x:.0f}"
        return rf"{s} \%" if plt.rcParams["text.usetex"] else f"{s} %"

    def plot_contour_env(self, f=None, rm=None):
        # if f is None:
        #     f, ax = plt.subplots()
        # else:
        #     ax = f.axes[0]
        #     rm = []
        ax = self.f_handle.axes[0]
        CS = ax.contour(
            self.x.numpy(),
            self.y.numpy(),
            self.tr_constraint.numpy(),
            np.array(
                [
                    self.common_params["constraint"],
                    self.common_params["constraint"] + 0.2,
                ]
            ),
        )
        # rm.append([CS])
        CS2 = ax.contourf(
            self.x.numpy(), self.y.numpy(), self.tr_density.numpy(), alpha=0.2
        )
        # rm.append([CS2])
        ax.clabel(CS, CS.levels, inline=True, fmt=self.fmt, fontsize=10)

        CS2.cmap.set_over("red")
        CS2.cmap.set_under("blue")
        self.f_handle.colorbar(CS2)
        # rm.append([self.f_handle.colorbar(CS2)])
        return self.f_handle, ax, rm

    def plot_optimal_point(self):
        f, ax = self.plot_contour_env()
        for key in self.optimal_feasible_boundary:
            loc = self.grid_V[self.optimal_feasible_boundary[key]]
            ax.plot(loc[:, 0], loc[:, 1], ".", color="mediumpurple", mew=2)

            # ax.plot(loc[0], loc[1], color="mediumpurple")
        for agent_idx, init_pt in enumerate(self.safe_boundary):
            ax.text(init_pt[0], init_pt[1], str(agent_idx), color="cyan", fontsize=12)
            ax.plot(init_pt[0], init_pt[1], "*", color="cyan", mew=2)

        # for loc in self.opt_goal:
        # for i in range(self.opt_goal["Fx_X"].shape[0]):
        #     loc = self.opt_goal["Fx_X"][i]
        #     ax.text(loc[0], loc[1], str(
        #         i), color="tab:green", fontsize=12)
        ax.plot(
            self.opt_goal["Fx_X"][:, 0],
            self.opt_goal["Fx_X"][:, 1],
            "*",
            color="tab:green",
            mew=3,
        )

        # ax.clabel(CS2, CS2.levels, inline=True, fmt=fmt, fontsize=10)
        ax.axis("equal")
        # plt.show()
        # plt.savefig("check.png")

    def CxVisuUpdate(
        self,
        Cx_model,
        current_loc,
        pessi_boundary,
        opti_boundary,
        Cx_data,
        agent_key,
        all_unsafe_nodes,
        unsafe_edges_set,
        unreachable_nodes,
    ):
        self.Cx_model = Cx_model
        self.pessi_boundary = pessi_boundary
        self.opti_boundary = opti_boundary
        self.Cx_data = Cx_data
        self.Cx_agent_key = agent_key
        if not (self.Cx_agent_key in self.agent_current_loc):
            self.agent_current_loc[self.Cx_agent_key] = []
        self.agent_current_loc[self.Cx_agent_key].append(current_loc)
        self.all_unsafe_nodes = all_unsafe_nodes
        self.unsafe_edges_set = unsafe_edges_set
        self.unreachable_nodes = unreachable_nodes

    def FxUpdate(
        self,
        Fx_model,
        current_goal,
        single_disc_nodes,
        acq_density,
        M_dist,
        Fx_data,
        agent_key,
    ):
        self.Fx_model = Fx_model
        self.Fx_agent_key = agent_key
        self.Fx_data = Fx_data
        if not (self.Fx_agent_key in self.agent_current_goal):
            self.agent_current_goal[self.Fx_agent_key] = []
        self.agent_current_goal[self.Fx_agent_key].append(current_goal)
        self.acq_density = acq_density
        self.M_dist = M_dist
        if not (self.Fx_agent_key in self.discs_nodes):
            self.discs_nodes[self.Fx_agent_key] = []
        self.discs_nodes[self.Fx_agent_key] = single_disc_nodes

    def plot_Fx(self, f_handle):
        ax = self.f_handle.axes[0]
        rm = []
        for agent_key in self.agent_current_goal:
            data = self.agent_current_goal[agent_key][-1]
            # print("Visu check", data[0])
            rm.append(
                [
                    ax.text(
                        data["Fx_X"][0],
                        data["Fx_X"][1],
                        str(agent_key),
                        color="gold",
                        fontsize=12,
                    )
                ]
            )
            # Plot the currently pursuing goal along with goal of 3 agent with text
        rm.append(
            ax.plot(
                self.agent_current_goal[self.Fx_agent_key][-1]["Fx_X"][0],
                self.agent_current_goal[self.Fx_agent_key][-1]["Fx_X"][1],
                "*",
                color="gold",
                mew=1.5,
            )
        )

        for agent_key in self.discs_nodes:
            single_disc_connections = self.discs_nodes[agent_key]
            for edges in single_disc_connections:
                loc = self.grid_V[edges].reshape(-1, 2)
                rm.append(ax.plot(loc[:, 0], loc[:, 1], color="tab:brown"))
                # for loc in self.opt_goal:
        # ax.savefig("check2.png")
        # ax.close()
        ax.axis("equal")
        if self.use_goose:
            ax.set_title(
                "Iteration "
                + str(self.n_iter)
                + " Goose Iter "
                + str(self.goose_step)
                + " Agent "
                + str(self.Fx_agent_key)
            )
        else:
            ax.set_title(
                "Iteration " + str(self.n_iter) + " Agent " + str(self.Fx_agent_key)
            )
        return rm

    def plot1Dobj_GP(self, f_handle):
        ax = f_handle.axes[0]
        x = self.grid_V
        # observed_pred = self.Cx_model.likelihood(self.Cx_model(test_x))
        # posterior is only avaialble with botorch and not in gpytorch
        observed_posterior = self.Fx_model.posterior(x)
        # ax.plot(observed_pred.mean.detach().numpy())
        lower, upper = observed_posterior.mvn.confidence_region()
        lower = lower + self.mean_shift_val
        upper = upper + self.mean_shift_val
        temp = lower * (1 + self.Fx_beta) / 2 + upper * (1 - self.Fx_beta) / 2
        upper = upper * (1 + self.Fx_beta) / 2 + lower * (1 - self.Fx_beta) / 2
        lower = temp
        rm = []
        rm.append(
            [
                ax.fill_between(
                    x[:, 0].numpy(),
                    lower.detach().numpy(),
                    upper.detach().numpy(),
                    alpha=0.5,
                    color="tab:purple",
                )
            ]
        )
        rm.append(
            ax.plot(
                x[:, 0].numpy(),
                observed_posterior.mean.detach().numpy() + self.mean_shift_val,
                color="tab:purple",
                label="Fx-mean",
            )
        )
        ax.plot(x[:, 0].numpy(), self.true_objective_func, color="tab:orange")
        # rm.append(ax.plot(x[:, 0].reshape(-1, 1)[self.S_opti[self.Fx_agent_key].StateInSet], self.acq_density.detach().reshape(-1, 1)[self.S_opti[self.Fx_agent_key].StateInSet],
        #                    color="tab:pink"))

        ax.plot(
            self.Fx_data["Fx_X"][:, 0].numpy(),
            self.Fx_data["Fx_Y"].numpy(),
            "*",
            color="red",
            mew=2,
        )
        ax.plot(
            self.opt_goal["Fx_X"][:, 0].numpy(),
            self.opt_goal["Fx_Y"].numpy(),
            "*",
            color="tab:green",
            mew=3,
        )

        y_loc = -0.5
        fact = 0.1
        for agent_key in self.discs_nodes:
            single_disc_nodes = self.discs_nodes[agent_key]
            left = self.grid_V[np.min(single_disc_nodes)][0]
            right = self.grid_V[np.max(single_disc_nodes)][0]
            rm.append(
                ax.plot(
                    [left, right],
                    [y_loc - fact * agent_key, y_loc - fact * agent_key],
                    color="tab:brown",
                    linewidth=6.0,
                )
            )

        for agent_key in self.agent_current_goal:
            data = self.agent_current_goal[agent_key][-1]  # last data point
            # print("Visu check", data[0])
            rm.append(
                [
                    ax.text(
                        data["Fx_X"][0],
                        data["Fx_Y"].view(-1),
                        str(agent_key),
                        color="gold",
                        fontsize=12,
                    )
                ]
            )
        # Plot the currently pursuing goal along with goal of 3 agent with text
        rm.append(
            ax.plot(
                self.agent_current_goal[self.Fx_agent_key][-1]["Fx_X"][0],
                self.agent_current_goal[self.Fx_agent_key][-1]["Fx_Y"].view(-1),
                "*",
                color="gold",
                mew=1.5,
            )
        )
        ax.legend(loc="upper left")
        # rm.append(ax.plot(x[:, 0].numpy(),
        #           self.M_dist[0]/1e6*2 + 4, color="brown"))
        # rm.append(ax.plot(x[:, 0].numpy(),
        #           self.M_dist[1]/1e6*2 + 1, color="brown"))
        if self.use_goose:
            ax.set_title(
                "Iteration "
                + str(self.n_iter)
                + " Goose Iter "
                + str(self.goose_step)
                + " Agent "
                + str(self.Fx_agent_key)
            )
        else:
            ax.set_title(
                "Iteration " + str(self.n_iter) + " Agent " + str(self.Fx_agent_key)
            )

        return rm

    def UpdateIter(self, iter, goose_step):
        self.n_iter = iter + 1
        self.goose_step = goose_step

    def plot_safe_GP(self, f_handle):
        rm = []
        # _, _, rm = self.plot_contour_env(f_handle, rm)
        ax = self.f_handle.axes[0]

        for key in self.optimal_feasible_boundary:
            loc = self.grid_V[self.optimal_feasible_boundary[key]]
            ax.plot(loc[:, 0], loc[:, 1], ".", color="mediumpurple", mew=2)

        for key in self.opti_boundary:
            loc = self.grid_V[self.opti_boundary[key]]
            rm.append(ax.plot(loc[:, 0], loc[:, 1], ".", color="gold", mew=2))

        for key in self.pessi_boundary:
            loc = self.grid_V[self.pessi_boundary[key]]
            rm.append(ax.plot(loc[:, 0], loc[:, 1], ".", color="black", mew=2))

        for init_pt in self.safe_boundary:
            ax.plot(init_pt[0], init_pt[1], "*", color="cyan", mew=2)

        # for loc in self.opt_goal:
        rm.append(
            ax.plot(
                self.opt_goal["Fx_X"][:, 0],
                self.opt_goal["Fx_X"][:, 1],
                "*",
                color="tab:green",
                mew=3,
            )
        )

        for key in self.agent_current_loc:
            data = self.agent_current_loc[key][-1]
            rm.append(
                [
                    ax.text(
                        data[0] - 2 * self.step_size,
                        data[1] - 2 * self.step_size,
                        str(key),
                        color="tab:brown",
                        fontsize=12,
                        weight="bold",
                    )
                ]
            )
            # rm.append(ax.plot(data[0], data[1],
            #                   ".", color="tab:brown",  mew=100))
            rm.append(ax.plot(data[0], data[1], ".", color="tab:brown", mew=10))
        if self.plot_unsafe_debug:
            all_unsafe_loc = self.grid_V[self.all_unsafe_nodes]
            ax.plot(all_unsafe_loc[:, 0], all_unsafe_loc[:, 1], "x", color="red", mew=1)
            unreachable_nodes = self.grid_V[self.unreachable_nodes]
            ax.plot(
                unreachable_nodes[:, 0],
                unreachable_nodes[:, 1],
                "x",
                color="black",
                mew=1,
            )

            for edge in list(self.unsafe_edges_set):
                st = self.grid_V[edge[0]]
                ed = self.grid_V[edge[1]]
                ax.plot([st[0], ed[0]], [st[1], ed[1]])
        # rm.append(ax.plot(self.Cx_data["Cx_X"][-self.num_players:, 0], self.Cx_data["Cx_X"]
        #                   [-self.num_players:, 1], ".", color="tab:brown",  mew=100))
        # rm.append(ax.plot(self.Cx_data["Cx_X"][-self.num_players:, 0], self.Cx_data["Cx_X"]
        #                   [-self.num_players:, 1], ".", color="tab:brown",  mew=10))

        ax.plot(
            self.Cx_data["Cx_X"][:, 0],
            self.Cx_data["Cx_X"][:, 1],
            "*",
            color="red",
            mew=0.2,
        )

        # ax.clabel(CS2, CS2.levels, inline=True, fmt=fmt, fontsize=10)
        ax.axis("equal")
        ax.set_title(
            "Iteration "
            + str(self.n_iter)
            + " Goose Iter "
            + str(self.goose_step)
            + " Agent "
            + str(self.Cx_agent_key)
        )
        # plt.show()
        # plt.savefig("check1.png")
        return rm

    def plot1Dsafe_GP(self, f_handle):
        ax = f_handle.axes[0]
        x = self.grid_V
        # observed_pred = self.Cx_model.likelihood(self.Cx_model(test_x))
        # posterior is only avaialble with botorch and not in gpytorch
        observed_posterior = self.Cx_model.posterior(x)
        # plt.plot(observed_pred.mean.detach().numpy())
        lower, upper = observed_posterior.mvn.confidence_region()
        temp1 = lower * (1 + self.Cx_beta) / 2 + upper * (1 - self.Cx_beta) / 2
        upper = upper * (1 + self.Cx_beta) / 2 + lower * (1 - self.Cx_beta) / 2
        lower = temp1
        w = upper - lower
        # print(self.prev_w - w)
        self.prev_w = upper - lower
        ax.plot(
            x[:, 0].numpy(), self.true_constraint_function.numpy(), color="tab:orange"
        )
        rm = []
        rm.append(
            [
                ax.fill_between(
                    x[:, 0].numpy(),
                    lower.detach().numpy(),
                    upper.detach().numpy(),
                    alpha=0.5,
                    color="tab:blue",
                )
            ]
        )

        rm.append(
            ax.plot(
                x[:, 0].numpy(),
                observed_posterior.mean.detach().numpy(),
                color="tab:blue",
                label="Cx-mean",
            )
        )
        # for lines in self.lines:
        #     rm.append(ax.plot(lines["opti"]["left"]["X"], lines["opti"]
        #                        ["left"]["Y"], color="tab:olive"))
        #     rm.append(ax.plot(lines["opti"]["right"]["X"], lines["opti"]
        #                        ["right"]["Y"], color="tab:olive"))
        #     rm.append(ax.plot(lines["pessi"]["left"]["X"], lines["pessi"]
        #                        ["left"]["Y"], color="tab:pink"))
        #     rm.append(ax.plot(lines["pessi"]["right"]["X"], lines["pessi"]
        #                        ["right"]["Y"], color="tab:pink"))
        # n_agents = len(self.lines)
        if self.Cx_data["Cx_X"].shape[0] < 2:
            ax.plot(
                self.Cx_data["Cx_X"][:-1].numpy(),
                self.Cx_data["Cx_Y"][:-1].numpy(),
                "*",
                color="red",
                mew=2,
            )
            ax.plot(
                self.Cx_data["Cx_X"][-1].numpy(),
                self.Cx_data["Cx_Y"][-1].numpy(),
                "*",
                color="yellow",
                mew=2,
            )
        else:
            ax.plot(
                self.Cx_data["Cx_X"][:-1][:, 0].numpy(),
                self.Cx_data["Cx_Y"][:-1].numpy(),
                "*",
                color="red",
                mew=2,
            )
            ax.plot(
                self.Cx_data["Cx_X"][-1:][:, 0].numpy(),
                self.Cx_data["Cx_Y"][-1:].numpy(),
                "*",
                color="yellow",
                mew=2,
            )

        for key in self.agent_current_loc:
            data = self.agent_current_loc[key][-1]
            y_mean = (
                self.Cx_model.posterior(data.reshape(-1, 2)).mvn.mean.detach().numpy()
            )
            rm.append(
                [
                    ax.text(
                        data[0] - 2 * self.step_size,
                        y_mean,
                        str(key),
                        color="tab:brown",
                        fontsize=12,
                        weight="bold",
                    )
                ]
            )

            # if not (self.Cx_agent_key in self.agent_current_loc):
            #     self.agent_current_loc[self.Cx_agent_key] = []
            # self.agent_current_loc[self.Cx_agent_key].append(self.Cx_data["loc"])
            # for agent_key in self.agent_current_loc:
            #     data = self.agent_current_loc[agent_key][-1]
            #     y_mean = self.Cx_model.posterior(
            #         data.reshape(-1, 2)).mvn.mean.detach().numpy()
            #     # print("Reached loc", data, "for agent ", agent_key)
            #     rm.append([ax.text(data[0], y_mean, str(
            #         agent_key), color="red", fontsize=12)])
        # ax.plot(self.Cx_data["Cx_X"], self.Cx_data["Cx_Y"], "*", color="red", mew=2)
        # plt_bound.lower.Xleft.set_data([self.S_pessi.Xleft.detach().numpy(), Safe.Xleft.detach().numpy()], [
        #     self.constraint, Safe_bound.lower.Xleft.detach().numpy()])
        # plt_bound.lower.Xright.set_data([self.S_pessi.Xright.detach().numpy(), Safe.Xright.detach().numpy()], [
        #     self.constraint, Safe_bound.lower.Xright.detach().numpy()])
        # plt_bound.upper.Xleft.set_data([self.S_opti.Xleft.detach().numpy(), Safe.Xleft.detach().numpy()], [
        #     self.constraint, Safe_bound.upper.Xleft.detach().numpy()])
        # plt_bound.upper.Xright.set_data([self.S_opti.Xright.detach().numpy(), Safe.Xright.detach().numpy()], [
        #     self.constraint, Safe_bound.upper.Xright.detach().numpy()])

        ax.legend(loc="upper left")
        # Draw pessimistic, optimistic and reachable sets
        if self.use_goose:
            k = -0.5
            w = 0.06
            for init_loc in self.safe_boundary:
                rm.append(
                    ax.plot(
                        [init_loc.numpy()[0] - w, init_loc.numpy()[0] + w],
                        [k - 0.35, k - 0.35],
                        color="cyan",
                        linewidth=6.0,
                    )
                )
            for key in self.pessi_boundary:
                rm.append(
                    ax.plot(
                        [
                            self.grid_V[self.pessi_boundary[key][0]][0] - w,
                            self.grid_V[self.pessi_boundary[key][-1]][0] + w,
                        ],
                        [k - 0.5, k - 0.5],
                        color="green",
                        linewidth=6.0,
                    )
                )
            for key in self.opti_boundary:
                rm.append(
                    ax.plot(
                        [
                            self.grid_V[self.opti_boundary[key][0]][0] - w,
                            self.grid_V[self.opti_boundary[key][-1]][0] + w,
                        ],
                        [k - 0.65 - key * 0.1, k - 0.65 - key * 0.1],
                        color="gold",
                        linewidth=6.0,
                    )
                )
            # for key in self.optimal_feasible_boundary:
            #     rm.append(ax.plot([self.grid_V[self.optimal_feasible_boundary[key][0]-1][0], self.grid_V[self.optimal_feasible_boundary[key][-1]+1][0]], [
            #         k-0.80, k-0.80], color="mediumpurple", linewidth=6.0))

            ax.axhline(y=self.constraint, linestyle="--", color="k")

        ax.set_title(
            "Iteration "
            + str(self.n_iter)
            + " Goose Iter "
            + str(self.goose_step)
            + " Agent "
            + str(self.Cx_agent_key)
        )

        return rm

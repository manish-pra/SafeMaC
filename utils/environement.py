import os
import pickle
import sys
from datetime import datetime

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import torch
import yaml
from botorch.models import SingleTaskGP
from gpytorch.kernels import (LinearKernel, MaternKernel,
                              PiecewisePolynomialKernel, PolynomialKernel,
                              RBFKernel, ScaleKernel)
from mpl_toolkits.axes_grid1 import make_axes_locatable
# import sys
# print(sys.path)
from openweathermap.KGS_environment import (GridFunction, get_gorillas_density,
                                            get_jungle_weather)

# import utils


class GridWorld:
    """[summary] ls_Fx: lengthscale of"""

    def __init__(self, env_params, common_params, env_dir) -> None:
        self.env_dim = common_params["dim"]
        self.N = env_params["shape"]["x"]
        self.Nx = env_params["shape"]["x"]
        self.Ny = env_params["shape"]["y"]
        self.Cx_beta = env_params["Cx_beta"]
        self.Fx_beta = env_params["Fx_beta"]
        self.Fx_lengthscale = env_params["Fx_lengthscale"]
        self.Fx_noise = env_params["Fx_noise"]
        self.Cx_lengthscale = env_params["Cx_lengthscale"]
        self.Cx_noise = env_params["Cx_noise"]
        self.n_players = env_params["n_players"]
        self.grid_V = grid(
            env_params["shape"], env_params["step_size"], env_params["start"]
        )
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
        self.env_dir = env_dir
        env_file_path = env_dir + env_params["env_file_name"]
        self.constraint = common_params["constraint"]
        self.epsilon = common_params["epsilon"]

        self.env_data = {}
        if env_params["generate"] == True:
            self.__Cx = self.__true_constraint_sampling()
            self.__Fx = self.__true_density_sampling()
            a, b = self.__Cx, self.__Fx
            self.__init_safe = {}
            init = self.__get_safe_init()
            self.__init_safe["loc"] = init[0]
            self.__init_safe["idx"] = init[1]
            self.env_data["Cx"] = self.__Cx
            self.env_data["Fx"] = self.__Fx
            self.env_data["init_safe"] = self.__init_safe
            self.plot()
            a_file = open(env_file_path, "wb")
            pickle.dump(self.env_data, a_file)
            a_file.close()
        elif env_params["generate"] == False:
            k = open(env_file_path, "rb")
            self.env_data = pickle.load(k)
            k.close()
            length = self.env_data["Cx"].shape[0]
            if length != self.Nx:  # for downsampling
                ratio = int(length / self.Nx)
                print(ratio, length, self.Nx)
                # .reshape(-1, ratio)[:, 0].reshape(-1)
                self.__Cx = self.env_data["Cx"]
                # .reshape(-1, ratio)[:, 0].reshape(-1)
                self.__Fx = self.env_data["Fx"]
                init = {}
                init["loc"] = []
                init["idx"] = []
                for init_agent in self.env_data["init_safe"]["loc"]:
                    idx = torch.abs((self.grid_V - init_agent)[:, 0]).argmin()
                    init["loc"].append(self.grid_V[idx])
                    init["idx"].append(idx)
                init["idx"] = torch.stack(init["idx"]).reshape(-1)
                self.__init_safe = init
                if self.Ny != 1:
                    self.__init_safe = self.env_data["init_safe"]
            else:
                self.__Cx = self.env_data["Cx"]
                self.__Fx = self.env_data["Fx"]
                self.__init_safe = self.env_data["init_safe"]
            print(self.env_data["init_safe"])
            # self.plot() # since model is generated only in generate mode , change to another plot latter
        elif env_params["generate"] == "gorilla":
            self.__Cx = get_jungle_weather("clouds")
            constraint = GridFunction((30, 30), get_jungle_weather("clouds"))
            density = GridFunction((30, 30), get_gorillas_density())
            # density.downsample(deltas=(3, 3))
            sx = env_params["shape"]["x"]
            density.upsample(current=True, factors=(sx / 100, sx / 100))
            constraint.upsample(factors=(sx / 10, sx / 10))
            self.__Fx, self.__Cx = torch.Tensor(density.current_function).reshape(
                -1
            ), -torch.Tensor(constraint.current_function).reshape(-1)
            self.__Cx -= self.__Cx.min()
            self.__Cx /= self.__Cx.max()
            a, b = self.__Fx, self.__Cx
            self.__init_safe = {}
            init = self.__get_safe_init()
            self.__init_safe["loc"] = init[0]
            self.__init_safe["idx"] = init[1]
            self.env_data["Cx"] = self.__Cx
            self.env_data["Fx"] = self.__Fx
            self.env_data["init_safe"] = self.__init_safe
            self.plot()
            a_file = open(env_file_path, "wb")
            pickle.dump(self.env_data, a_file)
            a_file.close()

        elif env_params["generate"] == "walls":
            # def dist(a, b): return torch.pow(torch.norm(a - b), 0.6)
            def dist(a, b):
                return 1 * (1 / (1 + torch.exp(-1.5 * torch.norm(a - b))) - 0.5)

            M = torch.zeros(self.grid_V.shape[0], self.grid_V.shape[0])
            for i, ele_i in enumerate(self.grid_V):
                for j in range(i):
                    M[i, j] = dist(ele_i, self.grid_V[j])
                    M[j, i] = M[i, j]
            # start = [torch.Tensor([-1.1, 1.0]),  torch.Tensor([0.1, 1.0]),
            #          torch.Tensor([0.1, -0.50]), torch.Tensor([0.3, -1.40])]
            # end = [torch.Tensor([-1.4, -0.80]),
            #        torch.Tensor([-0.2, -0.80]), torch.Tensor([-1.4, -0.80]), torch.Tensor([-0.5, -1.8])]
            if self.Ny != 1:
                start = [
                    torch.Tensor([-1.3, 1.0]),
                    torch.Tensor([0.2, -0.70]),
                    torch.Tensor([0.3, -1.40]),
                ]
                end = [
                    torch.Tensor([-1.4, -0.80]),
                    torch.Tensor([-1.4, -0.80]),
                    torch.Tensor([-0.5, -1.8]),
                ]
                # config 9
                # start = [torch.Tensor([-1.3, 1.0]),  torch.Tensor([0.4, 1.0]),
                #          torch.Tensor([0.2, -0.30]), torch.Tensor([0.3, -1.40])]
                # end = [torch.Tensor([-1.4, -0.80]),
                #        torch.Tensor([0.3, -0.80]), torch.Tensor([-0.5, -0.40]), torch.Tensor([-0.5, -1.8])]
                # start = [torch.Tensor([-1.3, 1.0]),  torch.Tensor([0.4, 1.0]),
                #          torch.Tensor([0.2, -0.70]), torch.Tensor([0.3, -1.40])]
                # end = [torch.Tensor([-1.4, -0.80]),
                #        torch.Tensor([0.3, -0.80]), torch.Tensor([-1.4, -0.80]), torch.Tensor([-0.5, -1.8])]
                # start = [torch.Tensor([-1.4, 1.0]),  torch.Tensor([0.1, 1.0]),
                #          torch.Tensor([0.1, -0.80]), torch.Tensor([0.3, -1.40])]
                # end = [torch.Tensor([-1.4, -0.80]),
                #        torch.Tensor([0.1, -0.80]), torch.Tensor([-1.4, -0.80]), torch.Tensor([-0.5, -1.8])]
                # config 6
                # start = [torch.Tensor([-0.0, -0.50]),
                #          torch.Tensor([-0.2, 0.2]), torch.Tensor([0.9, -1.2]),
                #          torch.Tensor([0.2, -1.2]), torch.Tensor([0.7, 0.9]),
                #          torch.Tensor([0.7, 0.6]), torch.Tensor([0.2, -1.9])]
                # end = [torch.Tensor([-1, -0.60]), torch.Tensor([-0.3, -0.8]),
                #        torch.Tensor([0.1, -1.3]), torch.Tensor([0.1, -1.5]
                #                                                ), torch.Tensor([0.6, 0.5]),
                #        torch.Tensor([0.5, 0.5]), torch.Tensor([0.1, -2.0])]
                # config 5
                # start = [torch.Tensor([-0.2, -0.50])]
                # end = [torch.Tensor([-0.6, -0.80])]
                # config 4
                # start = [torch.Tensor([-0.2, -0.50]),
                #          torch.Tensor([0.2, 0.50])]
                # end = [torch.Tensor([-0.6, -0.80]), torch.Tensor([-0.2, 0.20])]
                # config 3
                # start = []
                # end = []
                # d = 1.2
                # for h in range(0, 2):
                #     start += [torch.Tensor([0.5 - h*d, 0.5 - k*d])
                #               for k in range(0, 2)]
                #     # [torch.Tensor([-0.6, -0.80]), torch.Tensor([-0.2, 0.20])]
                #     end += [torch.Tensor([0.4 - h*d, 0.4 - k*d])
                #             for k in range(0, 2)]
                # config 2
                # start = []
                # end = []
                # d = 1
                # for h in range(0, 3):
                #     start += [torch.Tensor([0.5 - h*d, 0.5 - k*d])
                #               for k in range(0, 2)]
                #     # [torch.Tensor([-0.6, -0.80]), torch.Tensor([-0.2, 0.20])]
                #     end += [torch.Tensor([0.4 - h*d, 0.4 - k*d])
                #             for k in range(0, 2)]
                # start += [torch.Tensor([-1.5, 0.5]),
                #           torch.Tensor([0.5, -0.5]), torch.Tensor([-0.3, -1.5])]
                # end += [torch.Tensor([-1.6, -0.5]),
                #         torch.Tensor([-0.5, -0.6]), torch.Tensor([-0.4, -2.0])]
                # config 1
                # start = [torch.Tensor([-0.0, -0.50]),
                #          torch.Tensor([-0.2, 0.2])]
                # end = [torch.Tensor([-1, -0.60]), torch.Tensor([-0.3, -0.8])]
                # config 0
                # start = [torch.Tensor([-0.0, -0.50]),
                #          torch.Tensor([-0.2, 0.2]), torch.Tensor([0.9, -1.2]), torch.Tensor([0.2, -1.2])]
                # end = [torch.Tensor([-1, -0.60]), torch.Tensor([-0.3, -0.8]),
                #        torch.Tensor([0.1, -1.3]), torch.Tensor([0.1, -1.5])]
            else:
                start = [
                    torch.Tensor([7, -2.0]),
                    torch.Tensor([25, -2.0]),
                    torch.Tensor([-1.8, -2.0]),
                    torch.Tensor([2.0, -2.0]),
                ]

                end = [
                    torch.Tensor([5, -2.0]),
                    torch.Tensor([23, -2.0]),
                    torch.Tensor([-2.0, -2.0]),
                    torch.Tensor([1.6, -2.0]),
                ]
            wall_list = []
            for st, ed in zip(start, end):
                wall_list.append(add_a_wall(st, ed, self.grid_V)[1])
            wall_idx = torch.unique(torch.hstack(wall_list))
            non_wall_idx = list(
                set(torch.arange(self.Nx * self.Ny).tolist()) -
                set(wall_idx.tolist())
            )
            safe_idx = torch.LongTensor(non_wall_idx)
            safe_M = M[torch.meshgrid(safe_idx, wall_idx)]
            safe_val, _ = torch.min(safe_M, 1)
            un_safe_M = M[torch.meshgrid(wall_idx, safe_idx)]
            un_safe_val, _ = torch.min(un_safe_M, 1)
            sort_order = torch.argsort(torch.cat([wall_idx, safe_idx]))
            min_val = torch.cat([-un_safe_val, safe_val])
            self.__Cx = min_val[sort_order] + self.constraint
            # self.__true_density_sampling()
            # min_val[sort_order] + 2.5
            self.__Fx = self.__true_density_sampling()
            self.__init_safe = {}
            # idx = torch.LongTensor([95, 830, 440])
            # init = [loc.reshape(-1) for loc in self.grid_V[idx]], idx
            init = self.__get_safe_init()

            const_tr = self.__Cx.reshape(self.Nx, self.Ny)
            binary_wall = torch.cat(
                [torch.zeros_like(un_safe_val), torch.ones_like(safe_val)]
            )[sort_order].reshape(self.Nx, self.Ny)
            self.__init_safe["loc"] = init[0]
            self.__init_safe["idx"] = init[1]
            self.env_data["Cx"] = self.__Cx
            self.env_data["Fx"] = self.__Fx
            self.env_data["init_safe"] = self.__init_safe
            self.plot()
            a_file = open(env_file_path, "wb")
            pickle.dump(self.env_data, a_file)
            a_file.close()
        else:
            self.__Cx = self.__safety_function(self.V)
            self.__Fx = self.__generate_gp(self.V)
            self.__init_safe = [
                torch.Tensor([1.80]).reshape(-1, 1),
                torch.Tensor([2.40]).reshape(-1, 1),
                torch.Tensor([6.4]).reshape(-1, 1),
            ]
            self.env_data["Cx"] = self.__Cx
            self.env_data["Fx"] = self.__Fx
            self.env_data["init_safe"] = self.__init_safe
            # a_file = open(env_file_path, "wb")
            # pickle.dump(self.env_data, a_file)
            # a_file.close()

    def plot(self):
        if self.Ny == 1:
            # self.plot1D()
            a = 1
        else:
            self.plotContour()

    def plot1D(self):
        f, ax = plt.subplots()
        x = self.grid_V
        observed_posterior = self.Cx_model.posterior(x)
        # plt.plot(observed_pred.mean.detach().numpy())
        lower, upper = observed_posterior.mvn.confidence_region()
        lower = lower * (1 + self.Cx_beta) / 2 + upper * (1 - self.Cx_beta) / 2
        upper = upper * (1 + self.Cx_beta) / 2 + lower * (1 - self.Cx_beta) / 2
        ax.plot(x[:, 0].numpy(), self.__Cx, color="tab:orange")
        ax.fill_between(
            x[:, 0].numpy(),
            lower.detach().numpy(),
            upper.detach().numpy(),
            alpha=0.5,
            color="tab:blue",
        )
        ax.plot(
            x[:, 0].numpy(),
            observed_posterior.mean.detach().numpy(),
            color="tab:blue",
            label="Cx-mean",
        )

        for init_idx in self.__init_safe["idx"]:
            ax.plot(
                x[init_idx, 0],
                observed_posterior.mean.detach().numpy()[init_idx],
                "*",
                color="red",
                mew=2,
            )
        ax.axhline(y=self.constraint, linestyle="--", color="k")

        Fx_observed_posterior = self.Fx_model.posterior(x)
        # ax.plot(observed_pred.mean.detach().numpy())
        Fx_lower, Fx_upper = Fx_observed_posterior.mvn.confidence_region()
        Fx_lower = Fx_lower + 2
        Fx_upper = Fx_upper + 2
        temp = Fx_lower * (1 + self.Fx_beta) / 2 + \
            Fx_upper * (1 - self.Fx_beta) / 2
        Fx_upper = Fx_upper * (1 + self.Fx_beta) / 2 + \
            Fx_lower * (1 - self.Fx_beta) / 2
        Fx_lower = temp
        ax.plot(x[:, 0].numpy(), self.__Fx, color="tab:orange")
        ax.fill_between(
            x[:, 0].numpy(),
            Fx_lower.detach().numpy(),
            Fx_upper.detach().numpy(),
            alpha=0.5,
            color="tab:purple",
        )
        ax.plot(
            x[:, 0].numpy(),
            Fx_observed_posterior.mean.detach().numpy() + 2,
            color="tab:purple",
            label="Fx-mean",
        )
        # plt.show()
        plt.savefig(self.env_dir + "env.png")
        # f.savefig("init_setup.png")

    def plotContour(self):
        def fmt(x):
            s = f"{x:.1f}"
            if s.endswith("0"):
                s = f"{x:.0f}"
            return rf"{s} \%" if plt.rcParams["text.usetex"] else f"{s} %"

        x = self.grid_V.transpose(0, 1).reshape(-1, self.Nx, self.Ny)[0]
        y = self.grid_V.transpose(0, 1).reshape(-1, self.Nx, self.Ny)[1]
        z = self.__Cx.reshape(self.Nx, self.Ny)
        z2 = self.__Fx.reshape(self.Nx, self.Ny)
        # f, ax = plt.subplots()
        f, ax = plt.subplots(figsize=(4.0 / 2.54, 4.0 / 2.54))

        levels = 20
        CS1 = ax.contour(
            x.numpy(), y.numpy(), z2.numpy(), levels=levels, linewidths=0.01
        )
        CS = ax.contour(
            x.numpy(),
            y.numpy(),
            z.numpy(),
            np.array([0.26, 0.31, 0.35, 0.38, 0.40, 0.43, 0.47]),
            colors=[
                "grey",
                "dimgrey",
                "dimgray",
                "black",
                "dimgray",
                "dimgrey",
                "grey",
            ],
            linestyles="dashed",
            linewidths=[0.6, 0.9, 1.2, 1.5, 1.2, 0.9, 0.6],
        )
        plt.contourf(x.numpy(), y.numpy(), z2.numpy(),
                     cmap=plt.cm.viridis, alpha=0.5)
        # ax.clabel(CS, CS.levels, inline=True, fmt=fmt, fontsize=10)
        # CS2.cmap.set_over('red')
        # CS2.cmap.set_under('blue')
        # CS2.cmap.set_over('red')
        # CS2.cmap.set_under('blue')
        divider = make_axes_locatable(plt.gca())
        cax = divider.append_axes("right", "5%", pad="2%")
        cbar = plt.colorbar(cax=cax, aspect=60)
        cbar.set_ticks(ticks=[])
        x = 0.0
        cbar.ax.text(x, 0.5 / 32, "L", size=7)
        cbar.ax.text(x, 7 / 32, "H", size=7)
        cbar.ax.tick_params(labelsize=6)

        # f.colorbar(CS2)
        # cbar.ax.set_xticklabels(['Low', 'High'])
        # for init_pt in self.__init_safe["loc"]:
        #     ax.plot(init_pt[0], init_pt[1], "*", color="red", mew=2)
        # ax.clabel(CS2, CS2.levels, inline=True, fmt=fmt, fontsize=10)
        ax.yaxis.set_ticklabels([])
        ax.yaxis.set_ticks([])
        ax.xaxis.set_ticklabels([])
        ax.xaxis.set_ticks([])
        # plt.axis("equal")
        # plt.axis('off')
        plt.tick_params(
            axis="x",  # changes apply to the x-axis
            which="both",  # both major and minor ticks are affected
            bottom=False,  # ticks along the bottom edge are off
            top=False,  # ticks along the top edge are off
            labelbottom=False,
        )  # labels along the bottom edge are off
        plt.tick_params(
            axis="x",  # changes apply to the x-axis
            which="both",  # both major and minor ticks are affected
            bottom=False,  # ticks along the bottom edge are off
            top=False,  # ticks along the top edge are off
            labelbottom=False,
        )  # labels along the bottom edge are off
        # plt.show()
        # plt.savefig(self.env_dir + 'env.png')
        plt.tight_layout(pad=0)
        # plt.grid(axis='y')
        plt.savefig(self.env_dir + "gorilla-env.pdf")
        # self.plot3D()

    def goriplotContour(self):
        def fmt(x):
            s = f"{x:.1f}"
            if s.endswith("0"):
                s = f"{x:.0f}"
            return rf"{s} \%" if plt.rcParams["text.usetex"] else f"{s} %"

        x = self.grid_V.transpose(0, 1).reshape(-1, self.Nx, self.Ny)[0]
        y = self.grid_V.transpose(0, 1).reshape(-1, self.Nx, self.Ny)[1]
        z = self.__Cx.reshape(self.Nx, self.Ny)
        z2 = self.__Fx.reshape(self.Nx, self.Ny)
        f, ax = plt.subplots()

        levels = 20
        CS1 = ax.contour(
            x.numpy(), y.numpy(), z2.numpy(), levels=levels, linewidths=0.01
        )
        CS = ax.contour(
            x.numpy(),
            y.numpy(),
            z.numpy(),
            np.array([0.26, 0.31, 0.35, 0.38, 0.40, 0.43, 0.47]),
            colors=[
                "grey",
                "dimgrey",
                "dimgray",
                "black",
                "dimgray",
                "dimgrey",
                "grey",
            ],
            linestyles="dashed",
            linewidths=[1.1, 1.3, 1.6, 2.0, 1.6, 1.3, 1.1],
        )
        plt.contourf(x.numpy(), y.numpy(), z2.numpy(),
                     cmap=plt.cm.viridis, alpha=0.5)
        # ax.clabel(CS, CS.levels, inline=True, fmt=fmt, fontsize=10)
        # CS2.cmap.set_over('red')
        # CS2.cmap.set_under('blue')
        # CS2.cmap.set_over('red')
        # CS2.cmap.set_under('blue')
        divider = make_axes_locatable(plt.gca())
        cax = divider.append_axes("right", "5%", pad="2%")
        cbar = plt.colorbar(cax=cax)
        cbar.set_ticks(ticks=[0, 0.24])
        cbar.set_ticklabels(["L", "H"])
        cbar.ax.tick_params(labelsize=20)

        # f.colorbar(CS2)
        # cbar.ax.set_xticklabels(['Low', 'High'])
        # for init_pt in self.__init_safe["loc"]:
        #     ax.plot(init_pt[0], init_pt[1], "*", color="red", mew=2)
        # ax.clabel(CS2, CS2.levels, inline=True, fmt=fmt, fontsize=10)
        ax.yaxis.set_ticklabels([])
        ax.yaxis.set_ticks([])
        ax.xaxis.set_ticklabels([])
        ax.xaxis.set_ticks([])
        # plt.axis("equal")
        # plt.axis('off')
        plt.tick_params(
            axis="x",  # changes apply to the x-axis
            which="both",  # both major and minor ticks are affected
            bottom=False,  # ticks along the bottom edge are off
            top=False,  # ticks along the top edge are off
            labelbottom=False,
        )  # labels along the bottom edge are off
        plt.tick_params(
            axis="y",  # changes apply to the x-axis
            which="both",  # both major and minor ticks are affected
            bottom=False,  # ticks along the bottom edge are off
            top=False,  # ticks along the top edge are off
            labelbottom=False,
        )  # labels along the bottom edge are off
        # plt.show()
        # plt.savefig(self.env_dir + 'env.png')
        plt.tight_layout(pad=0)
        # plt.grid(axis='y')
        plt.savefig(self.env_dir + "gorilla-env.pdf")
        # self.plot3D()

    def plot3D(self):
        x = self.grid_V.transpose(0, 1).reshape(-1, self.Nx, self.Ny)[0]
        y = self.grid_V.transpose(0, 1).reshape(-1, self.Nx, self.Ny)[1]
        z = self.__Cx.reshape(self.Nx, self.Ny)
        z2 = self.__Fx.reshape(self.Nx, self.Ny)
        ax = plt.axes(projection="3d")
        ax.plot_surface(x.numpy(), y.numpy(), z.numpy())
        ax.plot_surface(x.numpy(), y.numpy(), z2.numpy())
        # ax.plot_surface(x.numpy(), y.numpy(), torch.arange(
        #     0, 10201).reshape(self.Nx, self.Ny).numpy())
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        plt.show()
        # plt.savefig(self.env_dir + 'env.png')

    def get_env_data(self):
        return self.env_data

    def __safety_function(self, individuals):
        result = []
        for x in individuals:
            result.append(
                np.exp(-((x[0] - 2) ** 2))
                + np.exp(-((x[0] - 6) ** 2) / 5)
                + 1 / (x[0] ** 2 + 1)
            )
        return torch.tensor(result)

    def __generate_gp_noise(self):
        dist = torch.distributions.multivariate_normal.MultivariateNormal(
            torch.zeros(100), torch.eye(100)
        )
        return dist.sample()

    # def __generate_gp(self):
    #     X_train = torch.linspace(-2, 10, self.N).reshape(-1, 1)
    #     Y_train = torch.zeros(X_train.shape)
    #     model = SingleTaskGP(X_train, Y_train)
    #     model.covar_module.base_kernel.lengthscale = self.ls_Fx
    #     model.likelihood.noise = self.mu_Fx
    #     return model.posterior(X_train).sample().reshape(-1, 1)

    def __generate_gp(self, individuals):
        result = []
        for x in individuals:
            result.append(
                np.exp(-((x[0] - 8.5) ** 2))
                + np.exp(-((x[0] - 1.5) ** 2) / 10)
                + 2
                + 1 / (x[0] ** 4 + 1)
            )
        return torch.tensor(result)

    def idxfromloc(self, loc):
        diff = self.grid_V - loc
        idx = torch.arange(self.grid_V.shape[0])[
            torch.isclose(diff, torch.zeros(2)).all(dim=1)
        ].item()
        return idx

    def get_multi_density_observation(self, sets):
        train = {}
        train["Fx_Y"] = []
        for locs in sets:
            Fx_Y = self.get_density_observation(locs)
            train["Fx_Y"].append(Fx_Y)
        return train["Fx_Y"]

    def get_multi_constraint_observation(self, sets):
        train = {}
        train["Cx_Y"] = []
        for locs in sets:
            Cx_Y = self.get_constraint_observation(locs)
            train["Cx_Y"].append(Cx_Y)
        return train["Cx_Y"]

    def get_density_observation(self, loc):
        obs_Fx = []
        noise = torch.normal(
            mean=torch.zeros(2, 1), std=torch.ones(2, 1) * self.Fx_noise
        )
        obs_Fx.append(self.__Fx[self.idxfromloc(loc)] + noise[0])
        return torch.stack(obs_Fx).reshape(-1, 1)

    def get_constraint_observation(self, loc):
        obs_Cx = []
        noise = torch.normal(
            mean=torch.zeros(2, 1), std=torch.ones(2, 1) * self.Fx_noise
        )
        obs_Cx.append(self.__Cx[self.idxfromloc(loc)] + noise[0])
        return torch.stack(obs_Cx).reshape(-1, 1)

    def get_multi_observation_idx(self, sets):
        train = {}
        train["Fx_Y"] = []
        train["Cx_Y"] = []
        for idxs in sets:
            Fx_Y, Cx_Y = self.get_observation_idx(idxs)
            train["Fx_Y"].append(Fx_Y)
            train["Cx_Y"].append(Cx_Y)
        return train["Fx_Y"], train["Cx_Y"]

    def get_observation_idx(self, idxs):
        obs_Fx = []
        obs_Cx = []
        noise = torch.normal(
            mean=torch.zeros(2, 1), std=torch.ones(2, 1) * self.Fx_noise
        )
        for idx in idxs:
            obs_Fx.append(self.__Fx[idx] + noise[0])
            obs_Cx.append(self.__Cx[idx] + noise[0])
        return torch.stack(obs_Fx).reshape(-1, 1), torch.stack(obs_Cx).reshape(-1, 1)

    def get_disk_constraint_observation(self, disc_nodes):
        noise = torch.normal(
            mean=torch.zeros(len(disc_nodes), 1),
            std=torch.ones(len(disc_nodes), 1) * self.Fx_noise,
        )
        obs_Cx = [self.__Cx[node] + noise[idx]
                  for idx, node in enumerate(disc_nodes)]
        disc_pts = self.grid_V[disc_nodes]
        return torch.stack(obs_Cx).reshape(-1, 1), disc_pts

    def get_disk_density_observation(self, disc_nodes):
        noise = torch.normal(
            mean=torch.zeros(len(disc_nodes), 1),
            std=torch.ones(len(disc_nodes), 1) * self.Fx_noise,
        )
        obs_Fx = [self.__Fx[node] + noise[idx]
                  for idx, node in enumerate(disc_nodes)]
        disc_pts = self.grid_V[disc_nodes]
        return torch.stack(obs_Fx).reshape(-1, 1), disc_pts

    def get_true_safety_func(self):
        return self.__Cx

    def get_true_objective_func(self):
        return self.__Fx

    def __true_density_sampling(self):
        # torch.Tensor([0]).reshape(-1, 1)
        self.Fx_X = (torch.rand(2) * 10).reshape(-1, self.env_dim)
        self.Fx_Y = torch.zeros(self.Fx_X.shape[0], 1)
        self.Fx_model = SingleTaskGP(
            self.Fx_X, self.Fx_Y, covar_module=self.Fx_covar_module
        )
        self.Fx_model.covar_module.base_kernel.lengthscale = self.Fx_lengthscale
        self.Fx_model.likelihood.noise = self.Fx_noise
        density = self.Fx_model.posterior(self.grid_V).sample().reshape(-1)
        if density.min() > -3:
            return density + 3
        else:
            return density + density.min()

    def __true_constraint_sampling(self):
        # torch.Tensor([0]).reshape(-1, 1)
        self.Cx_X = (torch.rand(6) * 10).reshape(-1, self.env_dim)
        self.Cx_Y = torch.zeros(self.Cx_X.shape[0], 1)
        # self.Cx_model = SingleTaskGP(
        #     self.Cx_X, self.Cx_Y, covar_module=self.Cx_covar_module)
        self.Cx_model = SingleTaskGP(
            self.Cx_X, self.Cx_Y, covar_module=self.Cx_covar_module
        )
        self.Cx_model.covar_module.base_kernel.lengthscale = self.Cx_lengthscale
        self.Cx_model.likelihood.noise = self.Cx_noise
        acq_density = self.Cx_model.posterior(self.grid_V).sample().reshape(-1)
        ret = acq_density - torch.mean(acq_density) + 0.8
        # plt.plot(torch.diff(ret)/0.12)
        # plt.plot(ret)
        # plt.show()
        return ret

    def __get_safe_init(self):
        opt_set = self.__Cx - self.epsilon > self.constraint
        p = opt_set.view(-1) * 1
        for i in range(2):
            p_forward = torch.cat([p[1:], torch.zeros(1)], dim=0)
            p_backward = torch.cat([torch.zeros(1), p[:-1]], dim=0)
            p = p_forward * p_backward
        dist = p / torch.sum(p)
        # number of places to put agents is less than number of agents
        if torch.sum(p) + 1 < self.n_players:
            raise "Non valid init"
        idx = dist.multinomial(self.n_players, replacement=False)
        return [loc.reshape(-1) for loc in self.grid_V[idx]], idx

    def get_safe_init(self):
        """_summary_

        Returns:
            _type_: _description_
        """
        return self.__init_safe


def nodes_to_states(nodes, world_shape, step_size):
    """Convert node numbers to physical states.
    Parameters
    ----------
    nodes: np.array
        Node indices of the grid world
    world_shape: tuple
        The size of the grid_world
    step_size: np.array
        The step size of the grid world
    Returns
    -------
    states: np.array
        The states in physical coordinates
    """
    nodes = torch.as_tensor(nodes)
    step_size = torch.as_tensor(step_size)
    return (
        torch.vstack(
            ((nodes // world_shape["y"]), (nodes % world_shape["y"]))).T
        * step_size
    )


def grid(world_shape, step_size, start_loc):
    """
    Creates grids of coordinates and indices of state space
    Parameters
    ----------
    world_shape: tuple
        Size of the grid world (rows, columns)
    step_size: tuple
        Phyiscal step size in the grid world
    Returns
    -------
    states_ind: np.array
        (n*m) x 2 array containing the indices of the states
    states_coord: np.array
        (n*m) x 2 array containing the coordinates of the states
    """
    nodes = torch.arange(0, world_shape["x"] * world_shape["y"])
    return nodes_to_states(nodes, world_shape, step_size) + start_loc


def add_a_wall(start, end, grid_V):
    """Returns a tensor from start to end

    Args:
        start (_type_): _description_
        end (_type_): _description_
    """
    upper_corner = torch.all(torch.le(grid_V, start + 0.01), 1)
    lower_corner = torch.all(torch.ge(grid_V, end - 0.01), 1)
    wall_bool = upper_corner & lower_corner
    return grid_V[wall_bool], torch.arange(grid_V.shape[0])[wall_bool]


def dist(a, b):
    return torch.norm(a - b)


if __name__ == "__main__":
    workspace = "SafeMaC"
    with open(workspace + "/params/smcc_SafeMac_LA3D40gorilla.yaml") as file:
        params = yaml.load(file, Loader=yaml.FullLoader)
    if params["env"]["generate"]:
        for i in range(0, 10):
            # save_path = workspace + "/experiments/" + datetime.today().strftime('%d-%m-%y') + \
            #     datetime.today().strftime(
            #         '-%A')[0:4] + "/environments/env_" + str(i) + "/"
            save_path = (
                workspace
                + "/experiments/"
                + params["experiment"]["folder"]
                + "/environments/env_"
                + str(i)
                + "/"
            )
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            env = GridWorld(
                env_params=params["env"],
                common_params=params["common"],
                env_dir=save_path,
            )
    else:
        exp_name = params["experiment"]["name"]
        env_load_path = (
            workspace + "/experiments/22-02-22-Tue/environments/env_" +
            str(0) + "/"
        )
        save_path = env_load_path + "/"

        env = GridWorld(
            env_params=params["env"], common_params=params["common"], env_dir=save_path
        )

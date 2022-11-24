import pickle

import matplotlib.pyplot as plt
import numpy as np
import torch
from pandas import array

from plotting_utilities.plotting_utilities.utilities import *

# path = "SafeMaC/pretrained_data/GP"
# k = open(path + "/constraint-GP.pkl", "rb")
# filename = 'safemac-gp'
# CUTOFF_SIZE = 550

path = "SafeMaC/pretrained_data/gorilla"
k = open(path + "/constraint-gorilla.pkl", "rb")
filename = 'safemac-gorilla'
CUTOFF_SIZE = 750

# path = "SafeMaC/pretrained_data/obstacles"
# k = open(path + "/constraint-obstacles.pkl", "rb")
# filename = 'safemac-obstacles'
# CUTOFF_SIZE = 800

# path = "SafeMaC/pretrained_data/gorilla"
# k = open(path + "/unconstraint-gorilla.pkl", "rb")
# filename = 'macopt-gorilla'
# CUTOFF_SIZE = 200


# path = "SafeMaC/pretrained_data/GP"
# k = open(path + "/unconstraint-GP.pkl", "rb")
# filename = 'macopt-gp'
# CUTOFF_SIZE = 200


def set_label(label):
    if label == "smcc_DiskCenter2D":
        return "UCB", "tab:orange"
    elif label == "smcc_UCB_GP":
        return "UCB", "tab:orange"
    elif label == "smcc_MacOpt_GP_con":
        return "MacOpt-H", "tab:green"
    elif label == "smcc_MacOpt_GP_cov":
        return "MacOpt-CUB", "tab:purple"
    elif label == "smcc_UCB_gorilla":
        return "UCB", "tab:orange"
    elif label == "smcc_MacOpt_gorilla_con":
        return "MacOpt-H", "tab:green"
    elif label == "smcc_MacOpt_gorilla_cov":
        return "MacOpt-CUB", "tab:purple"
    elif "MacOpt" in label:
        return "MacOpt", "tab:blue"
    elif "SafeMac" in label:
        return "SafeMac", "tab:blue"
    elif "TwoStage" in label:
        return "Two-Stage", "tab:orange"
    elif "2stage" in label:
        return "Two-Stage", "tab:orange"
    elif "PessiMac" in label:
        return "PassiveMac", "tab:green"
    else:
        return label, "tab:blue"


def RangeInterp(vec, n_samples):
    return torch.from_numpy(
        np.interp(
            np.arange(0, n_samples) / (n_samples / vec.shape[0]),
            np.arange(0, vec.shape[0]),
            vec.reshape(-1).numpy(),
        )
    ).reshape(-1, 1)


def RangeInterpnp(vec, n_samples):
    return np.interp(
        np.arange(0, n_samples) / (n_samples / vec.shape[0]),
        np.arange(0, vec.shape[0]),
        vec.reshape(-1),
    )


def cat_with(cat_param, vec, size_zero):
    if cat_param == "nan":
        return torch.cat([vec, torch.nan * torch.ones(size_zero, 1)])
    else:
        return torch.cat([vec, vec[-1] * torch.ones(size_zero, 1)])


def get_regret_of_one_env_one_class_at_each_iter(data_dict):
    opt_val = [data_dict[exp_num]["opt_Fx"] for exp_num in data_dict]
    length = len(opt_val)
    if length != 0:
        opt_val_mean = np.sum(opt_val) / length
        traj_list = []
        opt_traj_list = []
        rbar0_traj_list = []
        reco_rbar0_traj_list = []
        rbar_eps_traj_list = []
        max_sigma_list = []
        node_list = []
        cat_param = "last"
        for traj in data_dict:
            if torch.mean(data_dict[traj]["mat_FxIX_rho_opti"]) != torch.nan:
                n_samples = (
                    data_dict[traj]["samples"]["constraint"]
                    + data_dict[traj]["samples"]["density"]
                )
                size_zero = CUTOFF_SIZE - n_samples
                size_zero2 = (
                    CUTOFF_SIZE -
                    data_dict[traj]["list_sum_max_density_sigma"].shape[0]
                )

                traj_list.append(
                    cat_with(
                        cat_param,
                        RangeInterp(
                            data_dict[traj]["mat_FxIX_rho_opti"], n_samples),
                        size_zero,
                    )
                )
                rbar_eps_traj_list.append(
                    cat_with(
                        cat_param,
                        RangeInterp(
                            data_dict[traj]["mat_FxIX_rho_Rbar_eps"], n_samples
                        ),
                        size_zero,
                    )
                )
                rbar0_traj_list.append(
                    cat_with(
                        cat_param,
                        RangeInterp(
                            data_dict[traj]["mat_FxIX_rho_Rbar0"], n_samples),
                        size_zero,
                    )
                )
                reco_rbar0_traj_list.append(
                    cat_with(
                        cat_param,
                        RangeInterp(
                            data_dict[traj]["mat_recommendation_rho_Rbar0"], n_samples
                        ),
                        size_zero,
                    )
                )
                opt_traj_list.append(
                    cat_with(
                        cat_param,
                        RangeInterp(
                            data_dict[traj]["mat_FtildexIX_rho_opti"], n_samples
                        ),
                        size_zero,
                    )
                )
                max_sigma_list.append(
                    np.concatenate(
                        [
                            RangeInterpnp(
                                data_dict[traj]["list_sum_max_density_sigma"], n_samples
                            ),
                            torch.nan * np.ones(size_zero),
                        ]
                    )
                )
                node_list.append(data_dict[traj]["nodes"]["diff"])
                normalize_Rbar0 = data_dict[traj]["normalization_factor"]["Rbar0"]
                normalize_Rbar_eps = data_dict[traj]["normalization_factor"]["Rbar_eps"]

        mat_FxIX_rho_opti = torch.hstack(traj_list).transpose(
            0, 1
        )  # shape = len_traj * num_of_traj
        mat_FtildexIX_rho_opti = torch.hstack(opt_traj_list).transpose(
            0, 1
        )  # shape = len_traj * num_of_traj
        mat_of_zero_regret = mat_FtildexIX_rho_opti - mat_FxIX_rho_opti  #
        mat_of_zero_regret = mat_of_zero_regret / normalize_Rbar0
        mat_of_coverage_Rbareps = (
            torch.hstack(rbar_eps_traj_list).transpose(
                0, 1) / normalize_Rbar_eps
        )
        mat_of_coverage_Rbar0 = (
            torch.hstack(rbar0_traj_list).transpose(0, 1) / normalize_Rbar0
        )
        mat_of_coverage_reco_Rbar0 = (
            torch.hstack(reco_rbar0_traj_list).transpose(
                0, 1) / normalize_Rbar0
        )

        max_sigma_mat = np.vstack(max_sigma_list)
        max_sigma_mean = np.nanmean(max_sigma_mat, 0)
        max_sigma_std = np.nanstd(max_sigma_mat, 0)
        mean_regret_for_env_i = torch.nanmean(
            mat_of_zero_regret, 0).reshape(1, -1)
        std_of_regret_due_to_noise_for_env_i = torch.from_numpy(
            np.nanstd(mat_of_zero_regret.numpy(), 0)
        ).reshape(-1, 1)
        return (
            torch.hstack(opt_val).reshape(-1, 1),
            mat_of_zero_regret,
            length,
            mean_regret_for_env_i,
            std_of_regret_due_to_noise_for_env_i,
            max_sigma_mean,
            max_sigma_std,
            node_list,
            mat_of_coverage_Rbareps,
            mat_of_coverage_Rbar0,
            mat_of_coverage_reco_Rbar0,
        )
    return 0, 0, 0, 0, 0, 0, 0


def plotter(plot_data, class_):

    traj_mat = [plot_data[key][class_]["traj_mat"] for key in plot_data]
    traj_env_mean = [plot_data[key][class_]["traj_env_mean"]
                     for key in plot_data]
    traj_env_std = [plot_data[key][class_]["traj_env_std"]
                    for key in plot_data]
    opt_vec = [plot_data[key][class_]["opt_vec"] for key in plot_data]

    opt_vec = torch.vstack(opt_vec)
    traj_mat = torch.vstack(traj_mat)
    traj_env_mean = torch.vstack(traj_env_mean)
    traj_env_std = torch.vstack(traj_env_std)
    sample_size = traj_mat.shape[0]

    class_mean = torch.nanmean(traj_mat, 0)
    class_std = torch.std(traj_mat, 0) / np.sqrt(traj_mat.shape[0])
    class_std_env = torch.std(traj_env_mean, 0) / \
        np.sqrt(traj_env_mean.shape[0])
    class_std_noise = torch.nanmean(traj_env_std, 0) / np.sqrt(sample_size)
    return class_mean, class_std_noise


def plotter_traj(plot_data, class_, Yobject):
    traj_mat = [plot_data[key][class_][Yobject] for key in plot_data]

    traj_mat = torch.vstack(traj_mat)
    sample_size = traj_mat.shape[0]
    class_mean = torch.nanmean(traj_mat, 0)
    class_std_noise = (
        2
        * torch.from_numpy(np.nanstd(traj_mat.numpy(), 0)).reshape(-1)
        / np.sqrt(sample_size)
    )
    return class_mean, class_std_noise


def plotter_sigma(plot_data, class_):
    sigma_mean = [plot_data[key][class_]["sigma_mean"] for key in plot_data]
    sigma_std = [plot_data[key][class_]["sigma_std"] for key in plot_data]

    sigma_mean = np.vstack(sigma_mean)
    sigma_std = np.vstack(sigma_std)

    class_mean = np.nanmean(sigma_mean, 0)
    class_std = np.nanstd(sigma_std, 0) / np.sqrt(sigma_std.shape[0])
    return class_mean, class_std


all_env_data = pickle.load(k)
k.close()
counter = 0
plot_data = {}
for env in all_env_data:
    print(env)
    # if env in ['env_0', 'env_3', 'env_4']:
    # , 6D50 'env_6', 'env_7', 'env_9'
    # env_3,env_4,env_6, env_7 env_0
    # 'env_0', 'env_2', 'env_5','env_6', 'env_8'
    if env not in ["i"]:  # ["env_4", "env_5", "env_6"]:
        if float(env[4:]) > -1 and float(env[4:]) < 50:
            plot_data[env] = {}
            for class_ in all_env_data[env]:
                print(class_)
                if class_ not in [
                    "smcc_MacOpt_gorilla_cov_scale",
                    "smcc_SafeMac_LA10D60gorilla1",
                ]:
                    plot_data[env][class_] = {}
                    (
                        opt_val,
                        traj_list,
                        length,
                        mean,
                        std,
                        sigma_mean,
                        sigma_std,
                        node_list,
                        cov_Rbar_eps,
                        cov_Rbar0,
                        cov_reco_Rbar0,
                    ) = get_regret_of_one_env_one_class_at_each_iter(
                        all_env_data[env][class_]
                    )
                    if length != 0:
                        plot_data[env][class_]["traj_mat"] = traj_list
                        plot_data[env][class_]["traj_env_mean"] = mean
                        plot_data[env][class_]["traj_env_std"] = std
                        plot_data[env][class_]["opt_vec"] = opt_val
                        plot_data[env][class_]["sigma_mean"] = sigma_mean
                        plot_data[env][class_]["sigma_std"] = sigma_std
                        plot_data[env][class_]["nodes"] = torch.Tensor(
                            node_list)
                        plot_data[env][class_]["cov_Rbar_eps"] = cov_Rbar_eps
                        plot_data[env][class_]["cov_Rbar0"] = cov_Rbar0
                        plot_data[env][class_]["cov_reco_Rbar0"] = cov_reco_Rbar0
                    else:
                        stop = 1
                    env_included = env
set_figure_params()
f = plt.figure(figsize=(cm2inches(6.0), cm2inches(4.0)))
ax = f.axes
for class_ in plot_data[env_included]:  # assumed same class in all
    class_mean, class_std_noise = plotter_traj(
        plot_data, class_, "cov_reco_Rbar0")
    label, color = set_label(class_)
    plt.fill_between(
        np.arange(class_mean.shape[0]),
        y1=class_mean - class_std_noise,
        y2=class_mean + class_std_noise,
        alpha=0.3,
        color=color,
    )
    plt.plot(np.arange(class_mean.shape[0]),
             class_mean, label=label, color=color)

# plt.legend()
# leg = plt.legend(loc='lower right', bbox_to_anchor=(
#     1.023, -0.038), prop={"size": 8.5})
plt.tick_params(axis="x", direction="in")
plt.tick_params(axis="y", direction="in")
# plt.xlabel("Samples")
# plt.ylabel(r'$F(X_T; \rho, \bar{R}_{0}(X_0)$')
adapt_figure_size_from_axes(ax)
plt.tight_layout(pad=0)
plt.grid(axis="y")
plt.savefig(filename + ".pdf")
plt.show()


sub_opt = 0
f, ax = plt.subplots()
for class_ in plot_data[env_included]:  # assumed same class in all
    class_mean, class_std_noise = plotter(plot_data, class_)
    label, color = set_label(class_)
    ax.errorbar(
        np.arange(class_mean.shape[0]),
        class_mean,
        yerr=class_std_noise,
        label=label,
        color=color,
    )
    plt.plot(
        torch.cumsum(class_mean, dim=0) / (1 + np.arange(class_mean.shape[0])),
        label=label + "-CR/T",
        color=color,
    )

ax.legend(prop={"size": 16})
ax.set_title("Disk Coverage")
ax.set_xlabel("samples")
ax.set_ylabel("Regret")
ax.xaxis.label.set_fontsize(22)
ax.yaxis.label.set_fontsize(22)
ax.title.set_fontsize(22)
plt.grid(True)
plt.show()

# 3
f = plt.figure(figsize=(cm2inches(6.0), cm2inches(4.0)))
ax = f.axes
for class_ in plot_data[env_included]:  # assumed same class in all
    class_mean, class_std_noise = plotter_traj(plot_data, class_, "cov_Rbar0")
    label, color = set_label(class_)
    plt.fill_between(
        np.arange(class_mean.shape[0]),
        y1=class_mean - class_std_noise,
        y2=class_mean + class_std_noise,
        alpha=0.3,
        color=color,
    )
    plt.plot(np.arange(class_mean.shape[0]),
             class_mean, label=label, color=color)
    # ax.errorbar(np.arange(class_mean.shape[0]),
    #             class_mean, yerr=class_std_noise, label=label, color=color)


plt.legend()
leg = plt.legend(
    loc="lower right",
    ncol=2,
    labelspacing=0.001,
    columnspacing=0.6,
    handleheight=1,
    handletextpad=0.35,
    handlelength=1.0,
    bbox_to_anchor=(1.023, -0.038),
    prop={"size": 8.5},
)
plt.tick_params(axis="x", direction="in")
plt.tick_params(axis="y", direction="in")
plt.xlabel("Samples")
plt.ylabel(r"$F(X_T; \rho, V)$")
# ax.xaxis.label.set_fontsize(22)
# ax.yaxis.label.set_fontsize(22)
# ax.title.set_fontsize(22)
# plt.grid(True)
# plt.show()

adapt_figure_size_from_axes(ax)
plt.tight_layout(pad=0)
plt.grid(axis="y")
plt.savefig(filename + ".pdf")
plt.show()


f, ax = plt.subplots()
for class_ in plot_data[env_included]:  # assumed same class in all
    class_mean, class_std_noise = plotter_traj(
        plot_data, class_, "cov_reco_Rbar0")
    label, color = set_label(class_)
    ax.errorbar(
        np.arange(class_mean.shape[0]),
        class_mean,
        yerr=class_std_noise,
        label=label,
        color=color,
    )

ax.legend(prop={"size": 16})
ax.set_title("Disk Coverage")
ax.set_xlabel("Samples")
ax.set_ylabel("Safe Recommendation")
ax.xaxis.label.set_fontsize(22)
ax.yaxis.label.set_fontsize(22)
ax.title.set_fontsize(22)
plt.grid(True)
plt.show()

f, ax = plt.subplots()
for class_ in plot_data[env_included]:  # assumed same class in all
    class_mean, class_std_noise = plotter_sigma(plot_data, class_)
    label, color = set_label(class_)
    ax.errorbar(
        np.arange(class_mean.shape[0]),
        class_mean,
        yerr=class_std_noise,
        label=label,
        color=color,
    )

ax.legend(prop={"size": 16})
ax.set_title("Disk Coverage")
ax.set_xlabel("iterations")
ax.set_ylabel(r"$w_t$")
ax.xaxis.label.set_fontsize(22)
ax.yaxis.label.set_fontsize(22)
ax.title.set_fontsize(22)
plt.grid(True)
plt.show()

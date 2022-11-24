import pickle

import matplotlib.pyplot as plt
import numpy as np
import torch
from pandas import array

from plotting_utilities.plotting_utilities.utilities import *

CUTOFF_SIZE = 855


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
        n_samples_list = []
        for traj in data_dict:
            if torch.mean(data_dict[traj]["mat_FxIX_rho_opti"]) != torch.nan:
                n_samples = (
                    data_dict[traj]["samples"]["constraint"]
                    + data_dict[traj]["samples"]["density"]
                )
                n_samples_list.append(n_samples)
                # n_samples = 240
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
        mat_FtildexIX_rho_opti = torch.hstack(opt_traj_list).transpose(0, 1)

        mat_of_zero_regret = mat_FtildexIX_rho_opti - mat_FxIX_rho_opti
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
            n_samples_list,
        )
    return 0, 0, 0, 0, 0, 0, 0


def get_mean_var(plot_data, class_, Yobject):
    traj_mat = [plot_data[key][class_][Yobject] for key in plot_data]

    traj_mat = torch.vstack(traj_mat)
    if Yobject != "n_samples_list":
        return (
            torch.mean(traj_mat[:, -1]),
            torch.std(traj_mat[:, -1]) / np.sqrt(traj_mat[:, -1].shape[0]),
            traj_mat[:, 0],
        )
    else:
        return traj_mat


def get_lowest_start_cover(data):
    return torch.mean(torch.vstack(data).min(0)[0])


def get_norm_factor(data):
    """_summary_ Evaluate normaization factor. The factor is maimum number of samples required by any algorithm in that instance of the environment. 

    Args:
        data (list): list of pessi, safemac and 2 stage data. Each data is a tensor of #env X #runs

    Returns:
        min : _description_
        max : 
    """
    concat_data = torch.cat(data, 1)
    return concat_data.min(1)[0], concat_data.max(1)[0]
    a = 1


def cm2inches(cm):
    return cm / 2.54


path = "SafeMaC/pretrained_data/"
data_dir = ["GP", "obstacles", "gorilla"]
data_name = [
    "/constraint-GP.pkl",
    "/constraint-obstacles.pkl",
    "/constraint-gorilla.pkl",
]

bar_dict = {}
# Collection of trajectory termination samples and the final coverage value
for dir, name in zip(data_dir, data_name):
    k = open(path + dir + name, "rb")
    all_env_data = pickle.load(k)
    k.close()

    plot_data = {}
    for env in all_env_data:
        print(env)
        if env not in ["i"]:
            if float(env[4:]) > -1 and float(env[4:]) < 60:
                plot_data[env] = {}
                for class_ in all_env_data[env]:
                    print(class_)
                    if class_ not in ["smcc_2stage2D1"]:
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
                            n_samples_list,
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
                            plot_data[env][class_]["n_samples_list"] = torch.Tensor(
                                n_samples_list
                            )
                        else:
                            stop = 1
                        env_included = env
    bar_dict[dir] = {}
    for class_ in plot_data[env_included]:  # assumed same class in all
        bar_dict[dir][class_] = {}
        if class_ == "smcc_PessiMac2D":
            class_mean, class_std_noise, start_val = get_mean_var(
                plot_data, class_, "cov_reco_Rbar0"
            )
        else:
            class_mean, class_std_noise, start_val = get_mean_var(
                plot_data, class_, "cov_reco_Rbar0"
            )
        bar_dict[dir][class_]["coverage"] = {}
        bar_dict[dir][class_]["coverage"]["mean"] = class_mean
        bar_dict[dir][class_]["coverage"]["var"] = class_std_noise
        bar_dict[dir][class_]["coverage"]["start_val"] = start_val

        all_data = get_mean_var(plot_data, class_, "n_samples_list")
        bar_dict[dir][class_]["samples"] = {}
        bar_dict[dir][class_]["samples"]["data"] = all_data

    # compute normalization factor
    min_vec, max_vec = get_norm_factor(
        [bar_dict[dir][class_]["samples"]["data"] for class_ in bar_dict[dir]]
    )
    base_cover = get_lowest_start_cover(
        [bar_dict[dir][class_]["coverage"]["start_val"]
            for class_ in bar_dict[dir]]
    )

    for class_ in bar_dict[dir]:
        max_minus_vec = (
            bar_dict[dir][class_]["samples"]["data"].transpose(1, 0) - min_vec
        ).transpose(1, 0)
        normed_data = torch.divide(
            max_minus_vec.transpose(0, 1), max_vec - min_vec
        ).transpose(0, 1)

        bar_dict[dir][class_]["samples"]["mean"] = normed_data.mean()
        bar_dict[dir][class_]["samples"]["var"] = normed_data.std() / np.sqrt(
            normed_data.shape[0] * normed_data.shape[1]
        )
        # bar_dict[dir][class_]['coverage']['mean'] -= base_cover

y = {}
yerr = {}
for key in ["samples", "coverage"]:
    for class_ in bar_dict[dir]:
        y[class_] = []
        yerr[class_] = []
        for env_dir in bar_dict:
            yerr[class_].append(bar_dict[env_dir][class_][key]["var"].item())
            y[class_].append(bar_dict[env_dir][class_][key]["mean"].item())

    set_figure_params()
    f = plt.figure(figsize=(cm2inches(3.5), cm2inches(4.0)))
    ax = f.axes
    width = 0.25
    x = np.arange(3)
    plt.bar(
        x - width,
        y["smcc_SafeMac2D"],
        width,
        yerr=yerr["smcc_SafeMac2D"],
        color="Tab:blue",
        align="center",
        alpha=0.7,
        ecolor="black",
        capsize=3,
    )
    plt.bar(
        x,
        y["smcc_2stage2D"],
        width,
        yerr=yerr["smcc_SafeMac2D"],
        color="Tab:orange",
        align="center",
        alpha=0.7,
        ecolor="black",
        capsize=3,
    )
    plt.bar(
        x + width,
        y["smcc_PessiMac2D"],
        width,
        yerr=yerr["smcc_SafeMac2D"],
        color="Tab:green",
        align="center",
        alpha=0.7,
        ecolor="black",
        capsize=3,
    )
    plt.xticks(x, ["GP", "Obst.", "Gorilla"])
    plt.tick_params(axis="x", direction="in")
    plt.tick_params(axis="y", direction="in")
    adapt_figure_size_from_axes(ax)
    if key == "samples":
        plt.ylabel("Samples")
    else:
        plt.ylabel("Coverage")
    plt.tight_layout(pad=0)
    plt.grid(axis="y")
    plt.savefig("bar" + key + ".pdf")

    plt.show()

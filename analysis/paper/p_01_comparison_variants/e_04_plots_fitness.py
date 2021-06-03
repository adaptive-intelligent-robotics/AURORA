import os

import argparse
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import sys
import pandas as pd
import seaborn as sns
from matplotlib import ticker

sys.path.append('/Users/looka/git/sferes2/exp/aurora/submodules/figure_generator')
sys.path.append('/Users/looka/git/sferes2/exp/aurora/')

sys.path.append('/git/sferes2/exp/aurora/submodules/figure_generator')
sys.path.append('/git/sferes2/exp/aurora/')

from analysis.metrics.metrics import Metrics
import analysis.metrics.air_hockey
import analysis.metrics.maze
from data_reader import get_data_proj, get_data_modifier, get_data_offspring


import analysis.paper.plot_utils as pu
import analysis.paper.latent_space_dim_comparison as latent_space_dim_comparison
import analysis.paper.dataframe_preprocessor as dataframe_preprocessor

CB_color_cycle = ['#377eb8', '#ff7f00', '#4daf4a',
                  '#f781bf', '#a65628', '#984ea3',
                  '#999999', '#e41a1c', '#dede00']

def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('-s', '--save')

    args = parser.parse_args()
    return args


def get_data(path_proj_file, xlim, ylim, nb_div, array_gt_positions=None, array_fitness_values=None):
    try:
        if array_gt_positions is None or array_fitness_values is None:
            _, array_gt_positions, array_fitness_values, _ = get_data_proj(path_proj_file)
        array_gt_positions = array_gt_positions[:, 0:2]
        start_lim = np.asarray([xlim[0], ylim[0]])
        stop_lim = np.asarray([xlim[1], ylim[1]])

        temp = np.floor(nb_div * (array_gt_positions - start_lim) / (stop_lim - start_lim))
        bins_array = start_lim + temp * (stop_lim - start_lim) / nb_div
        bins_array = np.round(bins_array, 4)
        # print(start_lim, stop_lim, bins_array)

        unique_bins_pos, count_per_bin_pos = np.unique(bins_array, axis=0, return_counts=True)
        count_per_bin_pos = np.reshape(count_per_bin_pos, unique_bins_pos[:, 1].shape)
        # print(array_fitness_values)

        dict_best_fitness_per_bin = {}
        for bin, fitness in zip(bins_array, array_fitness_values):
            tuple_bin = tuple(bin)
            if tuple_bin not in dict_best_fitness_per_bin:
                dict_best_fitness_per_bin[tuple_bin] = fitness.item()
            else:
                if fitness.item() > dict_best_fitness_per_bin[tuple_bin]:
                    dict_best_fitness_per_bin[tuple_bin] = fitness.item()

        list_best_fitnesses_in_order = []
        for bin in unique_bins_pos:
            tuple_bin = tuple(bin)
            list_best_fitnesses_in_order.append(dict_best_fitness_per_bin[tuple_bin])

        array_best_fitnesses = np.asarray(list_best_fitnesses_in_order)
        array_best_fitnesses = np.reshape(array_best_fitnesses, unique_bins_pos[:, 1].shape)

        list_unique_bins_pos = unique_bins_pos.tolist()
        list_missing_bins = []
        #
        for i in np.round(np.linspace(*xlim, nb_div, endpoint=False), 4):
            for j in np.round(np.linspace(*ylim, nb_div, endpoint=False), 4):
                if [i, j] not in list_unique_bins_pos:
                    list_missing_bins.append([i, j])

        array_missing_bins = np.asarray(list_missing_bins)
        if not list_missing_bins:
            array_missing_bins = array_missing_bins.reshape((-1, 2))
        count_for_missing_bins = np.zeros_like(array_missing_bins[:, 0])
        fitness_for_missing_bins = np.zeros_like(array_missing_bins[:, 0])
        unique_bins_pos = np.vstack((unique_bins_pos, array_missing_bins))
        print(unique_bins_pos)
        # unique_bins_pos[:, 1] = 1 - unique_bins_pos[:, 1]

        best_fitness = np.max(array_best_fitnesses)
        mean_fitness = np.mean(array_best_fitnesses)

        # count_per_bin_pos = np.vstack((count_per_bin_pos.reshape(-1,1), count_for_missing_bins.reshape(-1,1)))
        array_best_fitnesses = np.vstack((array_best_fitnesses.reshape(-1,1), fitness_for_missing_bins.reshape(-1,1)))

        return array_best_fitnesses, unique_bins_pos, mean_fitness, best_fitness
        # print(df)
    except FileNotFoundError:
        pass

def heatmap_hexapod(ax, path_proj_file, nb_div, xlim, ylim):
    try:
        array_best_fitnesses, unique_bins_pos, mean_fitness, best_fitness = get_data(path_proj_file, xlim, ylim, nb_div)

        # print(np.reshape(array_best_fitnesses, unique_bins_pos[:, 1].shape))
        df = pd.DataFrame({
            "y": unique_bins_pos[:, 1],
            "x": unique_bins_pos[:, 0],
            "fitness": np.reshape(array_best_fitnesses, unique_bins_pos[:, 1].shape)
        })
        # print(df)

        df = df.pivot("x", "y", "fitness")
        # df = df.reset_index()
        # print(df)
        # sns.set_palette(sns.color_palette(CB_color_cycle))

        im = sns.heatmap(df,
                    ax=ax,
                     vmin=-np.pi, vmax=0,
                     cmap="viridis",
                     cbar=False,

                         # cbar=True if col == 0 else False,
                         # cbar_ax=None if col > 0 else cbar_ax,
                     mask=(df == 0),
                     xticklabels=[str(ylim[0])] + int(nb_div - 2) * [""] + [str(ylim[1])],
                     yticklabels=[str(xlim[0])] + int(nb_div - 2) * [""] + [str(xlim[1])])

        ax.set_xlim(0, nb_div)
        ax.set_ylim(0, nb_div)
        ax.tick_params(axis='both', which='both', length=0)

        ax.set_xticks(list(range(0, 50+1, 10)))
        ax.set_yticks(list(range(0, 50+1, 10)))

        print(list(np.round(-1.25 + (np.arange(0, 51, 10) / 50) * 2.5, 2)))

        ax.set_xticklabels(
            np.round(ylim[0] + (np.arange(0, 51, 10) / 50) * (ylim[1] - ylim[0]),2)
        )

        ax.set_yticklabels(
            np.round(xlim[0] + (np.arange(0, 51, 10) / 50) * (xlim[1] - xlim[0]), 2)
        )
        return im
    except FileNotFoundError:
        pass


def heatmap_maze(ax, path_proj_file, nb_div, xlim, ylim):
    try:
        array_best_fitnesses, unique_bins_pos, mean_fitness, best_fitness = get_data(path_proj_file, xlim, ylim, nb_div)

        # print(np.reshape(array_best_fitnesses, unique_bins_pos[:, 1].shape))
        df = pd.DataFrame({
            "x": unique_bins_pos[:, 1],
            "y": unique_bins_pos[:, 0],
            "fitness": np.reshape(array_best_fitnesses, unique_bins_pos[:, 1].shape)
        })
        # print(df)

        df = df.pivot("x", "y", "fitness")
        # df = df.reset_index()
        # print(df)
        # sns.set_palette(sns.color_palette(CB_color_cycle))
        # ax.grid()
        im = sns.heatmap(df,
                    ax=ax,
                    # vmax=-1, vmin=0,
                    cmap="viridis",
                    cbar=False,
                    # cbar=True if col == 0 else False,
                    # cbar_ax=None if col > 0 else cbar_ax,
                    mask=(df == 0),
                    xticklabels=[str(xlim[0])] + int(nb_div - 2) * [""] + [str(xlim[1])],
                    yticklabels=[str(ylim[0])] + int(nb_div - 2) * [""] + [str(ylim[1])],
                     vmax=0)
        ax.set_xlim(0, nb_div)
        ax.set_ylim(0, nb_div)
        ax.tick_params(axis='both', which='both', length=0)

        ax.set_xticks(list(range(0, 50+1, 10)))
        ax.set_yticks(list(range(0, 50+1, 10)))

        print(list(np.round(-1.25 + (np.arange(0, 51, 10) / 50) * 2.5, 2)))

        ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%d'))
        ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%d'))

        ax.set_xticklabels(
            np.round(xlim[0] + (np.arange(0, 51, 10) / 50) * (xlim[1] - xlim[0]), 0).astype(int)
        )

        ax.set_yticklabels(
            np.round(ylim[0] + (np.arange(0, 51, 10) / 50) * (ylim[1] - ylim[0]), 0).astype(int)
        )



        return im
    except FileNotFoundError:
        pass


def heatmap_air_hockey(ax, path_proj_file, nb_div, xlim, ylim, array_gt_positions=None, array_fitness_values=None, vmax=0, vmin=None, linewidths=0., linecolor="white"):
    try:
        array_best_fitnesses, unique_bins_pos, mean_fitness, best_fitness = get_data(path_proj_file, xlim, ylim, nb_div,
                                                                                     array_gt_positions=array_gt_positions,
                                                                                     array_fitness_values=array_fitness_values,)

        # print(np.reshape(array_best_fitnesses, unique_bins_pos[:, 1].shape))
        df = pd.DataFrame({
            "x": unique_bins_pos[:, 1],
            "y": unique_bins_pos[:, 0],
            "fitness": np.reshape(array_best_fitnesses, unique_bins_pos[:, 1].shape)
        })
        # print(df)

        df = df.pivot("x", "y", "fitness")
        # df = df.reset_index()
        # print(df)
        # sns.set_palette(sns.color_palette(CB_color_cycle))
        # ax.grid()
        im = sns.heatmap(df,
                         ax=ax,
                         # vmax=-1, vmin=0,
                         cmap="viridis",
                         cbar=False,
                         # cbar=True if col == 0 else False,
                         # cbar_ax=None if col > 0 else cbar_ax,
                         mask=(df == 0),
                         xticklabels=[str(xlim[0])] + int(nb_div - 2) * [""] + [str(xlim[1])],
                         yticklabels=[str(ylim[0])] + int(nb_div - 2) * [""] + [str(ylim[1])],
                         vmax=vmax,
                         linewidths=linewidths,
                         linecolor=linecolor,
                         vmin=vmin,
                         )

        ax.set_xlim(0, nb_div)
        ax.set_ylim(0, nb_div)
        ax.tick_params(axis='both', which='both', length=0)

        ax.set_xticks(list(range(0, nb_div+1, 10)))
        ax.set_yticks(list(range(0, nb_div+1, 10)))

        print(list(np.round(-1.25 + (np.arange(0, 51, 10) / 50) * 2.5, 2)))

        ax.set_xticklabels(
            np.round(xlim[0] + (np.arange(0, nb_div+1, 10) / nb_div) * (xlim[1] - xlim[0]),2)
        )

        ax.set_yticklabels(
            np.round(ylim[0] + (np.arange(0, nb_div+1, 10) / nb_div) * (ylim[1] - ylim[0]), 2)
        )

        return im
    except FileNotFoundError:
        pass



def get_last_gen_proj_file(path_folder_proj_file):
    for gen in np.arange(30000, 0, -1):
        path_supposed_proj_file = os.path.join(path_folder_proj_file, f"proj_{gen}.dat")
        if os.path.exists(path_supposed_proj_file):
            return path_supposed_proj_file

    raise FileNotFoundError


def generate_figure(path_save):
    plt.clf()
    plt.cla()
    plt.close()

    matplotlib.rc_file_defaults()

    pu.figure_setup()

    fig_size = pu.get_fig_size(21, 18)
    # fig = plt.figure(figsize=fig_size)
    fig, axes = plt.subplots(3, 3, constrained_layout=True)
    fig.set_size_inches(*fig_size)
    # cbar_ax = fig.add_axes([.91, .3, .03, .4])

    current_folder_path = "/Users/looka/git/sferes2/exp/aurora/analysis/paper/p_01_comparison_variants/"

    # ax = fig.add_subplot(131)
    # ax2 = fig.add_subplot(132)
    # ax3 = fig.add_subplot(133)


    # ax.plot(x, y, c='b', lw=pu.plot_lw())


    list_experiments = [
        "maze",
        "hexapod_camera_vertical",
        "air_hockey",
    ]

    dict_experiment_to_title = {
        "maze": "Maze",
        "hexapod_camera_vertical": "Hexapod",
        "air_hockey": "Air-Hockey",
    }

    dict_experiment_to_folder = {
        "maze": "data_archives_fit/maze/",
        "hexapod_camera_vertical": "data_archives_fit/hexapod/",
        "air_hockey": "data_archives_fit/air_hockey/",
    }

    # name_folder -> title
    list_algorithms_folders = [
        "aurora_10_fit",
        "qd_fit",
        "aurora_10",
        # "taxons_10",
        # "ns",
        # "qd_no_sel",
    ]

    dict_algorithm_folder_to_title = {
        "aurora_10_fit": "aurora_10_uniform_psat (fit)",
        "qd_fit": "qd_uniform_psat (fit)",
        "aurora_10": "aurora_10_uniform_psat",
    }

    for key in dict_algorithm_folder_to_title:
        dict_algorithm_folder_to_title[key] = dict_algorithm_folder_to_title[key].replace("_", "\_")

    for row in range(len(list_experiments)):
        for col in range(len(list_algorithms_folders)):

            experiment = list_experiments[row]
            algorithm_folder = list_algorithms_folders[col]
            folder_proj_file = os.path.join(current_folder_path, dict_experiment_to_folder[experiment], algorithm_folder)

            path_proj_file = get_last_gen_proj_file(folder_proj_file)
            print(path_proj_file)

            if experiment == "maze":
                im = heatmap_maze(axes[row, col], path_proj_file, 50, xlim=(0, 600), ylim=(0, 600))
                axes[row, col].set_xlabel("$x_T$")
                axes[row, col].set_ylabel("$y_T$")
                # plot_comparison_maze(axes[row, col], path_proj_file)
            elif experiment == "hexapod_camera_vertical":
                im = heatmap_hexapod(axes[row, col], path_proj_file, 50, xlim=(-1.25, 1.25), ylim=(-1, 1))
                axes[row, col].set_xlabel("$x_T$")
                axes[row, col].set_ylabel("$y_T$")
            elif experiment == "air_hockey":
                im = heatmap_air_hockey(axes[row, col], path_proj_file, 50, xlim=(-1., 1.), ylim=(-1., 1.))
                axes[row, col].set_xlabel("$x_T^{\\text{puck}}$")
                axes[row, col].set_ylabel("$y_T^{\\text{puck}}$")
            print(im.get_images())
            # elif experiment == "air_hockey":
            #     plot_comparison_air_hockey(axes[row, col], path_proj_file)

            # axes[row, col].get_legend().remove()

            axes[row, col].tick_params(axis='both', which='both', length=1)



            # ax.set_xlabel('$x$')
            # axes[row, col].set_xlabel(dict_algorithm_folder_to_title[algorithm_folder])
            # axes[row, col].set_ylabel(dict_experiment_to_title[experiment])



            # axes[row, col].xaxis.label.set_visible(False)

            # axes[row, col].set_axisbelow(True)

            if row == 0:
                axes[row, col].set_title(dict_algorithm_folder_to_title[algorithm_folder])

            axes[row, col].spines["top"].set_visible(False)
            axes[row, col].spines["bottom"].set_visible(True)
            axes[row, col].spines["right"].set_visible(False)
            axes[row, col].spines["left"].set_visible(True)

            if col > 0:
                axes[row, col].yaxis.label.set_visible(False)
                axes[row, col].tick_params(axis='y',          # changes apply to the x-axis
                                           which='both',      # both major and minor ticks are affected
                                           right=False,      # ticks along the bottom edge are off
                                           left=True,         # ticks along the top edge are off
                                           labelleft=False)

    pad = 5  # in points
    for ax, row in zip(axes[:,0], range(len(list_experiments))):
        ax.annotate(dict_experiment_to_title[list_experiments[row]], xy=(0, 0.5), xytext=(-ax.yaxis.labelpad - pad, 0),
                    xycoords=ax.yaxis.label, textcoords='offset points',
                    size='medium', ha='right', va='center', rotation=90)

    # ---------------

    # ax.set_xlabel('$x$')
    # plt.grid()
    # plt.tight_layout()

    for axis in axes.flat:
        print(axis.collections)
    plt.colorbar(mappable=axes[0,0].collections[0], ax=axes[0,:])
    plt.colorbar(mappable=axes[1,0].collections[0], ax=axes[1,:])
    plt.colorbar(mappable=axes[2,0].collections[0], ax=axes[2,:])

    if path_save:
        pu.save_fig(fig, path_save)
    else:
        plt.show()

    plt.clf()
    plt.cla()
    plt.close()


def main():
    args = get_args()
    generate_figure(path_save=args.save)


if __name__ == '__main__':
    main()

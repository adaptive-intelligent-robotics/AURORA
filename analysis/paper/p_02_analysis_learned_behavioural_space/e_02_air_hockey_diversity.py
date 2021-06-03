import os
import sys

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import ListedColormap

sys.path.append('/Users/looka/git/sferes2/exp/aurora/submodules/figure_generator')
sys.path.append('/Users/looka/git/sferes2/exp/aurora/')

sys.path.append('/git/sferes2/exp/aurora/submodules/figure_generator')
sys.path.append('/git/sferes2/exp/aurora/')

import analysis.paper.plot_utils as pu

from analysis.paper.p_01_comparison_variants.e_02_scatter_plots_variants import get_last_gen_proj_file, get_args
from analysis.paper import latent_space_dim_comparison


from data_reader import convert_to_rgb, get_data_proj, read_data
from airl_2D_colormap import AIRL_ColorMap


def read_observation_file(path_observation_file):
    OBS = "obs"
    dict_data_per_component = read_data(path_observation_file, ["_1",
                                                                OBS
                                                                ])
    return dict_data_per_component[OBS]


def get_data_percentages(array_observations, xlim, ylim, nb_div):
    try:
        print(array_observations)
        array_gt_positions = array_observations[:, -2:]
        array_gt_positions = array_gt_positions[:, 0:2]
        start_lim = np.asarray([xlim[0], ylim[0]])
        stop_lim = np.asarray([xlim[1], ylim[1]])

        temp = np.floor(nb_div * (array_gt_positions - start_lim) / (stop_lim - start_lim))
        bins_array = start_lim + temp * (stop_lim - start_lim) / nb_div
        bins_array = np.round(bins_array, 4)
        # print(start_lim, stop_lim, bins_array)

        # print(array_fitness_values)

        # Indexes of individuals ending in each bin.
        dict_indexes_per_bin = {}
        for index, bin in enumerate(bins_array):
            tuple_bin = tuple(bin)
            if tuple_bin not in dict_indexes_per_bin:
                dict_indexes_per_bin[tuple_bin] = [index]
            else:
                dict_indexes_per_bin[tuple_bin].append(index)

        # For each bin getting the percentage of bins through which all trajectories go
        dict_percentage_bins_trajectories_ending_in_bin = dict()
        for bin in dict_indexes_per_bin:
            list_indexes_bin = dict_indexes_per_bin[bin]
            flattened_all_obs_for_bin = array_observations[np.array(list_indexes_bin), :].flatten()
            all_pos_for_bin = flattened_all_obs_for_bin.reshape(-1, 2)
            _temp = np.floor(nb_div * (all_pos_for_bin - start_lim) / (stop_lim - start_lim))
            all_bins_ending_in_bin = start_lim + _temp * (stop_lim - start_lim) / nb_div
            all_bins_ending_in_bin = np.round(all_bins_ending_in_bin, 4)
            unique_bins_pos, _ = np.unique(all_bins_ending_in_bin, axis=0, return_counts=True)
            total_number_bins_trajectories_ending_in_bin = np.size(unique_bins_pos, axis=0)
            dict_percentage_bins_trajectories_ending_in_bin[bin] = total_number_bins_trajectories_ending_in_bin / (nb_div * nb_div)

        #
        list_percentage_bins_trajectories_ending_in_bin = []
        unique_bins_array = np.unique(bins_array, axis=0)
        for bin in unique_bins_array:
            tuple_bin = tuple(bin)
            list_percentage_bins_trajectories_ending_in_bin.append(dict_percentage_bins_trajectories_ending_in_bin[tuple_bin])

        array_percentage_bins = np.asarray(list_percentage_bins_trajectories_ending_in_bin)
        array_percentage_bins = np.reshape(array_percentage_bins, unique_bins_array[:, 1].shape)

        # Getting list missing bins
        list_unique_bins_pos = unique_bins_array.tolist()
        list_missing_bins = []

        for i in np.round(np.linspace(*xlim, nb_div, endpoint=False), 4):
            for j in np.round(np.linspace(*ylim, nb_div, endpoint=False), 4):
                if [i, j] not in list_unique_bins_pos:
                    list_missing_bins.append([i, j])

        array_missing_bins = np.asarray(list_missing_bins)
        if not list_missing_bins:
            array_missing_bins = array_missing_bins.reshape((-1, 2))
        percentage_for_missing_bins = np.zeros_like(array_missing_bins[:, 0])
        unique_bins_array = np.vstack((unique_bins_array, array_missing_bins))
        # bins_array[:, 1] = 1 - bins_array[:, 1]

        # count_per_bin_pos = np.vstack((count_per_bin_pos.reshape(-1,1), count_for_missing_bins.reshape(-1,1)))
        array_percentage_bins = np.vstack((array_percentage_bins.reshape(-1,1),
                                          percentage_for_missing_bins.reshape(-1,1)))

        return array_percentage_bins, unique_bins_array
        # print(df)
    except FileNotFoundError:
        pass


def heatmap_air_hockey(ax, path_obs_file, nb_div, xlim, ylim, mask=None, alpha=1., linewidths=0.15, linecolor="white", array_observations=None):
    try:
        if array_observations is None:
            array_observations = read_observation_file(path_obs_file)
        array_percentages, unique_bins_pos = get_data_percentages(array_observations, xlim, ylim, nb_div)
        # print(array_percentages)

        # print(np.reshape(array_best_fitnesses, unique_bins_pos[:, 1].shape))
        df = pd.DataFrame({
            "x": unique_bins_pos[:, 1],
            "y": unique_bins_pos[:, 0],
            "percentage": np.reshape(array_percentages, unique_bins_pos[:, 1].shape)
        })
        print("mean diversity", array_percentages.mean())
        # print(df)


        df = df.pivot("x", "y", "percentage")
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
                         mask=mask,
                         xticklabels=[str(xlim[0])] + int(nb_div - 2) * [""] + [str(xlim[1])],
                         yticklabels=[str(ylim[0])] + int(nb_div - 2) * [""] + [str(ylim[1])],
                         vmin=0, vmax=1, alpha=alpha,
                         linewidths=linewidths,
                         linecolor=linecolor,)
        ax.set_xlim(0, nb_div)
        ax.set_ylim(0, nb_div)
        ax.tick_params(axis='both', which='both', length=0)

        ax.set_xticks(list(range(0, nb_div+1, nb_div // 5)))
        ax.set_yticks(list(range(0, nb_div+1, nb_div // 5)))

        print(list(np.round(-1.25 + (np.arange(0, nb_div + 1, nb_div // 5) / nb_div) * 2.5, 2)))

        ax.set_xticklabels(
            np.round(xlim[0] + (np.arange(0, nb_div + 1, nb_div // 5) / nb_div) * (xlim[1] - xlim[0]),2)
        )

        ax.set_yticklabels(
            np.round(ylim[0] + (np.arange(0, nb_div + 1, nb_div // 5) / nb_div) * (ylim[1] - ylim[0]), 2)
        )

        return im
    except FileNotFoundError:
        pass

def _multiply_resolution_array(Y, multiplication_factor):
    N = len(Y)
    X = np.arange(0, multiplication_factor * N, multiplication_factor)
    X_new = np.arange(multiplication_factor * N - 1)
    Y_new = np.interp(X_new, X, Y)
    return Y_new


def generate_example(ax,
                    list_experiments,
                    dict_experiment_to_folder,
                    dict_algorithm_folder_to_title,
                    current_folder_path,
                     ax_text=None,
                     ):
    experiment = list_experiments[0]
    algorithm_folder = "example"

    path_proj_file = os.path.join(current_folder_path,
                                  dict_experiment_to_folder[experiment],
                                  algorithm_folder,
                                  "observation_gen_0001000.dat")
    array_observations = read_observation_file(path_proj_file)
    # array_gt_positions = array_observations[:, -2:]
    # array_gt_positions = array_gt_positions[:, 0:2]
    array_observations = array_observations[-0.6 <= array_observations[:, -2]]
    array_observations = array_observations[array_observations[:, -2] <= -0.4]
    array_observations = array_observations[-0.2 <= array_observations[:, -1]]
    array_observations = array_observations[array_observations[:, -1] <= 0.]

    array_observations = 10 * ((array_observations + 1.0) / 2.0)

    array_observations_diff_x = array_observations[:, :-2:2] - array_observations[:, 2::2]
    indexes_first_non_zero_delta = (array_observations_diff_x != 0).argmax(axis=1) * 2


    delta_x = array_observations[np.arange(len(array_observations)), indexes_first_non_zero_delta + 2] \
              - array_observations[np.arange(len(array_observations)), indexes_first_non_zero_delta]

    delta_y = array_observations[np.arange(len(array_observations)), indexes_first_non_zero_delta + 1 + 2] \
              - array_observations[np.arange(len(array_observations)), indexes_first_non_zero_delta + 1]


    angles = np.arctan2(delta_y, delta_x)
    array_normalised_angles = (angles - np.min(angles)) / (np.max(angles) - np.min(angles))


    np.random.seed(102)
    # indexes_trajectories_to_select = [0, 78, 66, 73, 72]
    indexes_trajectories_to_select = [0, 66]
    trajectories = None

    for index_trajectory in indexes_trajectories_to_select:
        print(index_trajectory)

        trajectory = array_observations[index_trajectory, :]

        if trajectories is None:
            trajectories = trajectory.reshape(1, -1)
        else:
            trajectories = np.vstack((trajectories, trajectory.reshape(1, -1)))


        trajectory_x = trajectory[::2].reshape(-1, 1)
        trajectory_y = trajectory[1::2].reshape(-1, 1)




        trajectory_x = _multiply_resolution_array(trajectory_x.flatten(), multiplication_factor=100).reshape(-1, 1)
        trajectory_y = _multiply_resolution_array(trajectory_y.flatten(), multiplication_factor=100).reshape(-1, 1)

        array_points = np.hstack((trajectory_x, trajectory_y))


        array_points_original_coord = (array_points / 5) - 1
        print(array_points_original_coord, array_points_original_coord.shape, "SHAPE")

        xlim=(-1., 1.)
        ylim=(-1., 1.)
        nb_div=10
        array_percentages, unique_bins_pos = get_data_percentages(array_points_original_coord, nb_div=nb_div, xlim=xlim, ylim=ylim)



        # print(np.reshape(array_best_fitnesses, unique_bins_pos[:, 1].shape))
        df = pd.DataFrame({
            "x": unique_bins_pos[:, 1],
            "y": unique_bins_pos[:, 0],
            "percentage": np.reshape(array_percentages, unique_bins_pos[:, 1].shape)
        })
        print("mean diversity", array_percentages.mean())
        # print(df)


        df = df.pivot("x", "y", "percentage")
        # df = df.reset_index()
        # print(df)
        # sns.set_palette(sns.color_palette(CB_color_cycle))
        # ax.grid()

        cmap_example = matplotlib.colors.ListedColormap(['gainsboro'])

        print(df.isnull(), df.isna())
        mask = np.ones(shape=(nb_div, nb_div))
        mask[nb_div - 1, 2] = 0
        im = sns.heatmap(df,
                         ax=ax,
                         # vmax=-1, vmin=0,
                         cmap=cmap_example,
                         cbar=False,
                         # cbar=True if col == 0 else False,
                         # cbar_ax=None if col > 0 else cbar_ax,
                         mask=df == 0,
                         xticklabels=[str(xlim[0])] + int(nb_div - 2) * [""] + [str(xlim[1])],
                         yticklabels=[str(ylim[0])] + int(nb_div - 2) * [""] + [str(ylim[1])],
                         alpha=1.,
                         # linewidths=linewidths,
                         linecolor="gray")
        ax.set_xlim(0, nb_div)
        ax.set_ylim(0, nb_div)
        ax.tick_params(axis='both', which='both', length=0)

        ax.set_xticks(list(range(0, nb_div+1, nb_div // 5)))
        ax.set_yticks(list(range(0, nb_div+1, nb_div // 5)))

        print(list(np.round(-1.25 + (np.arange(0, nb_div + 1, nb_div // 5) / nb_div) * 2.5, 2)))

        ax.set_xticklabels(
            np.round(xlim[0] + (np.arange(0, nb_div + 1, nb_div // 5) / nb_div) * (xlim[1] - xlim[0]),2)
        )

        ax.set_yticklabels(
            np.round(ylim[0] + (np.arange(0, nb_div + 1, nb_div // 5) / nb_div) * (ylim[1] - ylim[0]), 2)
        )

        angles = get_initial_angle_trajectories(trajectory.reshape(1,-1))

        array_normalised_angles = (angles + np.pi) / (2 * np.pi)

        rgb_colors = AIRL_ColorMap.get_color_function(array_normalised_angles * np.pi * 2 + 0.9)
        rgb_colors = rgb_colors * 0.7 + 0.3
        print(rgb_colors)
        ax.plot(trajectory[::2].flatten(),
                trajectory[1::2].flatten(),
                linewidth=1,
                # marker='o',
                # s=1,
                color=rgb_colors.flatten(),
                alpha=1)

    if experiment == "air_hockey":
        nb_div = 10
        mask = np.ones(shape=(nb_div, nb_div))
        mask[4, 2] = 0


        viridis = cm.get_cmap('viridis')


        cmap_example_divisity_cell = matplotlib.colors.ListedColormap(viridis(0.36))

        linewidths = 0.15
        im = sns.heatmap(df,
                         ax=ax,
                         # vmax=-1, vmin=0,
                         cmap=cmap_example_divisity_cell,
                         cbar=False,
                         # cbar=True if col == 0 else False,
                         # cbar_ax=None if col > 0 else cbar_ax,
                         mask=mask,
                         xticklabels=[str(xlim[0])] + int(nb_div - 2) * [""] + [str(xlim[1])],
                         yticklabels=[str(ylim[0])] + int(nb_div - 2) * [""] + [str(ylim[1])],
                         alpha=1.,
                         linewidths=linewidths,
                         linecolor="gray")

        ax.set_xlim(0, nb_div)
        ax.set_ylim(0, nb_div)
        ax.tick_params(axis='both', which='both', length=0)

        ax.set_xticks(list(range(0, nb_div+1, nb_div // 5)))
        ax.set_yticks(list(range(0, nb_div+1, nb_div // 5)))

        print(list(np.round(-1.25 + (np.arange(0, nb_div + 1, nb_div // 5) / nb_div) * 2.5, 2)))

        ax.set_xticklabels(
            np.round(xlim[0] + (np.arange(0, nb_div + 1, nb_div // 5) / nb_div) * (xlim[1] - xlim[0]),2)
        )

        ax.set_yticklabels(
            np.round(ylim[0] + (np.arange(0, nb_div + 1, nb_div // 5) / nb_div) * (ylim[1] - ylim[0]), 2)
        )

        ax.set_xlabel("$x$")
        ax.set_ylabel("$y$")

    ax.tick_params(axis='both', which='both', length=1)

    ax.spines["top"].set_visible(False)
    ax.spines["bottom"].set_visible(True)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(True)

    ax.yaxis.label.set_visible(False)
    ax.xaxis.label.set_visible(False)
    ax.tick_params(axis='y',          # changes apply to the x-axis
                   which='both',      # both major and minor ticks are affected
                   right=False,      # ticks along the bottom edge are off
                   left=True,         # ticks along the top edge are off
                   labelleft=False)
    ax.tick_params(axis='x',          # changes apply to the x-axis
                   which='both',      # both major and minor ticks are affected
                   top=False,      # ticks along the bottom edge are off
                   bottom=True,         # ticks along the top edge are off
                   labelbottom=False)

    ax.annotate("Diversity score$\:=\:\\frac{36}{100}$", xy=(2.5, 5.1), xytext=(0.3, 1.05), xycoords="data", arrowprops=dict(arrowstyle="->", connectionstyle="angle,angleA=180,angleB=90,rad=5"), textcoords=ax.transAxes)

def get_initial_angle_trajectories(array_observations):
    array_observations_diff_x = array_observations[:, :-2:2] - array_observations[:, 2::2]
    indexes_first_non_zero_delta = (array_observations_diff_x != 0).argmax(axis=1) * 2


    delta_x = array_observations[np.arange(len(array_observations)), indexes_first_non_zero_delta + 2] \
              - array_observations[np.arange(len(array_observations)), indexes_first_non_zero_delta]

    delta_y = array_observations[np.arange(len(array_observations)), indexes_first_non_zero_delta + 1 + 2] \
              - array_observations[np.arange(len(array_observations)), indexes_first_non_zero_delta + 1]


    angles = np.arctan2(delta_y, delta_x)
    return angles

def generate_trajectories_one_square(axes,
                                     list_algorithms_folders,
                                     list_experiments,
                                     dict_experiment_to_folder,
                                     dict_algorithm_folder_to_title,
                                     current_folder_path,
                                     ):
    row = 1

    for col in range(len(list_algorithms_folders)):



        experiment = list_experiments[0]
        algorithm_folder = list_algorithms_folders[col]



        path_proj_file = os.path.join(current_folder_path,
                                      dict_experiment_to_folder[experiment],
                                      algorithm_folder,
                                      "observation_gen_0001000.dat")


        print(path_proj_file)
        print(experiment)
        col_axis = col + 1

        mini = 0
        maxi = 10

        array_observations = read_observation_file(path_proj_file)
        # array_gt_positions = array_observations[:, -2:]
        # array_gt_positions = array_gt_positions[:, 0:2]
        print(array_observations[:, -2] <= -0.4)
        print(array_observations[-0.6 <= array_observations[:, -2]])
        array_observations = array_observations[-0.6 <= array_observations[:, -2]]
        array_observations = array_observations[array_observations[:, -2] <= -0.4]
        array_observations = array_observations[0.8 <= array_observations[:,-1]]
        array_observations = array_observations[array_observations[:, -1] <= 1.]


        array_observations = 10 * ((array_observations + 1.0) / 2.0)

        angles = get_initial_angle_trajectories(array_observations)

        array_normalised_angles = (angles + np.pi) / (2 * np.pi)

        rgb_colors = AIRL_ColorMap.get_color_function(array_normalised_angles * np.pi * 2 + 0.9)
        rgb_colors = rgb_colors * 0.7 + 0.3


        for trajectory, color in zip(array_observations, rgb_colors):
            axes[row, col_axis].plot(trajectory[::2].flatten(),
                                     trajectory[1::2].flatten(),
                                     linewidth=1,
                                     # marker='o',
                                     # s=1,
                                     c=color,
                                     alpha=0.5,
                                     zorder=-1)


        if experiment == "air_hockey":
            nb_div = 10
            mask = np.ones(shape=(nb_div, nb_div))
            mask[nb_div - 1, 2] = 0
            im = heatmap_air_hockey(axes[row, col_axis], path_proj_file, nb_div=nb_div, xlim=(-1., 1.), ylim=(-1., 1.), mask=mask, alpha=0.75, linecolor="gray")
            axes[row, col_axis].set_xlabel("$x$")
            axes[row, col_axis].set_ylabel("$y$")
        print(im.get_images())
        # elif experiment == "air_hockey":
        #     plot_comparison_air_hockey(axes[row, col], path_proj_file)

        # axes[row, col].get_legend().remove()

        axes[row, col_axis].tick_params(axis='both', which='both', length=1)



        # ax.set_xlabel('$x$')
        # axes[row, col].set_xlabel(dict_algorithm_folder_to_title[algorithm_folder])
        # axes[row, col].set_ylabel(dict_experiment_to_title[experiment])



        # axes[row, col].xaxis.label.set_visible(False)

        # axes[row, col].set_axisbelow(True)


        axes[row, col_axis].spines["top"].set_visible(False)
        axes[row, col_axis].spines["bottom"].set_visible(True)
        axes[row, col_axis].spines["right"].set_visible(False)
        axes[row, col_axis].spines["left"].set_visible(True)




        if col_axis > 1:
            axes[row, col_axis].yaxis.label.set_visible(False)
            axes[row, col_axis].tick_params(axis='y',          # changes apply to the x-axis
                                            which='both',      # both major and minor ticks are affected
                                            right=False,      # ticks along the bottom edge are off
                                            left=True,         # ticks along the top edge are off
                                            labelleft=False)


def add_annotation(ax, text, x_offset=-0.1, y_offset=1.1):
    ax.text(x_offset, y_offset, f"\\textbf{{{text}}}", transform=ax.transAxes,
            size=12, weight='bold')


def text_annotations(axes):
    add_annotation(axes[0, 0], text="Illu", y_offset=1.15)
    add_annotation(axes[1, 0], text="Prog")
    add_annotation(axes[0, 1], text="AU1", y_offset=1.15)
    add_annotation(axes[1, 1], text="AU2")
    add_annotation(axes[0, 2], text="HC1", y_offset=1.15)
    add_annotation(axes[1, 2], text="HC2")


def generate_ax_comparison(ax, ):
    df = pd.read_csv("/Users/looka/git/sferes2/exp/aurora/analysis/paper/dataframes/df.csv")
    order_legend = [
        "aurora_uniform_10_psat",
        "qd_uniform_psat",
    ]

    dict_order_replacement = {
        "aurora_uniform_10_psat": "AURORA-CSC-uniform-10",
        "qd_uniform_psat": "HC-CSC-uniform",
    }

    df = df[df["name_variant"].isin(order_legend)]

    for current_name_variant, updated_name_variant in dict_order_replacement.items():
        df["name_variant"] = df["name_variant"].replace(current_name_variant, updated_name_variant, regex=True)

    # add escape character for latex rendering
    for i in range(len(order_legend)):
        order_legend[i] = dict_order_replacement[order_legend[i]]
        order_legend[i] = order_legend[i].replace('_', '\_')

    df["name_variant"] = df["name_variant"].replace('_', '\_', regex=True)

    y_lim = (0., 1.)
    environment = "air_hockey"

    # Color palette chosen
    my_cmap = ListedColormap(sns.color_palette("colorblind").as_hex())
    list_colors = [
        my_cmap.colors[index_color] for index_color in [4, 0]
    ]
    list_colors = list_colors[::-1]

    medians = latent_space_dim_comparison.LatentSpaceDimComparison.get_comparison_chosen_variants_custom_scripts(
        folder_path_save=None,
        df=df,
        environment=environment,
        y="diversity",
        y_lim=y_lim,
        name_file=None,
        ax=ax,
        hue_order=order_legend,
        list_colors=list_colors,
        alpha_median=1.,
        linewidth_median=2.,
    )
    ax.grid()
    ax.set_xlabel('Iteration')
    ax.set_title("Mean Diversity score (\%)")

    print(medians["AURORA-CSC-uniform-10"].iloc[-1])
    ax.annotate("$\\bigstar$", xy=(1000, medians["AURORA-CSC-uniform-10"].iloc[-1]), xycoords="data", horizontalalignment="center", verticalalignment="center")
    ax.annotate("$\\blacksquare$", xy=(1000, medians["HC-CSC-uniform"].iloc[-1]), xycoords="data", horizontalalignment="center", verticalalignment="center")

    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, bbox_to_anchor=(1.02, -0.02), ncol=1, loc="lower right", prop={'size': 9})


def generate_figure(path_save):
    plt.clf()
    plt.cla()
    plt.close()

    matplotlib.rc_file_defaults()

    pu.figure_setup()
    pu.set_font_size(new_font_size=12)

    fig_size = pu.get_fig_size(22, 12)
    # fig = plt.figure(figsize=fig_size)
    fig, axes = plt.subplots(2, 3, constrained_layout=True)
    fig.set_size_inches(*fig_size)
    # cbar_ax = fig.add_axes([.91, .3, .03, .4])

    current_folder_path = "/Users/looka/git/sferes2/exp/aurora/analysis/paper/p_02_analysis_learned_behavioural_space/"

    # ax = fig.add_subplot(131)
    # ax2 = fig.add_subplot(132)
    # ax3 = fig.add_subplot(133)


    # ax.plot(x, y, c='b', lw=pu.plot_lw())


    list_experiments = [
        # "maze",
        # "hexapod_camera_vertical",
        "air_hockey",
    ]

    dict_experiment_to_title = {
        # "maze": "Maze",
        # "hexapod_camera_vertical": "Hexapod",
        "air_hockey": "Air-Hockey",
    }

    dict_experiment_to_folder = {
        # "maze": "data_archives_fit/maze/",
        # "hexapod_camera_vertical": "data_archives_fit/hexapod/",
        "air_hockey": "data_obs_air_hockey/",
    }

    # name_folder -> title
    list_algorithms_folders = [
        "aurora_10",
        "qd",
        # "taxons_10",
        # "ns",
        # "qd_no_sel",
    ]

    dict_algorithm_folder_to_title = {
        "aurora_10": "AURORA-CSC-uniform-10 $\\bigstar$",
        "qd": "HC-CSC-uniform $\\blacksquare$",
    }

    for key in dict_algorithm_folder_to_title:
        dict_algorithm_folder_to_title[key] = dict_algorithm_folder_to_title[key].replace("_", "\_")

    row = 0

    for col in range(len(list_algorithms_folders)):

        experiment = list_experiments[0]
        algorithm_folder = list_algorithms_folders[col]

        path_proj_file = os.path.join(current_folder_path,
                                      dict_experiment_to_folder[experiment],
                                      algorithm_folder,
                                      "observation_gen_0001000.dat")
        print(path_proj_file)
        print(experiment)
        col_axis = col + 1
        if experiment == "air_hockey":
            im = heatmap_air_hockey(axes[row, col_axis], path_proj_file, nb_div=10, xlim=(-1., 1.), ylim=(-1., 1.))
            axes[row, col_axis].set_xlabel("$x$")
            axes[row, col_axis].set_ylabel("$y$")
        print(im.get_images())
        # elif experiment == "air_hockey":
        #     plot_comparison_air_hockey(axes[row, col], path_proj_file)

        # axes[row, col].get_legend().remove()

        axes[row, col_axis].tick_params(axis='both', which='both', length=1)



        # ax.set_xlabel('$x$')
        # axes[row, col].set_xlabel(dict_algorithm_folder_to_title[algorithm_folder])
        # axes[row, col].set_ylabel(dict_experiment_to_title[experiment])



        # axes[row, col].xaxis.label.set_visible(False)

        # axes[row, col].set_axisbelow(True)

        axes[row, col_axis].set_title(dict_algorithm_folder_to_title[algorithm_folder])

        axes[row, col_axis].spines["top"].set_visible(False)
        axes[row, col_axis].spines["bottom"].set_visible(True)
        axes[row, col_axis].spines["right"].set_visible(False)
        axes[row, col_axis].spines["left"].set_visible(True)

        if row == 0 and col_axis > 0:
            axes[row, col_axis].xaxis.label.set_visible(False)
            axes[row, col_axis].tick_params(axis='x',          # changes apply to the x-axis
                                            which='both',      # both major and minor ticks are affected
                                            top=False,      # ticks along the bottom edge are off
                                            bottom=True,         # ticks along the top edge are off
                                            labelbottom=False)

        if col_axis > 1:
            axes[row, col_axis].yaxis.label.set_visible(False)
            axes[row, col_axis].tick_params(axis='y',          # changes apply to the x-axis
                                       which='both',      # both major and minor ticks are affected
                                       right=False,      # ticks along the bottom edge are off
                                       left=True,         # ticks along the top edge are off
                                       labelleft=False)

    generate_trajectories_one_square(
        axes,
        list_algorithms_folders,
        list_experiments,
        dict_experiment_to_folder,
        dict_algorithm_folder_to_title,
        current_folder_path
    )

    generate_example(axes[0, 0],
                     list_experiments,
                     dict_experiment_to_folder,
                     dict_algorithm_folder_to_title,
                     current_folder_path,
                     )

    generate_ax_comparison(axes[1, 0])

    text_annotations(axes)

    # pad = 5  # in points
    # for ax, col in zip(axes[:,0], range(len(list_algorithms_folders))):
    #     ax.annotate(dict_algorithm_folder_to_title[list_algorithms_folders[col]], xy=(0, 0.5), xytext=(-ax.yaxis.labelpad - pad, 0),
    #                 xycoords=ax.yaxis.label, textcoords='offset points',
    #                 size='medium', ha='right', va='center', rotation=90)

    # ---------------

    # ax.set_xlabel('$x$')
    # plt.grid()
    # plt.tight_layout()

    for axis in axes.flat:
        print(axis.collections)

    cbar = plt.colorbar(mappable=axes[0,1].collections[0], ax=axes[0, 1:3])
    cbar.ax.tick_params(axis='both',          # changes apply to the x-axis
                        which='both',      # both major and minor ticks are affected
                        width=0.5, length=3,
                        direction="in")
    cbar.ax.set_title(r"Diversity (\%)")

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

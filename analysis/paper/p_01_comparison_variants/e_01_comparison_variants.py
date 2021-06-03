import copy
import os

import argparse
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import sys
import pandas as pd
import seaborn as sns
from matplotlib import gridspec
from matplotlib.colors import ListedColormap

sys.path.append('/Users/looka/git/sferes2/exp/aurora/submodules/figure_generator')
sys.path.append('/Users/looka/git/sferes2/exp/aurora/')

sys.path.append('/git/sferes2/exp/aurora/submodules/figure_generator')
sys.path.append('/git/sferes2/exp/aurora/')

from analysis.metrics.metrics import Metrics
import analysis.metrics.air_hockey
import analysis.metrics.maze


import analysis.paper.plot_utils as pu
import analysis.paper.latent_space_dim_comparison as latent_space_dim_comparison
import analysis.paper.dataframe_preprocessor as dataframe_preprocessor
import analysis.paper.p_00_diagram.e_03_illustration_grid_mean_fitness as e_03_illustration_grid_mean_fitness


import singularity.experiment
import singularity.collections_experiments.maze as maze_experiments
import singularity.collections_experiments.air_hockey as air_hockey_experiments
import singularity.collections_experiments.hexapod_camera_vertical as hexapod_camera_vertical_experiments


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('-s', '--save')

    args = parser.parse_args()
    return args


def plot_comparison(ax, environment, metric, df, order_legend, linestyles, list_colors, list_markers, markevery, dict_legends):
    # metric = analysis.metrics.air_hockey.MetricsAirHockey.COVERAGE_POS + "_50"

    folder_path_save_taxons = os.path.join(os.path.dirname(os.path.abspath(__file__)), "comparison_aurora")
    if not os.path.exists(folder_path_save_taxons):
        os.mkdir(folder_path_save_taxons)

    print(environment, metric)
    if metric in [metric,
                  analysis.metrics.air_hockey.MetricsAirHockey.JDS_UNIFORM_EXPLORED]:
        y_lim = None
    elif metric == analysis.metrics.air_hockey.MetricsAirHockey.SIZE_POP:
        y_lim = (1e2, 1e4)
    else:
        y_lim = None

    if environment == "maze":
        df = df[df["gen"] <= 10000]
    elif environment == "hexapod_camera_vertical":
        df = df[df["gen"] <= 15000]

    latent_space_dim_comparison.LatentSpaceDimComparison.get_comparison_chosen_variants_custom_scripts(
        folder_path_save=folder_path_save_taxons,
        df=df,
        environment=environment,
        y=metric,
        y_lim=y_lim,
        name_file=f"latent_space_dim_comparison_{environment}_{metric}.png",
        ax=ax,
        hue_order=order_legend,
        # rasterize=True,
        linestyles=linestyles,
        list_colors=list_colors,
        list_markers=list_markers,
        markevery=markevery,
        dict_legends=dict_legends,
    )



def generate_figure(path_save):
    plt.clf()
    plt.cla()
    plt.close()

    matplotlib.rc_file_defaults()

    plt.subplots(constrained_layout=True)


    pu.figure_setup()

    fig_size = pu.get_fig_size(26, 10)
    # fig = plt.figure(figsize=fig_size)
    fig, axes = plt.subplots(2, 4, constrained_layout=True)

    # gs0 = gridspec.GridSpec(2, 3, figure=fig,)
    # gs0.update(wspace=0.25, hspace=0.1)


    # axes = []
    # for i in range(2):
    #     tmp_list_axes = []
    #     for j in range(3):
    #         if i == 1 and j == 3:
    #             continue
    #         tmp_list_axes.append(fig.add_subplot(gs0[i, j]))
    #     axes.append(tmp_list_axes)

    fig.set_size_inches(*fig_size)



    # ax = fig.add_subplot(131)
    # ax2 = fig.add_subplot(132)
    # ax3 = fig.add_subplot(133)


    # ax.plot(x, y, c='b', lw=pu.plot_lw())

    metrics = [
        analysis.metrics.air_hockey.MetricsAirHockey.COVERAGE_POS + "_40",
        analysis.metrics.air_hockey.MetricsAirHockey.MEAN_FITNESS,
    ]

    dict_metrics_to_y_label = {
        "size_pop": "Container Size",
        analysis.metrics.air_hockey.MetricsAirHockey.COVERAGE_POS + "_40": "Coverage (\%)",
        analysis.metrics.air_hockey.MetricsAirHockey.MEAN_FITNESS: "Grid Mean Fitness"
    }

    experiment = [
        "maze",
        "hexapod_camera_vertical",
        "air_hockey",
    ]

    dict_experiment_to_title = {
        "maze": "Maze",
        "hexapod_camera_vertical": "Hexapod",
        "air_hockey": "Air-Hockey",
    }

    df = pd.read_csv("/Users/looka/git/sferes2/exp/aurora/analysis/paper/dataframes/df.csv")

    for exp in [
        maze_experiments.MAZE_AURORA_SURPRISE_10_COLORS,
        air_hockey_experiments.AIR_HOCKEY_AURORA_10_COLORS_SURPRISE,
        hexapod_camera_vertical_experiments.CAMERA_VERTICAL_AURORA_10_COLORS_SURPRISE,
    ]:
        df.loc[df[analysis.metrics.air_hockey.MetricsAirHockey.MAIN_FOLDER] == exp.get_results_folder_name(), "name_variant"] = "aurora_surprise_10_psat"
        # exp_without_fit = copy.deepcopy(exp)
        # exp_without_fit._has_fit = False
        # print("bouh", exp_without_fit.get_results_folder_name())
        # df.loc[df[analysis.metrics.air_hockey.MetricsAirHockey.MAIN_FOLDER] == exp_without_fit.get_results_folder_name(), "name_variant"] = "aurora_surprise_10_psat"


    df["name_variant"] = df["name_variant"].replace('_', '\_', regex=True)

    order_legend = [
        "aurora_uniform_10_psat",
        # "aurora_curiosity_10_psat",
        "aurora_novelty_10_psat",
        "aurora_surprise_10_psat",
        "TAXONS_10",

        # "aurora_nov_sur_10_psat",

        "qd_uniform_psat",

        "qd_no_selection_psat",
        "NS",
    ]


    dict_legends = {
        "qd\_no\_selection\_psat": "Random Search",
        "aurora\_uniform\_10\_psat": "AURORA-CSC-uniform-10",
        "aurora\_novelty\_10\_psat": "AURORA-CSC-novelty-10",
        "aurora\_surprise\_10\_psat": "AURORA-CSC-surprise-10",
        "qd\_uniform\_psat": "HC-CSC-uniform",
        "TAXONS\_10": "TAXONS-10",
    }

    order_legend = order_legend[::-1]

    separation = 4

    linestyles = ["-"] * 4 + ["--"] * (len(order_legend) - separation)

    linestyles = linestyles[::-1]
    my_cmap = ListedColormap(sns.color_palette("colorblind").as_hex())
    list_colors = [
        my_cmap.colors[index_color] for index_color in [0,1,2,3,4,7,5]
    ]
    list_colors = list_colors[::-1]

    list_markers = ["+"] * 3 + [None] * (len(order_legend) - 3)
    list_markers = list_markers[::-1]

    for i in range(len(order_legend)):
        order_legend[i] = order_legend[i].replace('_', '\_')

    df = df[df["name_variant"].isin(order_legend)]

    indexes_legend = [0, 1, 2, 3, 4, 5, 6]

    def get_sub_list(l, indexes):
        return [l[len(l) -1 - index] for index in indexes][::-1]
    list_markers = get_sub_list(list_markers, indexes_legend)
    order_legend = get_sub_list(order_legend, indexes_legend)
    list_colors = get_sub_list(list_colors, indexes_legend)
    linestyles = get_sub_list(linestyles, indexes_legend)

    for row in range(len(metrics)):
        for col in range(len(experiment)):
            if experiment[col] == "air_hockey":
                markevery = 2
            elif experiment[col] == "maze":
                markevery = 4
            elif experiment[col] == "hexapod_camera_vertical":
                markevery = 10
            plot_comparison(axes[row][col], experiment[col], metrics[row], df, order_legend, linestyles, list_colors, list_markers, markevery, dict_legends)

            # axes[row][col].get_legend().remove()

            axes[row][col].grid()

            # ax.set_xlabel('$x$')
            axes[row][col].set_ylabel(dict_metrics_to_y_label[metrics[row]])
            if col > 0:
                axes[row][col].yaxis.label.set_visible(False)
            if row == 0:
                axes[row][col].xaxis.label.set_visible(False)
                axes[row][col].tick_params(axis='x',
                                           which='both',
                                           bottom=True,
                                           top=False,
                                           labelbottom=False)
            axes[row][col].set_xlabel('Iteration')

            axes[row][col].set_axisbelow(True)

            if row == 0:
                axes[row][col].set_title(dict_experiment_to_title[experiment[col]])


            axes[row][col].plot([0], marker='None', linestyle='None', label='dummy-tophead')
            axes[row][col].plot([0], marker='None', linestyle='None', label='dummy-empty')
    # ---------------

    # ax.set_xlabel('$x$')

    handles, labels = axes[1][2].get_legend_handles_labels()

    handles = handles[::-1]
    labels = labels[::-1]

    p_empty, p_tophead = handles[0], handles[1]

    handles = handles[2:]
    labels = labels[2:]

    legend_1 = axes[0, 3].legend([p_tophead] + handles[:separation], [r"\textit{Unsupervised BDs}"] + labels[:separation], bbox_to_anchor=(0., 1.15), ncol=1, loc="upper left", prop={'size': 8})
    axes[0, 3].legend([p_tophead] + handles[separation:], [r"\textit{Hand-coded BDs}"] + labels[separation:], bbox_to_anchor=(0., 0.0), ncol=1, loc="lower left", prop={'size': 8})
    axes[0, 3].add_artist(legend_1)
    axes[0, 3].axis("off")

    axes[1, 3].axis("off")

    axes[0,0].text(-0.1, 1.05, '\\textbf{Prog}', ha='right', transform=axes[0,0].transAxes, fontsize=11)


    # illu_gridspec = gridspec.GridSpec(2, 2, wspace=0.4, hspace=0.3)
    # sub_axes = illu_gridspec.subplots()
    #
    # e_03_illustration_grid_mean_fitness.generage_figure_axes(fig, sub_axes, font_size=7)
    #
    # gs0.tight_layout(fig, rect=[0,0,0.75,1])
    # ill

    # plt.grid()
    # plt.tight_layout()

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

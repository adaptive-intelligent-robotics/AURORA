import os

import argparse
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import sys
import pandas as pd

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

import singularity.collections_experiments.air_hockey as exp_air_hockey


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('-s', '--save')

    args = parser.parse_args()
    return args


def plot_comparison(ax, metric):
    # metric = analysis.metrics.air_hockey.MetricsAirHockey.COVERAGE_POS + "_50"

    list_experiments = [
        *exp_air_hockey.DICT_AIR_HOCKEY_AURORA_UNIFORM_n_COLORS_UPDATE_CONTAINER_PERIOD.values()
    ]

    order_legend = [
        exp.get_results_folder_name()
        for exp in list_experiments
    ]

    dict_order_replacement = {
        exp.get_results_folder_name(): f"period {exp.update_container_period}" for exp in list_experiments
    }

    df = pd.read_csv("/Users/looka/git/sferes2/exp/aurora/analysis/paper/df_air_hockey.csv")


    df = df[df["main_folder"].isin(order_legend)]

    for current_name_variant, updated_name_variant in dict_order_replacement.items():
        df["main_folder"] = df["main_folder"].replace(current_name_variant, updated_name_variant, regex=True)

    # add escape character for latex rendering
    for i in range(len(order_legend)):
        order_legend[i] = dict_order_replacement[order_legend[i]]
        order_legend[i] = order_legend[i].replace('_', '\_')

    df["main_folder"] = df["main_folder"].replace('_', '\_', regex=True)

    folder_path_save_taxons = os.path.join(os.path.dirname(os.path.abspath(__file__)), "comparison_aurora")
    if not os.path.exists(folder_path_save_taxons):
        os.mkdir(folder_path_save_taxons)

    y_lim = (4000, 8500)

    latent_space_dim_comparison.LatentSpaceDimComparison.get_comparison_chosen_variants_custom_scripts(
        folder_path_save=folder_path_save_taxons,
        df=df,
        environment="air_hockey",
        y=metric,
        y_lim=y_lim,
        name_file=f"tesat.png",
        ax=ax,
        hue_order=order_legend,
        specified_hue="main_folder",
        no_y_log_scale=True,
    )


def generate_figure(path_save):
    plt.clf()
    plt.cla()
    plt.close()

    matplotlib.rc_file_defaults()

    pu.figure_setup()

    fig_size = pu.get_fig_size(16, 7)
    # fig = plt.figure(figsize=fig_size)
    fig, axes = plt.subplots(1, 1, constrained_layout=True)
    fig.set_size_inches(*fig_size)

    # ax = fig.add_subplot(131)
    # ax2 = fig.add_subplot(132)
    # ax3 = fig.add_subplot(133)


    # ax.plot(x, y, c='b', lw=pu.plot_lw())

    metrics = [
        analysis.metrics.maze.MetricsMaze.SIZE_POP,
    ]

    dict_metrics_to_y_label = {
        analysis.metrics.maze.MetricsMaze.SIZE_POP: "Container Size",
    }

    experiment = [
        "air_hockey",
    ]

    dict_experiment_to_title = {
        "air_hockey": "Air-Hockey",
    }

    row = 0

    for col in range(len(experiment)):
        plot_comparison(axes, metrics[row])

        axes.get_legend().remove()

        axes.grid()

        # ax.set_xlabel('$x$')
        axes.set_ylabel(dict_metrics_to_y_label[metrics[row]])
        if col > 0:
            axes.yaxis.label.set_visible(False)
        axes.set_xlabel('Iteration')

        axes.set_axisbelow(True)

        axes.set_title(dict_experiment_to_title[experiment[col]])
    # ---------------

    # ax.set_xlabel('$x$')

    handles, labels = axes.get_legend_handles_labels()
    fig.legend(handles, labels, bbox_to_anchor=(0.5, -0.02), ncol=2, loc="upper center")
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

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


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('-s', '--save')

    args = parser.parse_args()
    return args


def plot_comparison(ax, environment, metric, df, order_legend):
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

    latent_space_dim_comparison.LatentSpaceDimComparison.get_comparison_chosen_variants_custom_scripts(
        folder_path_save=folder_path_save_taxons,
        df=df,
        environment=environment,
        y=metric,
        y_lim=y_lim,
        name_file=f"latent_space_dim_comparison_{environment}_{metric}.png",
        ax=ax,
        hue_order=order_legend,
    )

def generate_figure(path_save):
    plt.clf()
    plt.cla()
    plt.close()

    matplotlib.rc_file_defaults()

    pu.figure_setup()

    fig_size = pu.get_fig_size(24, 7)
    # fig = plt.figure(figsize=fig_size)
    fig, axes = plt.subplots(1, 3, constrained_layout=True)
    fig.set_size_inches(*fig_size)

    # ax = fig.add_subplot(131)
    # ax2 = fig.add_subplot(132)
    # ax3 = fig.add_subplot(133)


    # ax.plot(x, y, c='b', lw=pu.plot_lw())

    metrics = [
        analysis.metrics.maze.MetricsMaze.MEAN_FITNESS,
    ]

    dict_metrics_to_y_label = {
        analysis.metrics.maze.MetricsMaze.MEAN_FITNESS: "Average Fitness",
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

    order_legend = [
        "aurora_uniform_10_psat_fit",
        "qd_uniform_psat_fit",
        "aurora_uniform_10_psat",
    ]

    dict_order_replacement = {
        "aurora_uniform_10_psat_fit": "aurora_uniform_10_psat (fit)",
        "qd_uniform_psat_fit": "qd_uniform_psat (fit)",
        "aurora_uniform_10_psat": "aurora_uniform_10_psat",
    }

    row = 0
    df = pd.read_csv("/Users/looka/git/sferes2/exp/aurora/analysis/paper/dataframes/df.csv")


    df = df[df["name_variant"].isin(order_legend)]

    for current_name_variant, updated_name_variant in dict_order_replacement.items():
        df["name_variant"] = df["name_variant"].replace(current_name_variant, updated_name_variant, regex=True)

    # add escape character for latex rendering
    for i in range(len(order_legend)):
        order_legend[i] = dict_order_replacement[order_legend[i]]
        order_legend[i] = order_legend[i].replace('_', '\_')

    df["name_variant"] = df["name_variant"].replace('_', '\_', regex=True)

    for col in range(len(experiment)):
        plot_comparison(axes[col], experiment[col], metrics[row], df, order_legend)

        axes[col].get_legend().remove()

        axes[col].grid()

        # ax.set_xlabel('$x$')
        axes[col].set_ylabel(dict_metrics_to_y_label[metrics[row]])
        if col > 0:
            axes[col].yaxis.label.set_visible(False)
        axes[col].set_xlabel('Iteration')

        axes[col].set_axisbelow(True)

        axes[col].set_title(dict_experiment_to_title[experiment[col]])
    # ---------------

    # ax.set_xlabel('$x$')

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, bbox_to_anchor=(0.5, -0.02), ncol=3, loc="upper center")
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

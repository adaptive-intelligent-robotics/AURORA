import os

import argparse
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import sys
import pandas as pd
import seaborn as sns

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

def sig_exp(num_str):
    parts = num_str.split('.', 2)
    decimal = parts[1] if len(parts) > 1 else ''
    exp = -len(decimal)
    digits = parts[0].lstrip('0') + decimal
    trimmed = digits.rstrip('0')
    exp += len(digits) - len(trimmed)
    sig = int(trimmed) if trimmed else 0
    return sig, exp

def plot_comparison(ax, environment, metric):
    # metric = analysis.metrics.air_hockey.MetricsAirHockey.COVERAGE_POS + "_50"

    list_experiments = [
        *exp_air_hockey.DICT_AIR_HOCKEY_AURORA_UNIFORM_n_COLORS_COEFFICIENT_PROPORTIONAL_CONTROL_L.values()
    ]

    order_legend = [
        exp.get_results_folder_name()
        for exp in list_experiments
    ]

    list_num_exp = [sig_exp(f"{exp.coefficient_proportional_control_l:.10f}")
                    for exp in list_experiments
                    ]

    dict_order_replacement = {
        exp.get_results_folder_name(): fr"${num} \times 10^{{{exponent}}}$" for exp, (num, exponent) in zip(list_experiments, list_num_exp)
    }

    df = pd.read_csv("/Users/looka/git/sferes2/exp/aurora/analysis/paper/df_air_hockey.csv")


    MAIN_FOLDER = analysis.metrics.maze.MetricsMaze.MAIN_FOLDER
    df = df[df[MAIN_FOLDER].isin(order_legend)]

    for current_name_variant, updated_name_variant in dict_order_replacement.items():
        df[MAIN_FOLDER] = df[MAIN_FOLDER].replace(current_name_variant, updated_name_variant)

    # add escape character for latex rendering
    for i in range(len(order_legend)):
        order_legend[i] = dict_order_replacement[order_legend[i]]
        order_legend[i] = order_legend[i].replace('_', '\_')

    df[MAIN_FOLDER] = df[MAIN_FOLDER].replace('_', '\_', regex=True)

    folder_path_save_taxons = os.path.join(os.path.dirname(os.path.abspath(__file__)), "comparison_aurora")
    if not os.path.exists(folder_path_save_taxons):
        os.mkdir(folder_path_save_taxons)

    print(environment, metric)

    if metric in [metric,
                  analysis.metrics.air_hockey.MetricsAirHockey.JDS_UNIFORM_EXPLORED]:
        y_lim = None
    elif metric == analysis.metrics.air_hockey.MetricsAirHockey.SIZE_POP:
        # y_lim = (1e2, 1e4)
        y_lim = None
    else:
        y_lim = None


    sns.set_palette(sns.color_palette("colorblind", as_cmap=True))

    latent_space_dim_comparison.LatentSpaceDimComparison.get_comparison_chosen_variants_custom_scripts(
        folder_path_save=folder_path_save_taxons,
        df=df,
        environment=environment,
        y=metric,
        y_lim=y_lim,
        name_file=f"latent_space_dim_comparison_{environment}_{metric}.png",
        ax=ax,
        hue_order=order_legend,
        specified_hue=MAIN_FOLDER,
        no_y_log_scale=True,
    )


def generate_figure(path_save):
    plt.clf()
    plt.cla()
    plt.close()

    matplotlib.rc_file_defaults()

    pu.figure_setup()

    fig_size = pu.get_fig_size(10, 7)
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
        analysis.metrics.maze.MetricsMaze.SIZE_POP: "Container Size"
    }

    experiment = [
        "air_hockey",
    ]

    dict_experiment_to_title = {
        "air_hockey": "Air-Hockey",
    }

    row = 0
    col = 0

    plot_comparison(axes, experiment[0], metrics[row])

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
    fig.legend(handles, labels, bbox_to_anchor=(1.05, 0.98), ncol=1, loc="upper left", title="$K_p$")
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

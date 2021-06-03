import os

import argparse
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import sys
import pandas as pd
from matplotlib import ticker
import seaborn as sns
import scipy.stats as stats
from statsmodels.sandbox.stats.multicomp import MultiComparison

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

import analysis.paper.p_values as p_values

GEN = "gen"
SIZE_POP = analysis.metrics.maze.MetricsMaze.SIZE_POP
MAIN_FOLDER = analysis.metrics.maze.MetricsMaze.MAIN_FOLDER
SUBDIR = analysis.metrics.maze.MetricsMaze.SUBDIR
SIZE_POP_POST = analysis.metrics.maze.MetricsMaze.SIZE_POP + "_post"
INCREASE = "increase"

def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('-s', '--save')

    args = parser.parse_args()
    return args

def generate_p_values_log_files_custom_latent_size(df,
                                gen,

                                environment,
                                list_metrics,
                                path_file="p-values.log",
                                comparison_on_metric="name_variant",
                                ):
    df = df[df['environment'] == environment]

    df = df[~df[comparison_on_metric].isnull()]


    if not df.empty:

        with open(path_file, 'w') as f:

            for metric in list_metrics:
                print(df[metric], list(df[comparison_on_metric]))
                MultiComp = MultiComparison(df[metric], df[comparison_on_metric])
                MultiComp.decimal_tvalues = 10
                comp = MultiComp.allpairtest(stats.ranksums, method='Holm',)
                print(f"------ {metric} -------")
                print(comp[0])
                print(f"=======================")

                f.write(f"------ {metric} -------\n")
                f.write(f"{comp[0]}\n")

                for (data, adjusted_p_value) in zip(comp[2], comp[1][2]):
                    str_info = f"{data[0]} - {data[1]} ---> {adjusted_p_value}"
                    print(str_info)
                    f.write(f"{str_info}\n")

                f.write(f"=======================\n\n")

def plot_population_loss(df, ax, environment):
    print(f"plot population loss - {environment}")

    df, list_legends = latent_space_dim_comparison.filter_and_add_supplementary_legend_comparison_latent_dim(df, do_get_list_legends=True)

    print(df.columns)
    # Adding relevant info to dataframe
    new_df = pd.DataFrame({
        column: []
        for column in df.columns
    })
    new_df = new_df.rename(columns={SIZE_POP: SIZE_POP_POST})
    for t in df[MAIN_FOLDER].drop_duplicates():
        sub_df = df[df[MAIN_FOLDER] == t]
        sub_df.loc[:, GEN] -= 1
        sub_df = sub_df.rename(columns={SIZE_POP: SIZE_POP_POST})
        print(sub_df)
        new_df = new_df.append(sub_df)
    print(new_df)

    print(df[GEN], new_df[GEN])
    new_df = new_df[[GEN, MAIN_FOLDER, SUBDIR, SIZE_POP_POST, "name_variant"]]

    merged_df = pd.merge(df, new_df, how="inner", on=[GEN, MAIN_FOLDER, SUBDIR, "name_variant"])
    print(merged_df)
    merged_df[INCREASE] = merged_df[SIZE_POP_POST] - merged_df[SIZE_POP]
    merged_df_with_decreasing = merged_df[merged_df[INCREASE] < 0]

    merged_df_with_decreasing.loc[:, INCREASE] *= -1

    # print(merged_df[[INCREASE, VALUE]])
    print("merged_df_with_decreasing", merged_df_with_decreasing.columns)
    # merged_df_with_decreasing.sum(INCREASE)
    df_sum_loss_indiv = merged_df_with_decreasing.groupby([MAIN_FOLDER,
                                                           SUBDIR,
                                                           dataframe_preprocessor.ENVIRONMENT,
                                                           dataframe_preprocessor.LATENT_SPACE_DIM,
                                                           analysis.metrics.maze.MetricsMaze.LEGEND,
                                                           "name_variant"],
                                                          )[INCREASE].mean().to_frame()
    df_sum_loss_indiv = df_sum_loss_indiv.reset_index()


    for i in range(len(list_legends)):
        list_legends[i] = list_legends[i].replace('_', '\_')

    df_sum_loss_indiv["legend"] = df_sum_loss_indiv["legend"].replace('_', '\_', regex=True)

    # Useless code for retro-compatibility
    folder_path_save = os.path.join(os.path.dirname(os.path.abspath(__file__)), "latent_space_dim_comparison")
    if not os.path.exists(folder_path_save):
        os.mkdir(folder_path_save)

    # if environment == "maze":
    #     gen = 10000
    # elif environment == "hexapod_camera_vertical":
    #     gen = 15000
    # elif environment == "air_hockey":
    #     gen = 1000

    generate_p_values_log_files_custom_latent_size(
        df=df_sum_loss_indiv,
        gen=None,
        environment=environment,
        list_metrics=[INCREASE],
        path_file=f"influence_latent_dim-{environment}-{INCREASE}.log",
        # comparison_on_metric="legend",
    )

    latent_space_dim_comparison.LatentSpaceDimComparison.get_comparison_latent_space_metric_custom_script(
        folder_path_save=folder_path_save,
        df=df_sum_loss_indiv,
        environment=environment,
        y=INCREASE,
        y_lim=None,
        chosen_generation=None,
        name_file=f"latent_space_dim_comparison_{environment}_{INCREASE}.png",
        ax=ax,
        hue_order=list_legends,
    )
    ax.set_yscale("log")

    # ax.set_ylim(10, 10000)


def plot_comparison(df, ax, environment, metric):
    print(f"Plot Comparison - {environment} - {metric}")
    # metric = analysis.metrics.air_hockey.MetricsAirHockey.COVERAGE_POS + "_40"

    list_environments = [
        "hexapod_camera_vertical",
        "maze",
        "air_hockey"
    ]

    dict_environment_chosen_generation = {
        "hexapod_camera_vertical": 15000,
        "maze": 10000,
        "air_hockey": 1000,
    }

    df, list_legends = latent_space_dim_comparison.filter_and_add_supplementary_legend_comparison_latent_dim(df, do_get_list_legends=True)

    # Filter only relevant values for df
    condition = None
    for environment_item, gen_item in dict_environment_chosen_generation.items():
        if condition is None:
            condition = (df[analysis.metrics.air_hockey.MetricsAirHockey.GEN] == gen_item) & (df[dataframe_preprocessor.ENVIRONMENT] == environment_item)
        else:
            condition = condition | ((df[analysis.metrics.air_hockey.MetricsAirHockey.GEN] == gen_item)
                                     & (df[dataframe_preprocessor.ENVIRONMENT] == environment_item))

    for i in range(len(list_legends)):
        list_legends[i] = list_legends[i].replace('_', '\_')

    df["legend"] = df["legend"].replace('_', '\_', regex=True)

    folder_path_save = os.path.join(os.path.dirname(os.path.abspath(__file__)), "latent_space_dim_comparison")
    if not os.path.exists(folder_path_save):
        os.mkdir(folder_path_save)

    y_coverage = analysis.metrics.air_hockey.MetricsAirHockey.COVERAGE_POS + "_40"

    print(environment, metric)
    if metric in [y_coverage,
             analysis.metrics.air_hockey.MetricsAirHockey.JDS_UNIFORM_EXPLORED]:
        if environment == "hexapod_camera_vertical":
            print("GLOUDAKJH", environment)
            y_lim = None
        else:
            y_lim = (0., 1.)
    elif metric == analysis.metrics.air_hockey.MetricsAirHockey.SIZE_POP:
        y_lim = (0., 15000)
    else:
        y_lim = None

    if y_lim is None:
        # recompute the ax.dataLim
        ax.relim()
        # update ax.viewLim using the new dataLim
        ax.autoscale_view()

    if environment == "maze":
        gen = 10000
    elif environment == "hexapod_camera_vertical":
        gen = 15000
    elif environment == "air_hockey":
        gen = 1000

    p_values.generate_p_values_log_files(
        df=df,
        gen=gen,
        environment=environment,
        list_metrics=[metric],
        path_file=f"influence_latent_dim-{environment}-{gen}-{metric}.log"
    )


    latent_space_dim_comparison.LatentSpaceDimComparison.get_comparison_latent_space_metric_custom_script(
        folder_path_save=folder_path_save,
        df=df,
        environment=environment,
        y=metric,
        y_lim=y_lim,
        chosen_generation=dict_environment_chosen_generation[environment],
        name_file=f"latent_space_dim_comparison_{environment}_{metric}.png",
        ax=ax,
        hue_order=list_legends,
    )
    # ax.set_yscale("log")


def generate_figure(path_save: str):
    plt.clf()
    plt.cla()
    plt.close()

    matplotlib.rc_file_defaults()

    pu.figure_setup()
    font_size_params = {
        'font.size': 12,
        'axes.titlesize': 12,
        'legend.fontsize': 12,
    }
    plt.rcParams.update(font_size_params)

    fig_size = pu.get_fig_size(20, 10)

    # fig = plt.figure(figsize=fig_size)
    fig, axes = plt.subplots(3, 3, constrained_layout=True)
    fig.set_size_inches(*fig_size)


    # ax = fig.add_subplot(131)
    # ax2 = fig.add_subplot(132)
    # ax3 = fig.add_subplot(133)


    # ax.plot(x, y, c='b', lw=pu.plot_lw())

    metrics = [
        "size_pop",
        analysis.metrics.air_hockey.MetricsAirHockey.COVERAGE_POS + "_40",
        "increase"
    ]

    dict_metrics_to_y_label = {
        "size_pop": "Container Size",
        analysis.metrics.air_hockey.MetricsAirHockey.COVERAGE_POS + "_40": "Coverage (\%)",
        "increase": "Average Container\nLoss per Update",
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

    print("load dataframe")
    df = pd.read_csv("/Users/looka/git/sferes2/exp/aurora/analysis/paper/dataframes/df.csv")
    df_l = pd.read_csv("/Users/looka/git/sferes2/exp/aurora/analysis/paper/dataframes/df_l.csv")

    df = df[~df["gen"].isnull()]
    df["gen"] = df["gen"].astype(int)

    print("finished loading dataframe")


    for row in range(len(metrics)):
        for col in range(len(experiment)):
            if 0 <= row <= 1:
                plot_comparison(df, axes[row, col], experiment[col], metrics[row])
            elif row == 2:
                if experiment[col] == "hexapod_camera_vertical":
                    plot_population_loss(df_l, axes[row, col], experiment[col])
                else:
                    plot_population_loss(df, axes[row, col], experiment[col])

            axes[row, col].get_legend().remove()

            axes[row, col].grid()

            # ax.set_xlabel('$x$')
            axes[row, col].set_ylabel(dict_metrics_to_y_label[metrics[row]])
            if col > 0:
                axes[row, col].yaxis.label.set_visible(False)
            if 0 <= row <= len(metrics) - 2:
                axes[row, col].xaxis.label.set_visible(False)
            axes[row, col].set_xlabel('BD Space Dimensionality $n$')
            axes[row, col].xaxis.set_ticks([2, 5, 8, 10, 15, 20])
            axes[row, col].xaxis.set_major_formatter(ticker.FormatStrFormatter('%d'))
            axes[row, col].set_axisbelow(True)

            if row == 0:
                axes[row, col].set_title(dict_experiment_to_title[experiment[col]])

            if row < 2:
                axes[row, col].tick_params(axis='x',          # changes apply to the x-axis
                                           which='both',      # both major and minor ticks are affected
                                           top=False,
                                           bottom=True,
                                           labelbottom=False)

            if row == 0 and 1 <= col <= 2:
                axes[row, col].tick_params(axis='y',
                                           which='both',
                                           right=False,
                                           left=True,
                                           labelleft=False)


    # ---------------

    # ax.set_xlabel('$x$')

    handles, labels = axes[2, 2].get_legend_handles_labels()
    fig.legend(handles, labels, bbox_to_anchor=(0.5, -0.02), ncol=2, loc="upper center")
    # plt.grid()
    # plt.tight_layout()

    print(plt.rcParams.keys(), sns.axes_style())


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

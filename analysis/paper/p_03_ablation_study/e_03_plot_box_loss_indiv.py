# TODO: For both period and alpha?

import os

import argparse
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import sys
import pandas as pd

from paper.p_01_comparison_variants.e_01_comparison_variants import get_args

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

import seaborn as sns


GEN = "gen"
VALUE = analysis.metrics.maze.MetricsMaze.SIZE_POP
T = analysis.metrics.maze.MetricsMaze.MAIN_FOLDER
SUB_T = analysis.metrics.maze.MetricsMaze.SUBDIR
VALUE_POST = analysis.metrics.maze.MetricsMaze.SIZE_POP + "_post"
TOTAL_POPULATION_LOSS = "Total\nPopulation Loss"


def generate_figure(path_save):
    plt.clf()
    plt.cla()
    plt.close()

    matplotlib.rc_file_defaults()

    pu.figure_setup()

    fig_size = pu.get_fig_size(16, 7)
    # fig = plt.figure(figsize=fig_size)
    fig, axes = plt.subplots(1, 2, constrained_layout=True)
    fig.set_size_inches(*fig_size)

    # --------------------------------
    df = pd.read_csv("/Users/looka/git/sferes2/exp/aurora/analysis/paper/df_air_hockey.csv")

    list_experiments = [
        exp_air_hockey.DICT_AIR_HOCKEY_AURORA_UNIFORM_n_COLORS_UPDATE_CONTAINER_PERIOD[update_container_period]
        for update_container_period in exp_air_hockey.LIST_UPDATE_CONTAINER_PERIOD
    ]

    order_legend = [
        exp.get_results_folder_name()
        for exp in list_experiments
    ]


    print("\n".join(list(df["main_folder"].drop_duplicates())))
    df = df[df["main_folder"].isin(order_legend)]
    print(df)

    dict_order_replacement = {
        exp.get_results_folder_name(): f"{exp.update_container_period}" for exp in list_experiments
    }

    for current_name_variant, updated_name_variant in dict_order_replacement.items():
        df["main_folder"] = df["main_folder"].replace(current_name_variant, updated_name_variant, regex=True)

    list_hue_order = [
        dict_order_replacement[exp.get_results_folder_name()]
        for exp in list_experiments
    ]

    # add escape character for latex rendering
    for i in range(len(order_legend)):
        order_legend[i] = dict_order_replacement[order_legend[i]]
        order_legend[i] = order_legend[i].replace('_', '\_')

    new_df = pd.DataFrame({
        column: []
        for column in df.columns
    })
    new_df = new_df.rename(columns={VALUE: VALUE_POST})
    for t in df[T].drop_duplicates():
        sub_df = df[df[T] == t]
        sub_df.loc[:, GEN] -= 1
        sub_df = sub_df.rename(columns={VALUE: VALUE_POST})
        print(sub_df)
        new_df = new_df.append(sub_df)
    print(new_df)

    print(df[GEN], new_df[GEN])

    merged_df = pd.merge(df, new_df, how="inner", on=[GEN, T, SUB_T])
    print(merged_df)
    merged_df[TOTAL_POPULATION_LOSS] = merged_df[VALUE_POST] - merged_df[VALUE]
    merged_df_with_decreasing = merged_df[merged_df[TOTAL_POPULATION_LOSS] < 0]

    merged_df_with_decreasing.loc[:, TOTAL_POPULATION_LOSS] *= -1

    # print(merged_df[[INCREASE, VALUE]])
    print(merged_df_with_decreasing)
    # merged_df_with_decreasing.sum(INCREASE)
    df_sum_loss_indiv = merged_df_with_decreasing.groupby([T, SUB_T])[TOTAL_POPULATION_LOSS].sum().to_frame()
    df_sum_loss_indiv = df_sum_loss_indiv.reset_index()
    # print(df_sum_loss_indiv)
    # --------------------------------

    COVERAGE_POS_50 = "coverage_pos_50"

    dict_new_name = {
        T: "Update Period",
        COVERAGE_POS_50: "Coverage",
    }

    df_sum_loss_indiv = df_sum_loss_indiv.rename(columns=dict_new_name)
    df = df.rename(columns=dict_new_name)

    axes[0].yaxis.grid(True)
    axes[1].yaxis.grid(True)
    axes[0].set_axisbelow(True)
    axes[1].set_axisbelow(True)

    sns.set_palette(sns.color_palette("colorblind", as_cmap=True))
    print(df_sum_loss_indiv)
    sns.boxplot(x=dict_new_name[T], y=TOTAL_POPULATION_LOSS, data=df_sum_loss_indiv, order=list_hue_order, ax=axes[0])

    idx = df.groupby(['subdir'])['gen'].transform(max) == df['gen']
    df_max_gen = df[idx]
    sns.boxplot(x=dict_new_name[T], y=dict_new_name[COVERAGE_POS_50], data=df_max_gen, order=list_hue_order, ax=axes[1])
    # axes[1].set_ylim(0,1)



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
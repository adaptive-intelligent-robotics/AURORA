# TODO: For both period and alpha?

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
import singularity.collections_experiments.hexapod_camera_vertical as exp_hexa

GEN = "gen"
VALUE = analysis.metrics.maze.MetricsMaze.SIZE_POP
T = analysis.metrics.maze.MetricsMaze.MAIN_FOLDER
SUB_T = analysis.metrics.maze.MetricsMaze.SUBDIR
VALUE_POST = analysis.metrics.maze.MetricsMaze.SIZE_POP + "_post"
INCREASE = "increase"


def main():
    df = pd.read_csv("/Users/looka/git/sferes2/exp/aurora/analysis/paper/dataframes/df.csv")

    list_experiments = [
        exp_hexa.DICT_HEXAPOD_CAMERA_VERTICAL_AURORA_UNIFORM_n_COLORS_VAT[latent_dim]
        for latent_dim in exp_hexa.LIST_DIMS_LATENT_SPACES_AURORA
    ]

    order_legend = [
        exp.get_results_folder_name()
        for exp in list_experiments
    ]

    print("\n".join(list(df["main_folder"].drop_duplicates())))
    df = df[df["main_folder"].isin(order_legend)]
    print(df)

    dict_order_replacement = {
        exp.get_results_folder_name(): f"Dim {exp.latent_space}" for exp in list_experiments
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
    merged_df[INCREASE] = merged_df[VALUE_POST] - merged_df[VALUE]
    merged_df_with_decreasing = merged_df[merged_df[INCREASE] < 0]

    merged_df_with_decreasing.loc[:, INCREASE] *= -1

    # print(merged_df[[INCREASE, VALUE]])
    print(merged_df_with_decreasing)
    # merged_df_with_decreasing.sum(INCREASE)
    df_sum_loss_indiv = merged_df_with_decreasing.groupby([T, SUB_T])[INCREASE].mean().to_frame()
    df_sum_loss_indiv = df_sum_loss_indiv.reset_index()
    # print(df_sum_loss_indiv)
    sns.set_style("whitegrid")
    print(df_sum_loss_indiv)
    g = sns.boxplot(x=T, y=INCREASE, data=df_sum_loss_indiv, order=list_hue_order, ax=ax)
    g.set_yscale("log")


if __name__ == '__main__':
    main()

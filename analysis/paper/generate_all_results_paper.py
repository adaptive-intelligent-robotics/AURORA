import os

import argparse
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

import analysis.paper.p_01_comparison_variants.e_01_comparison_variants as p_01_e_01_comparison_variants
import analysis.paper.p_01_comparison_variants.e_02_scatter_plots_variants as p_01_e_02_scatter_plots_variants
import analysis.paper.p_01_comparison_variants.e_03_comparison_fitness as p_01_e_03_comparison_fitness
import analysis.paper.p_01_comparison_variants.e_04_plots_fitness as p_01_e_04_plots_fitness
import analysis.paper.p_01_comparison_variants.e_05_influence_latent_dim as p_01_e_05_influence_latent_dim

import analysis.paper.p_02_analysis_learned_behavioural_space.e_01_meaningful_latent_space as p_02_e_01_meaningful_latent_space
import analysis.paper.p_02_analysis_learned_behavioural_space.e_02_air_hockey_diversity as p_02_e_02_air_hockey_diversity
import analysis.paper.p_02_analysis_learned_behavioural_space.e_03_hexapod_diversity as p_02_e_03_hexapod_diversity

import analysis.paper.p_03_ablation_study.e_01_plot_alpha as p_03_e_01_plot_alpha
import analysis.paper.p_03_ablation_study.e_03_plot_box_loss_indiv as p_03_e_03_plot_box_loss_indiv


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder-save-results", type=str, default="all_results_paper_2021_05_18")
    return parser.parse_args()


def main():
    args = get_args()
    folder_results = args.folder_save_results

    if not os.path.exists(folder_results):
        os.mkdir(folder_results)

    p_01_e_01_comparison_variants.generate_figure(path_save=os.path.join(folder_results, "tmp_p_01_e_01_comparison_variants.pdf"))
    p_01_e_01_comparison_variants.generate_figure(path_save=os.path.join(folder_results, "tmp_p_01_e_01_comparison_variants.svg"))
    p_01_e_02_scatter_plots_variants.generate_figure(path_save=os.path.join(folder_results, "p_01_e_02_scatter_plots_variants.pdf"))
    p_01_e_05_influence_latent_dim.generate_figure(path_save=os.path.join(folder_results, "p_01_e_05_influence_latent_dim.pdf"))

    p_02_e_01_meaningful_latent_space.generate_figure(path_save=os.path.join(folder_results, "p_02_e_01_meaningful_latent_space.pdf"))
    p_02_e_02_air_hockey_diversity.generate_figure(path_save=os.path.join(folder_results, "p_02_e_02_air_hockey_diversity.pdf"))
    p_02_e_03_hexapod_diversity.generate_figure(path_save=os.path.join(folder_results, "tmp_p_02_e_03_hexapod_diversity.pdf"))
    p_02_e_03_hexapod_diversity.generate_figure(path_save=os.path.join(folder_results, "tmp_p_02_e_03_hexapod_diversity.svg"))

    # p_03_e_01_plot_alpha.generate_figure(path_save=os.path.join(folder_results, "p_03_e_01_plot_alpha.pdf"))
    # p_03_e_03_plot_box_loss_indiv.generate_figure(path_save=os.path.join(folder_results, "p_03_e_03_plot_box_loss_indiv.pdf"))


if __name__ == '__main__':
    main()

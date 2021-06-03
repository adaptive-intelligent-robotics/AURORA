import copy
import os
import sys
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


sys.path.append('/Users/looka/git/sferes2/exp/aurora/submodules/figure_generator')
sys.path.append('/Users/looka/git/sferes2/exp/aurora/')

sys.path.append('/git/sferes2/exp/aurora/submodules/figure_generator')
sys.path.append('/git/sferes2/exp/aurora/')

from analysis.metrics.metrics import Metrics


import singularity.experiment
import singularity.collections_experiments.maze as maze_experiments
import singularity.collections_experiments.air_hockey as air_hockey_experiments
import singularity.collections_experiments.hexapod_camera_vertical as hexapod_camera_vertical_experiments

NAME_DF_AIR_HOCKEY = "df_air_hockey.csv"
NAME_DF_MAZE = "df_maze.csv"
NAME_DF_HEXAPOD_CAMERA_VERTICAL = "df_hexapod_camera_vertical.csv"
NAME_DF_HEXAPOD_CAMERA_VERTICAL_L = "df_hexapod_camera_vertical_l.csv"
PATH_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dataframes/")
LATENT_SPACE_DIM = "latent_space_dim"
NAME_VARIANT = "name_variant"
ENVIRONMENT = "environment"
UNIFORMITY = "Uniformity"

DICT_ALL_EXPERIMENTS = {
    "maze": maze_experiments.LIST_EXPERIMENTS,
    "air_hockey": air_hockey_experiments.LIST_EXPERIMENTS,
    "hexapod_camera_vertical": hexapod_camera_vertical_experiments.LIST_EXPERIMENTS,
}


def get_combined_df():
    for name_df in [NAME_DF_MAZE, NAME_DF_AIR_HOCKEY, NAME_DF_HEXAPOD_CAMERA_VERTICAL]:
        print(os.path.join(PATH_FOLDER, name_df))
        assert os.path.exists(os.path.join(PATH_FOLDER, name_df))

    df_air_hockey = pd.read_csv(os.path.join(PATH_FOLDER, NAME_DF_AIR_HOCKEY))
    df_maze = pd.read_csv(os.path.join(PATH_FOLDER, NAME_DF_MAZE))
    df_hexapod_camera_vertical = pd.read_csv(os.path.join(PATH_FOLDER, NAME_DF_HEXAPOD_CAMERA_VERTICAL))

    df = df_air_hockey.append(df_maze)
    # df = df.append(df_hexapod_camera_vertical)
    df = df.append(df_hexapod_camera_vertical)
    return df



def save_df(df, path=None):
    if not path:
        df.to_csv(os.path.join(PATH_FOLDER, "df.csv"))
    else:
        df.to_csv(path)



def add_latent_space_dim_to_df(df_old):
    list_results_folders = []
    list_latent_space_dim = []
    list_environments = []

    tmp_without_fit = copy.deepcopy([
        maze_experiments.MAZE_AURORA_NOVELTY_10_COLORS,
        hexapod_camera_vertical_experiments.CAMERA_VERTICAL_AURORA_10_COLORS_NOVELTY,
        maze_experiments.MAZE_AURORA_HARD_CODED_POS_NO_SELECTION,
        hexapod_camera_vertical_experiments.CAMERA_VERTICAL_HAND_CODED_GT_COLORS_NO_SELECTION,
        maze_experiments.MAZE_AURORA_SURPRISE_10_COLORS,
        hexapod_camera_vertical_experiments.CAMERA_VERTICAL_AURORA_10_COLORS_SURPRISE,
    ])
    for tmp in tmp_without_fit:
        tmp._has_fit = False

    for environment, list_experiments in DICT_ALL_EXPERIMENTS.items():
        for experiment in list_experiments:
            list_results_folders.append(experiment.get_results_folder_name())
            list_latent_space_dim.append(experiment.latent_space)
            list_environments.append(environment)

    for environment, experiment in zip(["maze", "hexapod_camera_vertical"] * 3, tmp_without_fit):
        list_results_folders.append(experiment.get_results_folder_name())
        list_latent_space_dim.append(experiment.latent_space)
        list_environments.append(environment)

    df_latent_space_dim = pd.DataFrame({
        Metrics.MAIN_FOLDER: list_results_folders,
        LATENT_SPACE_DIM: list_latent_space_dim,
        ENVIRONMENT: list_environments,
    })

    df = pd.merge(df_old,
                  df_latent_space_dim,
                  on=Metrics.MAIN_FOLDER,
                  how="inner",
                  )
    return df

def add_names_variants_to_df(df_old):
    list_results_folders = []
    list_name_variant_paper = []

    LIST_AURORA_UNIFORM_n_COLORS = [
        *maze_experiments.DICT_MAZE_AURORA_UNIFORM_n_COLORS.values(),
        *air_hockey_experiments.DICT_AIR_HOCKEY_AURORA_n_COLORS_UNIFORM.values(),
        *hexapod_camera_vertical_experiments.DICT_HEXAPOD_CAMERA_VERTICAL_AURORA_n_COLORS_UNIFORM.values(),
    ]

    LIST_AURORA_UNIFORM_n_COLORS_VAT = [
        *maze_experiments.DICT_MAZE_AURORA_UNIFORM_n_COLORS_VAT.values(),
        *air_hockey_experiments.DICT_AIR_HOCKEY_AURORA_UNIFORM_n_COLORS_VAT.values(),
        *hexapod_camera_vertical_experiments.DICT_HEXAPOD_CAMERA_VERTICAL_AURORA_UNIFORM_n_COLORS_VAT.values(),
    ]


    LIST_AURORA_UNIFORM_n_COLORS_NO_NORMALISATION = [
        *maze_experiments.DICT_MAZE_AURORA_UNIFORM_n_COLORS.values(),
        *air_hockey_experiments.DICT_AIR_HOCKEY_AURORA_n_COLORS_UNIFORM.values(),
        *hexapod_camera_vertical_experiments.DICT_HEXAPOD_CAMERA_VERTICAL_AURORA_n_COLORS_UNIFORM.values(),
    ]

    LIST_AURORA_UNIFORM_n_COLORS_VAT_NO_NORMALISATION = [
        *maze_experiments.DICT_MAZE_AURORA_UNIFORM_n_COLORS_VAT.values(),
        *air_hockey_experiments.DICT_AIR_HOCKEY_AURORA_UNIFORM_n_COLORS_VAT.values(),
        *hexapod_camera_vertical_experiments.DICT_HEXAPOD_CAMERA_VERTICAL_AURORA_UNIFORM_n_COLORS_VAT.values(),
    ]

    for experiment in LIST_AURORA_UNIFORM_n_COLORS:
        list_results_folders.append(experiment.get_results_folder_name())
        list_name_variant_paper.append(f"aurora_uniform_{experiment.latent_space}_psat")
        # exp_without_fit = copy.deepcopy(experiment)
        # exp_without_fit._has_fit = False
        # list_results_folders.append(exp_without_fit.get_results_folder_name())
        # list_name_variant_paper.append(f"aurora_uniform_{exp_without_fit.latent_space}_psat")

    for experiment in LIST_AURORA_UNIFORM_n_COLORS_VAT:
        list_results_folders.append(experiment.get_results_folder_name())
        list_name_variant_paper.append(f"aurora_uniform_{experiment.latent_space}_vat")
        # exp_without_fit = copy.deepcopy(experiment)
        # exp_without_fit._has_fit = False
        # list_results_folders.append(exp_without_fit.get_results_folder_name())
        # list_name_variant_paper.append(f"aurora_uniform_{exp_without_fit.latent_space}_vat")

    for experiment in LIST_AURORA_UNIFORM_n_COLORS_NO_NORMALISATION:
        list_results_folders.append(experiment.get_results_folder_name())
        list_name_variant_paper.append(f"aurora_uniform_{experiment.latent_space}_no_norm")
        # exp_without_fit = copy.deepcopy(experiment)
        # exp_without_fit._has_fit = False
        # list_results_folders.append(exp_without_fit.get_results_folder_name())
        # list_name_variant_paper.append(f"aurora_uniform_{exp_without_fit.latent_space}_no_norm")

    for experiment in LIST_AURORA_UNIFORM_n_COLORS_VAT_NO_NORMALISATION:
        list_results_folders.append(experiment.get_results_folder_name())
        list_name_variant_paper.append(f"aurora_uniform_{experiment.latent_space}_vat_no_norm")
        # exp_without_fit = copy.deepcopy(experiment)
        # exp_without_fit._has_fit = False
        # list_results_folders.append(exp_without_fit.get_results_folder_name())
        # list_name_variant_paper.append(f"aurora_uniform_{exp_without_fit.latent_space}_vat_no_norm")

    for experiment in [
        maze_experiments.MAZE_AURORA_CURIOSITY_10_COLORS,
        air_hockey_experiments.AIR_HOCKEY_AURORA_10_COLORS_CURIOSITY,
        hexapod_camera_vertical_experiments.CAMERA_VERTICAL_AURORA_10_COLORS_CURIOSITY,
    ]:
        list_results_folders.append(experiment.get_results_folder_name())
        list_name_variant_paper.append(f"aurora_curiosity_{experiment.latent_space}_psat")

    # tmp_without_fit = copy.deepcopy([
    #     maze_experiments.MAZE_AURORA_NOVELTY_10_COLORS,
    #     hexapod_camera_vertical_experiments.CAMERA_VERTICAL_AURORA_10_COLORS_NOVELTY,
    # ])
    # for tmp in tmp_without_fit:
    #     tmp._has_fit = False
    for experiment in [
        maze_experiments.MAZE_AURORA_NOVELTY_10_COLORS,
        air_hockey_experiments.AIR_HOCKEY_AURORA_10_COLORS_NOVELTY,
        hexapod_camera_vertical_experiments.CAMERA_VERTICAL_AURORA_10_COLORS_NOVELTY,
        # *tmp_without_fit
    ]:
        list_results_folders.append(experiment.get_results_folder_name())
        list_name_variant_paper.append(f"aurora_novelty_{experiment.latent_space}_psat")

    for experiment in [
        maze_experiments.MAZE_AURORA_NOVELTY_SURPRISE_10_COLORS,
        air_hockey_experiments.AIR_HOCKEY_AURORA_10_COLORS_NOVELTY_SURPRISE,
        hexapod_camera_vertical_experiments.CAMERA_VERTICAL_AURORA_10_COLORS_NOVELTY_SURPRISE,
    ]:
        list_results_folders.append(experiment.get_results_folder_name())
        list_name_variant_paper.append(f"aurora_nov_sur_{experiment.latent_space}_psat")

    for experiment in [
        maze_experiments.MAZE_AURORA_HARD_CODED_POS,  # TODO: To change
        air_hockey_experiments.AIR_HOCKEY_HAND_CODED_GT,
        hexapod_camera_vertical_experiments.CAMERA_VERTICAL_HAND_CODED_GT,
    ]:
        list_results_folders.append(experiment.get_results_folder_name())
        list_name_variant_paper.append(f"qd_uniform_psat")

    # tmp_without_fit = copy.deepcopy([
    #     maze_experiments.MAZE_AURORA_HARD_CODED_POS_NO_SELECTION,
    #     hexapod_camera_vertical_experiments.CAMERA_VERTICAL_HAND_CODED_GT_COLORS_NO_SELECTION,
    # ])
    # for tmp in tmp_without_fit:
    #     tmp._has_fit = False
    for experiment in [
        maze_experiments.MAZE_AURORA_HARD_CODED_POS_NO_SELECTION,
        air_hockey_experiments.AIR_HOCKEY_HAND_CODED_GT_COLORS_NO_SELECTION,
        hexapod_camera_vertical_experiments.CAMERA_VERTICAL_HAND_CODED_GT_COLORS_NO_SELECTION,
        # *tmp_without_fit
    ]:
        print(experiment.get_results_folder_name())
        list_results_folders.append(experiment.get_results_folder_name())
        list_name_variant_paper.append(f"qd_no_selection_psat")

    for experiment in [
        maze_experiments.DICT_MAZE_TAXONS_n_COLORS[10],
        air_hockey_experiments.DICT_AIR_HOCKEY_TAXONS_n_COLORS[10],
        hexapod_camera_vertical_experiments.DICT_CAMERA_VERTICAL_TAXONS_n_COLORS[10],
    ]:
        list_results_folders.append(experiment.get_results_folder_name())
        list_name_variant_paper.append(f"TAXONS_{experiment.latent_space}")

    for experiment in [
        maze_experiments.MAZE_TAXO_NOVELTY_10_COLORS,
        air_hockey_experiments.AIR_HOCKEY_TAXO_NOVELTY_10_COLORS,
        hexapod_camera_vertical_experiments.CAMERA_VERTICAL_TAXO_NOVELTY_10_COLORS,
    ]:
        list_results_folders.append(experiment.get_results_folder_name())
        list_name_variant_paper.append(f"TAXO_N_{experiment.latent_space}")

    for experiment in [
        maze_experiments.MAZE_TAXO_SURPRISE_10_COLORS,
        air_hockey_experiments.AIR_HOCKEY_TAXO_SURPRISE_10_COLORS,
        hexapod_camera_vertical_experiments.CAMERA_VERTICAL_TAXO_SURPRISE_10_COLORS,
    ]:
        list_results_folders.append(experiment.get_results_folder_name())
        list_name_variant_paper.append(f"TAXO_S_{experiment.latent_space}")

    for experiment in [
        maze_experiments.MAZE_TAXONS_HARD_CODED_POS_ELITISM,
        air_hockey_experiments.AIR_HOCKEY_TAXONS_HARD_CODED_POS_ELITISM,
        hexapod_camera_vertical_experiments.CAMERA_VERTICAL_TAXONS_HARD_CODED_POS_ELITISM,
    ]:
        list_results_folders.append(experiment.get_results_folder_name())
        list_name_variant_paper.append(f"NS")

    # for experiment in [
    #     maze_experiments.MAZE_AURORA_HARD_CODED_POS,
    #     hexapod_camera_vertical_experiments.CAMERA_VERTICAL_HAND_CODED_GT,
    #     air_hockey_experiments.AIR_HOCKEY_HAND_CODED_GT,
    # ]:
    #     list_results_folders.append(experiment.get_results_folder_name())
    #     list_name_variant_paper.append(f"qd_uniform_psat_fit")
    #
    # for experiment in [
    #     maze_experiments.DICT_MAZE_AURORA_UNIFORM_n_COLORS[10],
    #     hexapod_camera_vertical_experiments.DICT_HEXAPOD_CAMERA_VERTICAL_AURORA_n_COLORS_UNIFORM[10],
    #     air_hockey_experiments.DICT_AIR_HOCKEY_AURORA_n_COLORS_UNIFORM[10],
    # ]:
    #     list_results_folders.append(experiment.get_results_folder_name())
    #     list_name_variant_paper.append(f"aurora_uniform_{experiment.latent_space}_psat_fit")

    # TODO: Hard maze sticky.

    print(list_results_folders)

    df_latent_space_dim = pd.DataFrame({
        Metrics.MAIN_FOLDER: list_results_folders,
        NAME_VARIANT: list_name_variant_paper,
    })
    print(df_latent_space_dim)

    df = pd.merge(df_old,
                  df_latent_space_dim,
                  on=Metrics.MAIN_FOLDER,
                  how="outer")

    return df


def get_preprocessed_df(df=None, treat_uniformity=True):
    if df is None:
        df = get_combined_df()
    # save_df(df)
    df = add_latent_space_dim_to_df(df)
    df = add_names_variants_to_df(df)

    if treat_uniformity:
        df.rename(columns={"JDS_uniform_explored": UNIFORMITY}, inplace=True)
        df[UNIFORMITY] = 1 - df[UNIFORMITY]

    return df


def main():

    df = get_preprocessed_df()
    save_df(df)

    df_l = pd.read_csv(os.path.join(PATH_FOLDER, NAME_DF_HEXAPOD_CAMERA_VERTICAL_L))
    df_l = get_preprocessed_df(df_l, treat_uniformity=False)
    save_df(df_l, path=os.path.join(PATH_FOLDER, "df_l.csv"))


    df_analysis_suppl_features = pd.read_csv("p_02_analysis_learned_behavioural_space/data_obs_hexapod/df_hexapod_camera_vertical.csv")
    df_analysis_suppl_features = get_preprocessed_df(df_analysis_suppl_features)

    if os.path.exists("p_02_analysis_learned_behavioural_space/data_obs_hexapod/df_hexapod_camera_vertical_3_4.csv"):
        df_analysis_suppl_features_3_4 = pd.read_csv("p_02_analysis_learned_behavioural_space/data_obs_hexapod/df_hexapod_camera_vertical_3_4.csv")
        df_analysis_suppl_features_3_4 = get_preprocessed_df(df_analysis_suppl_features_3_4)
        df_analysis_suppl_features = pd.concat([df_analysis_suppl_features, df_analysis_suppl_features_3_4])

    save_df(df_analysis_suppl_features, "p_02_analysis_learned_behavioural_space/data_obs_hexapod/df_hexapod_camera_vertical_processed.csv")


if __name__ == '__main__':
    main()

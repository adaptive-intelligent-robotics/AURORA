import glob
import os
import sys
import traceback
from collections import defaultdict
from typing import Tuple, Dict, List

import jinja2
import numpy as np
import pandas as pd
import seaborn as sns

sys.path.append('/Users/looka/git/sferes2/exp/aurora/submodules/figure_generator')
sys.path.append('/Users/looka/git/sferes2/exp/aurora/')

from configuration_analysis import *

sys.path.append('/git/sferes2/exp/aurora/submodules/figure_generator')
sys.path.append('/git/sferes2/exp/aurora/')

from mosaic import Mosaic

from data_reader import create_html_plot, read_data, get_data_proj
from analysis.metrics.hexapod_walls import MetricsWalls
from analysis.metrics.maze import MetricsMaze
from singularity.experiment import Experiment
from singularity.factory_experiments import DICT_EXPERIMENTS_TO_LAUNCH, get_all_experiments_from_dict
import experiment_serialiser

UPLOAD = '/UPLOAD/'

# TODO: Add other metric: ENTROPY


def get_data_metric(file_path_metric_archive: str) -> Tuple[np.ndarray, np.ndarray]:
    dict_data_per_component = read_data(file_path_metric_archive,
                                        ["1", "2"])
    pop_index = dict_data_per_component['1'][:, 0]
    metric_component = dict_data_per_component['2'][:, 0]
    return pop_index, metric_component


def convert_to_file_name(str_:str):
    str_ = str_.lower().replace('-', ' ').split(' ')
    return '_'.join([x for x in str_ if x])


def load_serialised_list_experiments(path_main_folder):
    path_serialised_list_experiments = os.path.abspath(
        os.path.join(path_main_folder,
                     experiment_serialiser.NAME_FILE_SERIALISED_LIST_EXPERIMENTS)
    )
    try:
        loaded_list_experiments = experiment_serialiser.load_serialised_experiments(path_serialised_list_experiments)
    except:
        print(traceback.format_exc())
        loaded_list_experiments = None
    return loaded_list_experiments


def is_in_list_experiments(exp, list_experiments, default_list_experiments):
    if not list_experiments:
        return exp in default_list_experiments
    else:
        return exp in list_experiments

def create_html_figures(folder_results_loaded,
                        folder_results_to_save,
                        list_gen,
                        name_exp=None):
    if not name_exp:
        prefix_exp = ''
    else:
        prefix_exp = f'{name_exp}_'

    list_gen_that_exist = []
    for gen in list_gen:
        path = os.path.join(folder_results_loaded, f'proj_{gen}.dat')
        path_metric = os.path.join(folder_results_loaded, f'metric_{gen}.dat')
        print("create_html_figures", path)

        if os.path.exists(path):
            list_gen_that_exist.append(gen)
            latent_component, ground_truth_component, *_ = get_data_proj(path) # TODO : also read archive_...dat to get

            # Try to get a metric component if these is one.
            try:
                _, metric_component = get_data_metric(path_metric)
            except:
                metric_component = None
            # for _metric_component in [metric_component, None]:
            _metric_component = metric_component
            if _metric_component is not None:
                suffix = "_metric"
            else:
                suffix = ""
            create_html_plot(path_saved_plot=os.path.join(folder_results_to_save, f"{prefix_exp}gt-rot_color-pos_gen_{gen:07}{suffix}"),
                             plot_component=ground_truth_component,
                             color_component=ground_truth_component,
                             indexes_plot_component=(3, 4, 5),
                             indexes_color_component=(0, 1, 2),
                             added_metric_component=_metric_component
                             )
            create_html_plot(path_saved_plot=os.path.join(folder_results_to_save, f"{prefix_exp}gt-pos_color-pos_gen_{gen:07}{suffix}"),
                             plot_component=ground_truth_component,
                             color_component=ground_truth_component,
                             indexes_plot_component=(0, 1, 2),
                             indexes_color_component=(0, 1, 2),
                             added_metric_component=_metric_component
                             )

            create_html_plot(path_saved_plot=os.path.join(folder_results_to_save, f"{prefix_exp}gt-pos_color-rot_gen_{gen:07}{suffix}"),
                             plot_component=ground_truth_component,
                             color_component=ground_truth_component,
                             indexes_plot_component=(0, 1, 2),
                             indexes_color_component=(3, 4, 5),
                             added_metric_component=_metric_component
                             )

            create_html_plot(path_saved_plot=os.path.join(folder_results_to_save, f"{prefix_exp}gt-rot_color-rot_gen_{gen:07}{suffix}"),
                             plot_component=ground_truth_component,
                             color_component=ground_truth_component,
                             indexes_plot_component=(3, 4, 5),
                             indexes_color_component=(3, 4, 5),
                             added_metric_component=_metric_component
                             )
    return list_gen_that_exist


def create_figures_maze(folder_results_loaded,
                        folder_results_to_save,
                        list_gen,
                        experiment: Experiment,
                        ):
    sns.set_style("white")

    list_gen_that_exist = []

    for gen in list_gen:
        path = os.path.join(folder_results_loaded, f'proj_{gen}.dat')
        print("create_figures_maze", path)

        if os.path.exists(path):
            list_gen_that_exist.append(gen)
            latent_component, ground_truth_component, *_ = get_data_proj(path) # TODO : also read archive_...dat to get

            # Flipping image to show it correctly in Figure
            ground_truth_component[:, 1] = 600 - ground_truth_component[:, 1]

            create_html_plot(path_saved_plot=os.path.join(folder_results_to_save,
                                                          f"gt_color_gt_{gen:07}"),
                             plot_component=ground_truth_component,
                             color_component=ground_truth_component,
                             indexes_plot_component=(0, 1),
                             indexes_color_component=(0, 1),
                             )

            if 2 <= np.size(latent_component, axis=1) <= 3:
                indexes_latent = tuple(range(np.size(latent_component, axis=1)))

                create_html_plot(path_saved_plot=os.path.join(folder_results_to_save,
                                                              f"latent_color_latent_{gen:07}"),
                                 plot_component=latent_component,
                                 color_component=latent_component,
                                 indexes_plot_component=indexes_latent,
                                 indexes_color_component=indexes_latent,
                                 )



                create_html_plot(path_saved_plot=os.path.join(folder_results_to_save,
                                                              f"latent_color_gt_{gen:07}"),
                                 plot_component=latent_component,
                                 color_component=ground_truth_component,
                                 indexes_plot_component=indexes_latent,
                                 indexes_color_component=(0, 1),
                                 )

                create_html_plot(path_saved_plot=os.path.join(folder_results_to_save,
                                                              f"gt_color_latent_{gen:07}"),
                                 plot_component=ground_truth_component,
                                 color_component=latent_component,
                                 indexes_plot_component=(0, 1),
                                 indexes_color_component=indexes_latent,
                                 )

            # Write graph sequences
            if experiment.do_consider_bumpers:
                size_one_input = 5
            else:
                size_one_input = 3
            # TODO : TO DEBUG
            # write_graphs_sequences(folder_results_loaded, folder_results_to_save, gen, size_one_input)

    return list_gen_that_exist


def create_figures_hexapod_camera_vertical(folder_results_loaded,
                                           folder_results_to_save,
                                           list_gen,
                                           ):
    sns.set_style("white")

    list_gen_that_exist = []

    for gen in list_gen:
        path = os.path.join(folder_results_loaded, f'proj_{gen}.dat')
        print("create_figures_hexapod_camera_vertical -", path)

        if os.path.exists(path):
            list_gen_that_exist.append(gen)
            latent_component, ground_truth_component, *_ = get_data_proj(path) # TODO : also read archive_...dat to get

            create_html_plot(path_saved_plot=os.path.join(folder_results_to_save,
                                                          f"gt_color_gt_{gen:07}"),
                             plot_component=ground_truth_component,
                             color_component=ground_truth_component,
                             indexes_plot_component=(0, 1),  # x, y positions
                             indexes_color_component=(0, 1),
                             )


            ground_truth_component[ground_truth_component[:, 4] < -np.pi / 2, 4] += 2 * np.pi

            create_html_plot(path_saved_plot=os.path.join(folder_results_to_save,
                                                          f"orientation_color_pos_{gen:07}"),
                             plot_component=ground_truth_component,
                             color_component=ground_truth_component,
                             indexes_plot_component=(3, 4),  # x, y positions
                             indexes_color_component=(0, 1),
                             )

            if 2 <= np.size(latent_component, axis=1) <= 3:
                indexes_latent = tuple(range(np.size(latent_component, axis=1)))

                create_html_plot(path_saved_plot=os.path.join(folder_results_to_save,
                                                              f"latent_color_latent_{gen:07}"),
                                 plot_component=latent_component,
                                 color_component=latent_component,
                                 indexes_plot_component=indexes_latent,
                                 indexes_color_component=indexes_latent,
                                 )

                create_html_plot(path_saved_plot=os.path.join(folder_results_to_save,
                                                              f"latent_color_gt_{gen:07}"),
                                 plot_component=latent_component,
                                 color_component=ground_truth_component,
                                 indexes_plot_component=indexes_latent,
                                 indexes_color_component=(0, 1),
                                 )

                create_html_plot(path_saved_plot=os.path.join(folder_results_to_save,
                                                              f"gt_color_latent_{gen:07}"),
                                 plot_component=ground_truth_component,
                                 color_component=latent_component,
                                 indexes_plot_component=(0, 1),
                                 indexes_color_component=indexes_latent,
                                 )

    return list_gen_that_exist


def create_figures_air_hockey(folder_results_loaded,
                              folder_results_to_save,
                              list_gen,
                              ):
    sns.set_style("white")

    list_gen_that_exist = []

    for gen in list_gen:
        path = os.path.join(folder_results_loaded, f'proj_{gen}.dat')
        print("create_figures_air_hockey -", path)

        if os.path.exists(path):
            list_gen_that_exist.append(gen)
            latent_component, ground_truth_component, *_ = get_data_proj(path) # TODO : also read archive_...dat to get

            create_html_plot(path_saved_plot=os.path.join(folder_results_to_save,
                                                          f"gt_color_gt_{gen:07}"),
                             plot_component=ground_truth_component,
                             color_component=ground_truth_component,
                             indexes_plot_component=(0, 1),  # x, y positions
                             indexes_color_component=(0, 1),
                             )

            if 2 <= np.size(latent_component, axis=1) <= 3:
                indexes_latent = tuple(range(np.size(latent_component, axis=1)))

                create_html_plot(path_saved_plot=os.path.join(folder_results_to_save,
                                                              f"latent_color_latent_{gen:07}"),
                                 plot_component=latent_component,
                                 color_component=latent_component,
                                 indexes_plot_component=indexes_latent,
                                 indexes_color_component=indexes_latent,
                                 )

                create_html_plot(path_saved_plot=os.path.join(folder_results_to_save,
                                                              f"latent_color_gt_{gen:07}"),
                                 plot_component=latent_component,
                                 color_component=ground_truth_component,
                                 indexes_plot_component=indexes_latent,
                                 indexes_color_component=(0, 1),
                                 )

                create_html_plot(path_saved_plot=os.path.join(folder_results_to_save,
                                                              f"gt_color_latent_{gen:07}"),
                                 plot_component=ground_truth_component,
                                 color_component=latent_component,
                                 indexes_plot_component=(0, 1),
                                 indexes_color_component=indexes_latent,
                                 )

    return list_gen_that_exist


def create_mosaic(folder_loaded_results,
                  folder_saved_results,
                  list_gen,
                  width):
    list_gen_mosaic_that_exist = []
    for gen in list_gen:
        list_paths_files_observation = sorted(glob.glob(
            os.path.join(folder_loaded_results, f"observation_gen_{gen:07}_indiv_*_color.png")))  # TODO Be careful, the name might change when using black and white images
        # print("list_paths_files_observation", '\n'.join(list_paths_files_observation[:5]))

        list_indexes_individuals_observations = [
            int(path[-17:-10])
            for path in list_paths_files_observation
        ]

        if len(list_paths_files_observation) >= 2:

            number_obs = len(list_paths_files_observation)
            max_number_images_in_mosaic = 501

            step_consider_image = (number_obs // max_number_images_in_mosaic) + 1
            period_saving_indiv = list_indexes_individuals_observations[1] - list_indexes_individuals_observations[0]

            height = (number_obs // step_consider_image) // width

            try:
                Mosaic.generate_mixed_mosaic(
                    path_template_1=os.path.join(folder_loaded_results,
                                                 f"observation_gen_{gen:07}_indiv_{{0:07d}}_color.png"),
                    # TODO Correct path
                    path_template_2=os.path.join(folder_loaded_results,
                                                 f"reconstruction_obs_gen_{gen:07}_indiv_{{0:07d}}_rgb.png"),
                    # TODO Correct path
                    list_indivs=list(range(0, number_obs * period_saving_indiv, step_consider_image * period_saving_indiv))[:(width * height)],
                    width=width,
                    height=height,
                    path_save=os.path.join(folder_saved_results, f"mosaic_gen_{gen:07}.png"),
                    new_shape_1=(64, 64),
                    new_shape_2=(64, 64)
                )
                list_gen_mosaic_that_exist.append(gen)
            except:
                try:
                    Mosaic.generate_mosaic(
                        path_template=os.path.join(folder_loaded_results,
                                                     f"observation_gen_{gen:07}_indiv_{{0:07d}}_color.png"),
                        # TODO Correct path
                        list_gen=list(range(0, number_obs * period_saving_indiv, step_consider_image * period_saving_indiv))[:(width * height)],
                        width=width,
                        height=height,
                        path_save=os.path.join(folder_saved_results, f"obs_mosaic_gen_{gen:07}.png"), # TODO TO change and take into account
                        new_shape=(64, 64),
                    )
                    list_gen_mosaic_that_exist.append(gen)
                except:
                    pass
                try:
                    Mosaic.generate_mosaic(
                        path_template=os.path.join(folder_loaded_results,
                                                   f"reconstruction_obs_gen_{gen:07}_indiv_{{0:07d}}_rgb.png"),
                        # TODO Correct path
                        list_gen=list(range(0, number_obs * period_saving_indiv, step_consider_image * period_saving_indiv))[:(width * height)],
                        width=width,
                        height=height,
                        path_save=os.path.join(folder_saved_results, f"reconst_mosaic_gen_{gen:07}.png"), # TODO TO change and take into account
                        new_shape=(64, 64),
                    )
                    list_gen_mosaic_that_exist.append(gen)
                except:
                    pass

    return list_gen_mosaic_that_exist


def get_path_variant(name_executable: str, path):
    all_variant_folder_results_loaded = glob.glob(os.path.join(path,
                                                               f"results_{name_executable}",
                                                               f"results_{name_executable}.*")
                                                  )
    print("all_variant_folder_results_loaded", all_variant_folder_results_loaded)
    if all_variant_folder_results_loaded:
        path_variant_results_loaded = all_variant_folder_results_loaded[0]
        path_variant_results_loaded = os.path.abspath(os.path.join(path_variant_results_loaded, os.pardir))
        return path_variant_results_loaded


def get_path_variant_one_run(name_executable: str, path_main_folder):
    """
    :param name_executable:
    :param path_main_folder:
    :return: str of ONE complete path "/path/to/exp/results_name_executable/2020_..."
    """
    all_variant_folder_results_loaded = glob.glob(os.path.join(path_main_folder,
                                                               f"results_{name_executable}",
                                                               f"results_{name_executable}.*")
                                                  )
    print("all_variant_folder_results_loaded", all_variant_folder_results_loaded)
    for path_variant_results_loaded in all_variant_folder_results_loaded:
        if os.listdir(path_variant_results_loaded):
            return path_variant_results_loaded


def get_all_run_folders_variant(name_executable: str, path_main_folder):
    """
    :param name_executable:
    :param path_main_folder:
    :return: list of subdirectories ['2020_...', ...] in the folder of the results for "name_executable"
    """
    all_variant_folder_results_loaded = glob.glob(os.path.join(path_main_folder,
                                                               f"results_{name_executable}",
                                                               f"results_{name_executable}.*")
                                                  )
    print("all_variant_folder_results_loaded", all_variant_folder_results_loaded)
    return [os.path.basename(path_variant_results_loaded)
            for path_variant_results_loaded in all_variant_folder_results_loaded
            if os.listdir(path_variant_results_loaded)]


def get_formatted_list_gen(list_gen):
    return [f'{gen:07}' for gen in list_gen]


def get_formatted_name(name: str):
    name = name.replace('-', ' ').replace('_', ' ')
    name = '_'.join([part_name for part_name in name.split(' ') if part_name])
    return name


def get_all_available_results_paths(path_main_folder, ChosenFactoryExp):
    list_all_available_results_paths = []
    for experiment in DICT_EXPERIMENT_TO_LIST_EXPERIMENT[ChosenFactoryExp]:
        path_variant_results_loaded_exp = os.path.join(path_main_folder, f"results_{experiment.get_exec_name()}")
        if os.path.exists(path_variant_results_loaded_exp):
            list_all_available_results_paths.append(path_variant_results_loaded_exp)
    return list_all_available_results_paths


def get_list_variants_data_walls(path_main_folder, path_results_to_save, list_gen, df, list_experiments):
    list_variants = []

    for experiment in DICT_EXPERIMENT_TO_LIST_EXPERIMENT[WALLS]:

        # Checking if experiment is in the set of chosen experiments.
        # If not, we do not consider it
        if not is_in_list_experiments(exp=experiment,
                                  list_experiments=list_experiments,
                                  default_list_experiments=get_all_experiments_from_dict(DICT_EXPERIMENTS_TO_LAUNCH)):
            continue

        error_traceback = None
        list_info_subdirs = []

        try:  # TODO try except in each part. (and refactor into functions)
            path_variant_results_loaded_exp = os.path.join(path_main_folder, f"results_{experiment.get_exec_name()}")
            if not os.path.exists(path_variant_results_loaded_exp):
                # pass
                raise FileNotFoundError(f"The folder of results '{path_variant_results_loaded_exp}' is missing")

            all_run_folders_variant = get_all_run_folders_variant(experiment.get_exec_name(), path_main_folder)

            if not all_run_folders_variant:
                # pass
                raise FileNotFoundError("there should be at least one valid folder of results")

            for subdir_results_2020 in all_run_folders_variant:

                info_subdir = {}
                path_variant_results_loaded_exp_one_run = os.path.join(path_variant_results_loaded_exp, subdir_results_2020)

                path_variant_results_to_save = os.path.join(path_results_to_save, experiment.get_exec_name(), subdir_results_2020)
                os.makedirs(path_variant_results_to_save)

                print("path_variant_results_to_save", path_variant_results_to_save)

                list_gen_that_exist = create_html_figures(path_variant_results_loaded_exp_one_run, path_variant_results_to_save, list_gen)
                list_gen_mosaic_that_exist = create_mosaic(path_variant_results_loaded_exp_one_run, path_variant_results_to_save, list_gen, width=12)

                info_subdir['subdir'] = subdir_results_2020
                info_subdir['list_gen'] = get_formatted_list_gen(list_gen_that_exist)
                info_subdir['list_gen_mosaic'] = get_formatted_list_gen(list_gen_mosaic_that_exist)
                list_info_subdirs.append(info_subdir)

            # Metric graph ({title: , path_pos: , ...}])
            path_pos, path_rot, path_gt, path_size_pop, path_mutual_information, path_l = MetricsWalls.generate_figures_per_variant(df, experiment, os.path.join(path_results_to_save, experiment.get_exec_name()))
        except Exception as e:
            path_pos, path_rot, path_gt, path_size_pop, path_mutual_information, path_l = '', '', '', '', '', ''
            error_traceback = traceback.format_exc()


        list_variants.append({
            'name': experiment.get_exec_name().replace('_', ' ').replace('-', ' '),
            'name_exec': experiment.get_exec_name(),
            'list_info_subdirs': list_info_subdirs,
            'graph': {
                'path_pos': os.path.join(UPLOAD, path_pos),
                'path_rot': os.path.join(UPLOAD, path_rot),
                'path_gt': os.path.join(UPLOAD, path_gt),
                'path_size_pop': os.path.join(UPLOAD, path_size_pop),
                'path_mutual_information': os.path.join(UPLOAD, path_mutual_information),
                'path_l': os.path.join(UPLOAD, path_l),
            },
            'error_traceback': error_traceback,
            'table_info': repr(experiment),
            'hash': hash(experiment),
        })

    return list_variants


def get_list_variants_data_hexapod_camera_vertical(path_main_folder, path_results_to_save, list_gen, df, df_l, list_experiments, raise_exception=False):
    list_variants = []

    for experiment in DICT_EXPERIMENT_TO_LIST_EXPERIMENT[HEXAPOD_CAMERA_VERTICAL]:

        # Checking if experiment is in the set of chosen experiments.
        # If not, we do not consider it
        if not is_in_list_experiments(exp=experiment,
                                      list_experiments=list_experiments,
                                      default_list_experiments=get_all_experiments_from_dict(DICT_EXPERIMENTS_TO_LAUNCH)):
            continue

        error_traceback = None
        list_info_subdirs = []

        dict_paths = defaultdict(str)

        PATH_POS = 'path_pos'
        PATH_SIZE_POP = 'path_size_pop'
        PATH_L = 'path_l'
        PATH_ENTROPY = "path_entropy"
        PATH_JDS_EXPLORED = "path_jds_explored"
        PATH_JDS_ENTIRE = "path_jds_entire"
        PATH_MEAN_FITNESS = "path_mean_fitness"
        PATH_MAX_FITNESS = "path_max_fitness"
        PATH_NOVELTY = "path_novelty"
        PATH_ANGLE_COVERAGE = "path_angle_coverage"

        LIST_LEGENDS_PATHS = [PATH_POS, PATH_SIZE_POP, PATH_L, PATH_ENTROPY, PATH_JDS_EXPLORED, PATH_JDS_ENTIRE,
                              PATH_MEAN_FITNESS, PATH_MAX_FITNESS, PATH_NOVELTY, PATH_ANGLE_COVERAGE]

        try:  # TODO try except in each part. (and refactor into functions)
            path_variant_results_loaded_exp = os.path.join(path_main_folder, f"results_{experiment.get_exec_name()}")
            if not os.path.exists(path_variant_results_loaded_exp):
                # pass
                raise FileNotFoundError(f"The folder of results '{path_variant_results_loaded_exp}' is missing")

            all_run_folders_variant = get_all_run_folders_variant(experiment.get_exec_name(), path_main_folder)

            if not all_run_folders_variant:
                # pass
                raise FileNotFoundError("there should be at least one valid folder of results")

            for subdir_results_2020 in all_run_folders_variant:

                info_subdir = {}
                path_variant_results_loaded_exp_one_run = os.path.join(path_variant_results_loaded_exp, subdir_results_2020)

                path_variant_results_to_save = os.path.join(path_results_to_save, experiment.get_exec_name(), subdir_results_2020)
                os.makedirs(path_variant_results_to_save)

                print("path_variant_results_to_save", path_variant_results_to_save)

                list_gen_that_exist = create_figures_hexapod_camera_vertical(path_variant_results_loaded_exp_one_run, path_variant_results_to_save, list_gen)
                list_gen_mosaic_that_exist = create_mosaic(path_variant_results_loaded_exp_one_run, path_variant_results_to_save, list_gen, width=12)

                info_subdir['subdir'] = subdir_results_2020
                info_subdir['list_gen'] = get_formatted_list_gen(list_gen_that_exist)
                info_subdir['list_gen_mosaic'] = get_formatted_list_gen(list_gen_mosaic_that_exist)
                list_info_subdirs.append(info_subdir)

                MetricsHexapodCameraVertical.save_gif_coverage(path_folder_load=path_variant_results_loaded_exp_one_run,
                                                               path_folder_save=path_variant_results_to_save,
                                                               list_nb_div=[10, 40], do_save_scatter=True)

            # Metric graph ({title: , path_pos: , ...}])

            dict_paths[PATH_POS], \
            dict_paths[PATH_SIZE_POP], \
            dict_paths[PATH_L], \
            dict_paths[PATH_ENTROPY], \
            dict_paths[PATH_JDS_EXPLORED], \
            dict_paths[PATH_JDS_ENTIRE], \
            dict_paths[PATH_MEAN_FITNESS], \
            dict_paths[PATH_MAX_FITNESS], \
            dict_paths[PATH_NOVELTY], \
            dict_paths[PATH_ANGLE_COVERAGE] = MetricsHexapodCameraVertical.generate_figures_per_variant(df, df_l, experiment, os.path.join(path_results_to_save, experiment.get_exec_name()))
        except FileNotFoundError:
            error_traceback = traceback.format_exc()
        except Exception as e:
            if raise_exception:
                print(traceback.format_exc())
                raise
            error_traceback = traceback.format_exc()

        DICT_LEGEND_PATH_X_TO_UPLOAD_PATH = {LEGEND_PATH_X: os.path.join(UPLOAD, dict_paths[LEGEND_PATH_X])
                                             for LEGEND_PATH_X in LIST_LEGENDS_PATHS}

        list_variants.append({
            'name': experiment.get_exec_name().replace('_', ' ').replace('-', ' '),
            'name_exec': experiment.get_exec_name(),
            'list_info_subdirs': list_info_subdirs,
            'graph': DICT_LEGEND_PATH_X_TO_UPLOAD_PATH,
            'error_traceback': error_traceback,
            'table_info': repr(experiment),
            'hash': hash(experiment),
        })

    return list_variants


def get_list_variants_data_air_hockey(path_main_folder, path_results_to_save, list_gen, df, list_experiments, raise_exception=False):
    list_variants = []

    for experiment in DICT_EXPERIMENT_TO_LIST_EXPERIMENT[AIR_HOCKEY]:

        # Checking if experiment is in the set of chosen experiments.
        # If not, we do not consider it
        if not is_in_list_experiments(exp=experiment,
                                      list_experiments=list_experiments,
                                      default_list_experiments=get_all_experiments_from_dict(DICT_EXPERIMENTS_TO_LAUNCH)):
            continue

        error_traceback = None
        list_info_subdirs = []

        dict_paths = defaultdict(str)

        PATH_POS = 'path_pos'
        PATH_SIZE_POP = 'path_size_pop'
        PATH_L = 'path_l'
        PATH_ENTROPY = "path_entropy"
        PATH_JDS_EXPLORED = "path_jds_explored"
        PATH_JDS_ENTIRE = "path_jds_entire"
        PATH_MEAN_FITNESS = "path_mean_fitness"
        PATH_MAX_FITNESS = "path_max_fitness"
        PATH_NOVELTY = "path_novelty"
        PATH_DIVERSITY = "path_diversity"

        LIST_LEGENDS_PATHS = [PATH_POS, PATH_SIZE_POP, PATH_L, PATH_ENTROPY, PATH_JDS_EXPLORED, PATH_JDS_ENTIRE,
                              PATH_MEAN_FITNESS, PATH_MAX_FITNESS, PATH_NOVELTY, PATH_DIVERSITY]

        try:  # TODO try except in each part. (and refactor into functions)
            path_variant_results_loaded_exp = os.path.join(path_main_folder, f"results_{experiment.get_exec_name()}")
            if not os.path.exists(path_variant_results_loaded_exp):
                # pass
                raise FileNotFoundError(f"The folder of results '{path_variant_results_loaded_exp}' is missing")

            all_run_folders_variant = get_all_run_folders_variant(experiment.get_exec_name(), path_main_folder)

            if not all_run_folders_variant:
                # pass
                raise FileNotFoundError("there should be at least one valid folder of results")

            for subdir_results_2020 in all_run_folders_variant:

                info_subdir = {}
                path_variant_results_loaded_exp_one_run = os.path.join(path_variant_results_loaded_exp, subdir_results_2020)

                path_variant_results_to_save = os.path.join(path_results_to_save, experiment.get_exec_name(), subdir_results_2020)
                os.makedirs(path_variant_results_to_save)

                print("path_variant_results_to_save", path_variant_results_to_save)

                list_gen_that_exist = create_figures_air_hockey(path_variant_results_loaded_exp_one_run, path_variant_results_to_save, list_gen)
                list_gen_mosaic_that_exist = create_mosaic(path_variant_results_loaded_exp_one_run, path_variant_results_to_save, list_gen, width=12)

                info_subdir['subdir'] = subdir_results_2020
                info_subdir['list_gen'] = get_formatted_list_gen(list_gen_that_exist)
                info_subdir['list_gen_mosaic'] = get_formatted_list_gen(list_gen_mosaic_that_exist)
                list_info_subdirs.append(info_subdir)

                MetricsAirHockey.save_gif_coverage(path_folder_load=path_variant_results_loaded_exp_one_run,
                                                   path_folder_save=path_variant_results_to_save,
                                                   list_nb_div=[10, 40], do_save_scatter=True)

            # Metric graph ({title: , path_pos: , ...}])

            dict_paths[PATH_POS], \
            dict_paths[PATH_SIZE_POP], \
            dict_paths[PATH_L], \
            dict_paths[PATH_ENTROPY], \
            dict_paths[PATH_JDS_EXPLORED], \
            dict_paths[PATH_JDS_ENTIRE], \
            dict_paths[PATH_MEAN_FITNESS], \
            dict_paths[PATH_MAX_FITNESS], \
            dict_paths[PATH_NOVELTY], \
            dict_paths[PATH_DIVERSITY] = MetricsAirHockey.generate_figures_per_variant(df, experiment, os.path.join(path_results_to_save, experiment.get_exec_name()))
        except FileNotFoundError:
            error_traceback = traceback.format_exc()
        except Exception as e:
            if raise_exception:
                print(traceback.format_exc())
                raise
            error_traceback = traceback.format_exc()

        DICT_LEGEND_PATH_X_TO_UPLOAD_PATH = {LEGEND_PATH_X: os.path.join(UPLOAD, dict_paths[LEGEND_PATH_X])
                                             for LEGEND_PATH_X in LIST_LEGENDS_PATHS}

        list_variants.append({
            'name': experiment.get_exec_name().replace('_', ' ').replace('-', ' '),
            'name_exec': experiment.get_exec_name(),
            'list_info_subdirs': list_info_subdirs,
            'graph': DICT_LEGEND_PATH_X_TO_UPLOAD_PATH,
            'error_traceback': error_traceback,
            'table_info': repr(experiment),
            'hash': hash(experiment),
        })

    return list_variants

# def run_in_forked_process(f):
#     def wrapf(q, f, args, kwargs):
#         q.put(f(*args, **kwargs))
#
#     @wraps(f)
#     def wrapper(*args, **kwargs):
#         q = Queue()
#         Process(target=wrapf, args=(q, f, args, kwargs)).start()
#         return q.get()
#
#     return wrapper
#
# @run_in_forked_process


def _generate_figures_subdir_results_2020(path_variant_results_loaded_exp,
                                          experiment,
                                          path_results_to_save,
                                          list_gen,
                                          subdir_results_2020):
    info_subdir = {}
    path_variant_results_loaded_exp_one_run = os.path.join(path_variant_results_loaded_exp, subdir_results_2020)

    path_variant_results_to_save = os.path.join(path_results_to_save, experiment.get_exec_name(), subdir_results_2020)
    os.makedirs(path_variant_results_to_save)

    print("path_variant_results_to_save", path_variant_results_to_save)

    list_gen_that_exist = create_figures_maze(path_variant_results_loaded_exp_one_run, path_variant_results_to_save, list_gen, experiment)

    info_subdir['subdir'] = subdir_results_2020
    info_subdir['list_gen'] = get_formatted_list_gen(list_gen_that_exist)
    MetricsMaze.save_gif_coverage(path_folder_load=path_variant_results_loaded_exp_one_run,
                                  path_folder_save=path_variant_results_to_save,
                                  list_nb_div=[10, 40], do_save_scatter=True)
    MetricsMaze.save_gif_offspring_analysis(path_folder_load=path_variant_results_loaded_exp_one_run,
                                            path_folder_save=path_variant_results_to_save,
                                            do_save_scatter=True)
    return info_subdir


def get_list_variants_data_maze(path_main_folder,
                                path_results_to_save,
                                list_gen,
                                df,
                                list_experiments):
    list_variants = []

    for experiment in DICT_EXPERIMENT_TO_LIST_EXPERIMENT[MAZE]:

        # Checking if experiment is in the set of chosen experiments.
        # If not, we do not consider it
        if not is_in_list_experiments(exp=experiment,
                                  list_experiments=list_experiments,
                                  default_list_experiments=get_all_experiments_from_dict(DICT_EXPERIMENTS_TO_LAUNCH)):
            continue

        error_traceback = None
        list_info_subdirs = []
        dict_paths = defaultdict(str)

        PATH_POS = 'path_pos'
        PATH_SIZE_POP = 'path_size_pop'
        PATH_L = 'path_l'
        PATH_ENTROPY = "path_entropy"
        PATH_JDS_EXPLORED = "path_jds_explored"
        PATH_JDS_ENTIRE = "path_jds_entire"
        PATH_MEAN_FITNESS = "path_mean_fitness"
        PATH_MAX_FITNESS = "path_max_fitness"
        PATH_NOVELTY = "path_novelty"

        LIST_LEGENDS_PATHS = [PATH_POS, PATH_SIZE_POP, PATH_L, PATH_ENTROPY, PATH_JDS_EXPLORED, PATH_JDS_ENTIRE,
                              PATH_MEAN_FITNESS, PATH_MAX_FITNESS, PATH_NOVELTY]

        try:  # TODO try except in each part. (and refactor into functions)
            path_variant_results_loaded_exp = os.path.join(path_main_folder, f"results_{experiment.get_exec_name()}")
            if not os.path.exists(path_variant_results_loaded_exp):
                # pass
                raise FileNotFoundError(f"The folder of results '{path_variant_results_loaded_exp}' is missing")

            all_run_folders_variant = get_all_run_folders_variant(experiment.get_exec_name(), path_main_folder)

            if not all_run_folders_variant:
                # pass
                raise FileNotFoundError("there should be at least one valid folder of results")

            list_tuple_generate_figures_subdir = [
                (path_variant_results_loaded_exp, experiment, path_results_to_save, list_gen, subdir_results_2020)
                for subdir_results_2020 in all_run_folders_variant
            ]

            #  with Pool(processes=number_processes) as pool:
                #  list_info_subdirs.extend(pool.starmap(_generate_figures_subdir_results_2020, list_tuple_generate_figures_subdir))
            for subdir_results_2020 in all_run_folders_variant:
                list_info_subdirs.append(_generate_figures_subdir_results_2020(path_variant_results_loaded_exp, 
                        experiment,
                        path_results_to_save,
                        list_gen,
                        subdir_results_2020))

        # Metric graph ({title: , path_pos: , ...}])
            dict_paths[PATH_POS], \
            dict_paths[PATH_SIZE_POP], \
            dict_paths[PATH_L],\
            dict_paths[PATH_ENTROPY], \
            dict_paths[PATH_JDS_EXPLORED], \
            dict_paths[PATH_JDS_ENTIRE], \
            dict_paths[PATH_MEAN_FITNESS], \
            dict_paths[PATH_MAX_FITNESS], \
            dict_paths[PATH_NOVELTY] = MetricsMaze.generate_figures_per_variant(df, experiment, os.path.join(path_results_to_save, experiment.get_exec_name()))

        except Exception as e:
            error_traceback = traceback.format_exc()

        DICT_LEGEND_PATH_X_TO_UPLOAD_PATH = {LEGEND_PATH_X: os.path.join(UPLOAD, dict_paths[LEGEND_PATH_X]) for LEGEND_PATH_X in LIST_LEGENDS_PATHS}

        list_variants.append({
            'name': experiment.get_exec_name().replace('_', ' ').replace('-', ' '),
            'name_exec': experiment.get_exec_name(),
            'list_info_subdirs': list_info_subdirs,
            'graph': DICT_LEGEND_PATH_X_TO_UPLOAD_PATH,
            'error_traceback': error_traceback,
            'table_info': repr(experiment),
            'hash': hash(experiment),
        })

    return list_variants


def at_least_one_executable_launched_in_comparison(dict_sub_comparison: Dict[Experiment, str],
                                                   list_experiments_launched: List[Experiment]
                                                   ) -> bool:
    if list_experiments_launched:
        return not set(dict_sub_comparison.keys()).isdisjoint(list_experiments_launched)
    return True


def get_list_metric_graphs_per_factory(path_main_folder, path_results_to_save, df, FACTORY, list_experiments_launched, df_l=None):

    METRICS_CLASS = DICT_EXPERIMENT_TO_METRIC[FACTORY]
    COMPARISONS = DICT_EXPERIMENT_TO_COMPARISON[FACTORY]

    list_metric_graphs = []
    for _name_comparison, dict_exec in COMPARISONS.items():

        # Disregard comparison if at least one experiment has not been launched:
        if not at_least_one_executable_launched_in_comparison(dict_sub_comparison=dict_exec,
                                                              list_experiments_launched=list_experiments_launched):
            continue

        print(f"COMPARISON - {_name_comparison})")
        name_comparison = get_formatted_name(_name_comparison)
        PATH_RESULTS_TO_SAVE_COMPARISON = os.path.join(path_results_to_save, name_comparison)
        if not os.path.exists(PATH_RESULTS_TO_SAVE_COMPARISON): # TODO : remove spaces
            os.mkdir(PATH_RESULTS_TO_SAVE_COMPARISON)

        dict_path_variants_legend = {get_path_variant(experiment_obj.get_exec_name(), path_main_folder): legend
                                     for experiment_obj, legend in dict_exec.items()
                                     if get_path_variant(experiment_obj.get_exec_name(), path_main_folder) is not None}
        error_traceback = None
        dict_paths = defaultdict(str)  # type: Dict[str, str]

        PATH_POS = 'path_pos'
        PATH_ROT = "path_rot"
        PATH_GT = "path_gt"
        PATH_SIZE_POP = "path_size_pop"
        PATH_MUTUAL_INFORMATION = "path_mutual_information"
        PATH_L = "path_l"
        PATH_ENTROPY = "path_entropy"
        PATH_JDS_EXPLORED = "path_jds_explored"
        PATH_JDS_ENTIRE = "path_jds_entire"
        PATH_MEAN_FITNESS = "path_mean_fitness"
        PATH_MAX_FITNESS = "path_max_fitness"
        PATH_NOVELTY = "path_novelty"
        PATH_DIVERSITY = "path_diversity"
        PATH_ANGLE_COVERAGE = "path_angle_coverage"

        LIST_LEGENDS_PATHS = [PATH_POS, PATH_ROT, PATH_GT, PATH_SIZE_POP, PATH_MUTUAL_INFORMATION, PATH_L,
                              PATH_ENTROPY, PATH_JDS_EXPLORED, PATH_JDS_ENTIRE, PATH_MEAN_FITNESS, PATH_MAX_FITNESS,
                              PATH_NOVELTY, PATH_DIVERSITY, PATH_ANGLE_COVERAGE]

        try:
            if issubclass(METRICS_CLASS, MetricsWalls):
                dict_paths[PATH_POS], \
                dict_paths[PATH_ROT], \
                dict_paths[PATH_GT], \
                dict_paths[PATH_SIZE_POP], \
                dict_paths[PATH_MUTUAL_INFORMATION], \
                dict_paths[PATH_L] = MetricsWalls.generate_figures(df,
                                                       dict_path_variants_legend=dict_path_variants_legend,
                                                       folder_path_save=PATH_RESULTS_TO_SAVE_COMPARISON)

            elif issubclass(METRICS_CLASS, MetricsMaze):
                dict_paths[PATH_POS], \
                dict_paths[PATH_SIZE_POP], \
                dict_paths[PATH_L], \
                dict_paths[PATH_ENTROPY], \
                dict_paths[PATH_JDS_EXPLORED], \
                dict_paths[PATH_JDS_ENTIRE], \
                dict_paths[PATH_MEAN_FITNESS], \
                dict_paths[PATH_MAX_FITNESS], \
                dict_paths[PATH_NOVELTY] = MetricsMaze.generate_figures(df,
                                                      dict_path_variants_legend=dict_path_variants_legend,
                                                      folder_path_save=PATH_RESULTS_TO_SAVE_COMPARISON)
            elif issubclass(METRICS_CLASS, MetricsHexapodCameraVertical):
                dict_paths[PATH_POS], \
                dict_paths[PATH_SIZE_POP], \
                dict_paths[PATH_L], \
                dict_paths[PATH_ENTROPY], \
                dict_paths[PATH_JDS_EXPLORED], \
                dict_paths[PATH_JDS_ENTIRE], \
                dict_paths[PATH_MEAN_FITNESS], \
                dict_paths[PATH_MAX_FITNESS], \
                dict_paths[PATH_NOVELTY], \
                dict_paths[PATH_ANGLE_COVERAGE] = MetricsHexapodCameraVertical.generate_figures(df,
                                                                                         df_l,
                                                                                   dict_path_variants_legend=dict_path_variants_legend,
                                                                                   folder_path_save=PATH_RESULTS_TO_SAVE_COMPARISON)
            elif issubclass(METRICS_CLASS, MetricsAirHockey):
                dict_paths[PATH_POS], \
                dict_paths[PATH_SIZE_POP], \
                dict_paths[PATH_L], \
                dict_paths[PATH_ENTROPY], \
                dict_paths[PATH_JDS_EXPLORED], \
                dict_paths[PATH_JDS_ENTIRE], \
                dict_paths[PATH_MEAN_FITNESS], \
                dict_paths[PATH_MAX_FITNESS], \
                dict_paths[PATH_NOVELTY], \
                dict_paths[PATH_DIVERSITY] = MetricsAirHockey.generate_figures(df,
                                                                                dict_path_variants_legend=dict_path_variants_legend,
                                                                                folder_path_save=PATH_RESULTS_TO_SAVE_COMPARISON)
            else:
                raise Exception("Class of Metrics Not Found")
        except Exception as e:
            error_traceback = traceback.format_exc()

        DICT_LEGEND_PATH_X_TO_UPLOAD_PATH = {LEGEND_PATH_X: os.path.join(UPLOAD, dict_paths[LEGEND_PATH_X])
              for LEGEND_PATH_X in LIST_LEGENDS_PATHS}
        info_metric_graphs_comparison = {'title': _name_comparison,
                                         'error_traceback': error_traceback}
        info_metric_graphs_comparison.update(DICT_LEGEND_PATH_X_TO_UPLOAD_PATH)
        list_metric_graphs.append(info_metric_graphs_comparison)

    return list_metric_graphs


def generate_report(list_formatted_gen_walls,
                    list_formatted_gen_maze,
                    list_formatted_gen_camera_vertical,
                    list_formatted_gen_air_hockey,

                    list_variants_walls,
                    list_variants_maze,
                    list_variants_hexapod_camera_vertical,
                    list_variants_air_hockey,

                    list_metric_graphs_walls,
                    list_metric_graphs_maze,
                    list_metric_graphs_hexapod_camera_vertical,
                    list_metric_graphs_air_hockey,

                    list_experiments_launched):

    path_folder_report_jinja2 = os.path.abspath(os.path.join(__file__, os.pardir))

    templateLoader = jinja2.FileSystemLoader(searchpath=path_folder_report_jinja2)
    templateEnv = jinja2.Environment(loader=templateLoader)
    template = templateEnv.get_template('report.jinja2')

    HASH_KEY = "hash"
    # Creating dict of lists based on single example
    from singularity.collections_experiments.maze import MAZE_AURORA_HARD_CODED_POS
    d = {
        key: [] for key in vars(MAZE_AURORA_HARD_CODED_POS).keys()
    }
    d[HASH_KEY] = []

    # Generating markdown table for exp launched only
    if list_experiments_launched:
        list_experiments_to_consider = list_experiments_launched
    else:
        list_experiments_to_consider = get_all_experiments_from_dict(DICT_EXPERIMENTS_TO_LAUNCH)

    for exp in list_experiments_to_consider:
        for key, value in vars(exp).items():
            d[key].append(value)
        d[HASH_KEY].append(hash(exp))

    df = pd.DataFrame(d)
    table_summary = df.to_markdown()

    report_str = template.render(table_summary=table_summary,
                                 list_metric_graphs_walls=list_metric_graphs_walls,
                                 list_metric_graphs_maze=list_metric_graphs_maze,
                                 list_metric_graphs_hexapod_camera_vertical=list_metric_graphs_hexapod_camera_vertical,
                                 list_metric_graphs_air_hockey=list_metric_graphs_air_hockey,
                                 list_variants_walls=list_variants_walls,
                                 list_variants_maze=list_variants_maze,
                                 list_variants_hexapod_camera_vertical=list_variants_hexapod_camera_vertical,
                                 list_variants_air_hockey=list_variants_air_hockey,
                                 list_gen_walls=list_formatted_gen_walls,
                                 list_gen_maze=list_formatted_gen_maze,
                                 list_gen_hexapod_camera_vertical=list_formatted_gen_camera_vertical,
                                 list_gen_air_hockey=list_formatted_gen_air_hockey,
                                 )

    print(report_str)

    # Saving it in the same folder as the template
    print(f"Saving repord.md there: {os.path.join(os.getcwd(), 'report.md')}")
    print("list_metric_graphs_maze", list_metric_graphs_maze)
    with open(os.path.join(os.getcwd(), "report.md"), 'w') as f:
        f.write(report_str)


def get_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--folder-dfs', type=str, default=None)
    parser.add_argument('--save-dfs-only', action="store_true")
    parser.add_argument('--data', type=str, default=None)
    parser.add_argument('--number-processes', type=int, default=1)
    parser.add_argument('--raise-exception', action="store_true")

    return parser.parse_known_args()


def launch_analysis(folder_dfs,
                    save_dfs_only,
                    number_processes,
                    path_main_folder,
                    list_experiments_launched,
                    raise_exception=False):

    LIST_GEN_WALLS = [5000, 15000]
    LIST_GEN_HEXAPOD_CAMERA_VERTICAL = [5000, 15000, 20000, 30000]
    LIST_GEN_MAZE = [5000, 10000, 15000, 20000]
    LIST_GEN_AIR_HOCKEY = [1000]

    list_formatted_gen_walls = get_formatted_list_gen(LIST_GEN_WALLS)
    list_formatted_gen_camera_vertical = get_formatted_list_gen(LIST_GEN_HEXAPOD_CAMERA_VERTICAL)
    list_formatted_gen_maze = get_formatted_list_gen(LIST_GEN_MAZE)
    list_formatted_gen_air_hockey = get_formatted_list_gen(LIST_GEN_AIR_HOCKEY)

    path_results_to_save = "results/"
    if not os.path.exists(path_results_to_save):
        os.mkdir(path_results_to_save)
        print("path_results_to_save", path_results_to_save)

    if folder_dfs is None: # generate dfs
        all_available_results_paths_walls = get_all_available_results_paths(path_main_folder, WALLS)
        all_available_results_paths_maze = get_all_available_results_paths(path_main_folder, MAZE)
        all_available_results_paths_hexapod_camera_vertical = get_all_available_results_paths(path_main_folder, HEXAPOD_CAMERA_VERTICAL)
        all_available_results_paths_air_hockey = get_all_available_results_paths(path_main_folder, AIR_HOCKEY)

        df_walls = MetricsWalls.get_dfs(all_available_results_paths_walls, number_processes)
        df_maze = MetricsMaze.get_dfs(all_available_results_paths_maze, number_processes)
        df_hexapod_camera_vertical, df_hexapod_camera_vertical_l = MetricsHexapodCameraVertical.get_dfs(all_available_results_paths_hexapod_camera_vertical, number_processes)
        df_air_hockey = MetricsAirHockey.get_dfs(all_available_results_paths_air_hockey, number_processes)

        # Making quick saving of df_maze and df_walls to CSV
        # Maze
        with open(os.path.join(path_results_to_save, "df_maze.csv"), "w") as f:
            f.write(df_maze.to_csv())

        # Walls
        with open(os.path.join(path_results_to_save, "df_walls.csv"), "w") as f:
            f.write(df_walls.to_csv())

        # Hexapod camera vertical
        with open(os.path.join(path_results_to_save, f"df_{HEXAPOD_CAMERA_VERTICAL}.csv"), "w") as f:
            f.write(df_hexapod_camera_vertical.to_csv())
        with open(os.path.join(path_results_to_save, f"df_{HEXAPOD_CAMERA_VERTICAL}_l.csv"), "w") as f:
            f.write(df_hexapod_camera_vertical_l.to_csv())

        # Air-Hockey
        with open(os.path.join(path_results_to_save, f"df_{AIR_HOCKEY}.csv"), "w") as f:
            f.write(df_air_hockey.to_csv())

    else: # folder_dfs is defined -> look for the dfs CSV files
        assert isinstance(folder_dfs, str)
        path_df_maze = os.path.join(folder_dfs, "df_maze.csv")
        path_df_walls = os.path.join(folder_dfs, "df_walls.csv")

        path_df_hexapod_camera_vertical = os.path.join(folder_dfs, f"df_{HEXAPOD_CAMERA_VERTICAL}.csv")
        path_df_hexapod_camera_vertical_l = os.path.join(folder_dfs, f"df_{HEXAPOD_CAMERA_VERTICAL}_l.csv")

        path_df_air_hockey = os.path.join(folder_dfs, f"df_{AIR_HOCKEY}.csv")

        assert os.path.exists(path_df_maze) or os.path.exists(path_df_walls) or os.path.exists(path_df_hexapod_camera_vertical)

        if os.path.exists(path_df_maze):
            df_maze = pd.read_csv(path_df_maze)
        else:
            df_maze = None

        if os.path.exists(path_df_walls):
            df_walls = pd.read_csv(path_df_walls)
        else:
            df_walls = None

        if os.path.exists(path_df_hexapod_camera_vertical):
            df_hexapod_camera_vertical = pd.read_csv(path_df_hexapod_camera_vertical)
        else:
            df_hexapod_camera_vertical = None

        if os.path.exists(path_df_hexapod_camera_vertical_l):
            df_hexapod_camera_vertical_l = pd.read_csv(path_df_hexapod_camera_vertical_l)
        else:
            df_hexapod_camera_vertical_l = None

        if os.path.exists(path_df_air_hockey):
            df_air_hockey = pd.read_csv(path_df_air_hockey)
        else:
            df_air_hockey = None

    if not save_dfs_only:
        if df_maze is not None:
            list_variants_maze = get_list_variants_data_maze(path_main_folder, path_results_to_save, LIST_GEN_MAZE, df_maze, list_experiments_launched)
            list_metric_graphs_maze = get_list_metric_graphs_per_factory(path_main_folder, path_results_to_save, df_maze, MAZE, list_experiments_launched)
        else:
            list_variants_maze = []
            list_metric_graphs_maze = []

        if df_walls is not None:
            list_variants_walls = get_list_variants_data_walls(path_main_folder, path_results_to_save, LIST_GEN_WALLS, df_walls, list_experiments_launched)
            list_metric_graphs_walls = get_list_metric_graphs_per_factory(path_main_folder, path_results_to_save, df_walls, WALLS, list_experiments_launched)
        else:
            list_variants_walls = []
            list_metric_graphs_walls = []

        if df_hexapod_camera_vertical is not None:
            list_variants_hexapod_camera_vertical = get_list_variants_data_hexapod_camera_vertical(path_main_folder,
                                                                                                   path_results_to_save,
                                                                                                   LIST_GEN_HEXAPOD_CAMERA_VERTICAL,
                                                                                                   df_hexapod_camera_vertical,
                                                                                                   df_hexapod_camera_vertical_l,
                                                                                                   list_experiments_launched,
                                                                                                   raise_exception)

            list_metric_graphs_hexapod_camera_vertical = get_list_metric_graphs_per_factory(path_main_folder,
                                                                                            path_results_to_save,
                                                                                            df_hexapod_camera_vertical,
                                                                                            HEXAPOD_CAMERA_VERTICAL,
                                                                                            list_experiments_launched,
                                                                                            df_l=df_hexapod_camera_vertical_l)
        else:
            list_variants_hexapod_camera_vertical = []
            list_metric_graphs_hexapod_camera_vertical = []

        if df_air_hockey is not None:
            list_variants_air_hockey = get_list_variants_data_air_hockey(path_main_folder,
                                                                         path_results_to_save,
                                                                         LIST_GEN_AIR_HOCKEY,
                                                                         df_air_hockey,
                                                                         list_experiments_launched,
                                                                         raise_exception)

            list_metric_graphs_air_hockey = get_list_metric_graphs_per_factory(path_main_folder,
                                                                               path_results_to_save,
                                                                               df_air_hockey,
                                                                               AIR_HOCKEY,
                                                                               list_experiments_launched)
        else:
            list_variants_air_hockey = []
            list_metric_graphs_air_hockey = []

        generate_report(list_formatted_gen_walls,
                        list_formatted_gen_maze,
                        list_formatted_gen_camera_vertical,
                        list_formatted_gen_air_hockey,
                        list_variants_walls,
                        list_variants_maze,
                        list_variants_hexapod_camera_vertical,
                        list_variants_air_hockey,
                        list_metric_graphs_walls,
                        list_metric_graphs_maze,
                        list_metric_graphs_hexapod_camera_vertical,
                        list_metric_graphs_air_hockey,
                        list_experiments_launched=list_experiments_launched
                        )


def main():
    print("Content of DICT_EXPERIMENTS_TO_LAUNCH:")
    for list_exp in DICT_EXPERIMENTS_TO_LAUNCH.values():
        for exp in list_exp:
            print(exp.get_exec_name())

    # Get main arguments
    args, others = get_args()
    folder_dfs = args.folder_dfs
    save_dfs_only = args.save_dfs_only
    number_processes = args.number_processes
    raise_exception = args.raise_exception

    assert not(args.data is not None and others) # In this case, we do not know which path to choose
    if args.data:
        path_main_folder = args.data
    elif others:
        path_main_folder = others[0]
    else:
        raise ValueError("Please provide a value for path_main_folder")
    print(f"PATH RESULTS FOLDER: {path_main_folder}")

    loaded_list_experiments = load_serialised_list_experiments(path_main_folder)

    launch_analysis(folder_dfs, save_dfs_only, number_processes, path_main_folder, loaded_list_experiments, raise_exception)


if __name__ == '__main__':
    main()
    # launch_analysis(folder_dfs="/Users/looka/git/sferes2/exp/aurora/analysis/data_frames/",
    #                 save_dfs_only=False,
    #                 number_processes=16,
    #                 path_main_folder="/Volumes/lg4615/ephemeral/final_aurora_2020-07-21_22_41_16/")

import os
import os.path

import cv2
import pandas as pd
import sys
from typing import List, Tuple
import itertools


import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from seaborn.relational import _LinePlotter
import imageio

sys.path.append('/git/sferes2/exp/aurora/submodules/figure_generator')
sys.path.append('/git/sferes2/exp/aurora/')

from singularity.experiment import Experiment

from analysis.metrics import metrics

from data_reader import get_data_proj, get_data_modifier, get_data_offspring


class MetricsMaze(metrics.Metrics):
    GEN = 'gen'
    LEGEND = "legend"

    LIST_NB_DIV_COVERAGE_POS = [10, 20, 30, 40, 50]
    COVERAGE_POS = "coverage_pos"
    DICT_COVERAGE_POS_STR = metrics.get_dict_coverage_pos_str(COVERAGE_POS, LIST_NB_DIV_COVERAGE_POS)

    SIZE_POP = 'size_pop'
    L = 'l'
    SUBDIR = "subdir"
    MAIN_FOLDER = 'main_folder'

    ENTROPY_NORMALISED = "entropy_normalised"
    JDS_UNIFORM_EXPLORED = "JDS_uniform_explored"
    JDS_UNIFORM_ENTIRE = "JDS_uniform_entire"

    MEAN_FITNESS = "mean_fitness"
    MAX_FITNESS = "max_fitness"

    NOVELTY = "novelty"

    LIST_GENS = 'list_gens'

    LIST_COVERAGE_POS = 'list_coverage_pos'
    DICT_LIST_COVERAGE_POS_STR = metrics.get_dict_coverage_pos_str(LIST_COVERAGE_POS, LIST_NB_DIV_COVERAGE_POS)

    LIST_SIZE_POP = 'list_size_pop'
    LIST_L = 'list_l'
    LIST_SUBDIR = 'list_subdir'
    LIST_MAIN_FOLDER = 'list_main_folder'

    LIST_ENTROPY_NORMALISED = "list_entropy_normalised"
    LIST_JDS_UNIFORM_EXPLORED = "list_JDS_uniform_explored"
    LIST_JDS_UNIFORM_ENTIRE = "list_JDS_uniform_entire"

    LIST_MEAN_FITNESS = "list_mean_fitness"
    LIST_MAX_FITNESS = "list_max_fitness"

    LIST_NOVELTY = "list_novelty"

    LIST_GENS_L = 'list_gens_l'
    LIST_MAIN_FOLDER_L = 'list_main_folder_l'
    LIST_SUBDIR_L = 'list_subdir_l'

    DICT_LIST_TO_NAME_ATTR = {
        LIST_GENS: GEN,
        LIST_GENS_L: GEN,
        **metrics.get_dict_list_to_attr_coverage_str(DICT_LIST_COVERAGE_POS_STR, DICT_COVERAGE_POS_STR),
        LIST_ENTROPY_NORMALISED: ENTROPY_NORMALISED,
        LIST_JDS_UNIFORM_EXPLORED: JDS_UNIFORM_EXPLORED,
        LIST_JDS_UNIFORM_ENTIRE: JDS_UNIFORM_ENTIRE,
        LIST_MEAN_FITNESS: MEAN_FITNESS,
        LIST_MAX_FITNESS: MAX_FITNESS,
        LIST_NOVELTY: NOVELTY,
        LIST_L: L,
        LIST_SIZE_POP: SIZE_POP,
        LIST_MAIN_FOLDER: MAIN_FOLDER,
        LIST_MAIN_FOLDER_L: MAIN_FOLDER,
        LIST_SUBDIR: SUBDIR,
        LIST_SUBDIR_L: SUBDIR,
    }

    NAME_LEGENDS_TO_SAVE = [
        *DICT_COVERAGE_POS_STR.values(),
        SIZE_POP,
        L,
        ENTROPY_NORMALISED,
        JDS_UNIFORM_EXPLORED,
        JDS_UNIFORM_ENTIRE,
        MEAN_FITNESS,
        MAX_FITNESS,
        NOVELTY,
    ]

    NAME_LEGENDS_REPORT = [
        f"{COVERAGE_POS}*",
        SIZE_POP,
        L,
        ENTROPY_NORMALISED,
        JDS_UNIFORM_EXPLORED,
        JDS_UNIFORM_ENTIRE,
        MEAN_FITNESS,
        MAX_FITNESS,
        NOVELTY,
    ]

    x_lim = (0., 600.)
    y_lim = (0., 600.)

    def __init__(self):
        super().__init__()

    @classmethod
    def get_number_generations(cls) -> int:
        return 15000

    @classmethod
    def _save_heatmap(cls,
                      gen: int,
                      path_folder_load: str,
                      list_nb_div: List[int],
                      path_folder_save: str,
                      do_save_scatter: bool,
                      extension: str="png"
                      ):
        _, array_gt_positions, array_fitness_values, array_novelty = get_data_proj(os.path.join(path_folder_load, f"proj_{gen}.dat"))
        array_gt_positions = array_gt_positions[:, 0:2]
        start_lim = np.asarray([0, 0])
        stop_lim = np.asarray([600, 600])

        for nb_div in list_nb_div:
            temp = np.floor(nb_div * (array_gt_positions - start_lim) / (stop_lim - start_lim))
            bins_array = start_lim + temp * (stop_lim - start_lim) / nb_div

            unique_bins_pos, count_per_bin_pos = np.unique(bins_array, axis=0, return_counts=True)
            count_per_bin_pos = np.reshape(count_per_bin_pos, unique_bins_pos[:, 1].shape)

            list_unique_bins_pos = unique_bins_pos.tolist()
            list_missing_bins = []

            for i in range(0, 600, int(600 / nb_div)):
                for j in range(0, 600, int(600 / nb_div)):
                    if [i, j] not in list_unique_bins_pos:
                        list_missing_bins.append([i, j])

            array_missing_bins = np.asarray(list_missing_bins)
            if not list_missing_bins:
                array_missing_bins = array_missing_bins.reshape((-1, 2))
            count_for_missing_bins = np.zeros_like(array_missing_bins[:, 0])
            unique_bins_pos = np.vstack((unique_bins_pos, array_missing_bins))
            unique_bins_pos[:, 1] = 600 - unique_bins_pos[:, 1]

            count_per_bin_pos = np.vstack((count_per_bin_pos.reshape(-1,1), count_for_missing_bins.reshape(-1,1)))


            df = pd.DataFrame({
                "x": unique_bins_pos[:, 1],
                "y": unique_bins_pos[:, 0],
                "count": np.reshape(count_per_bin_pos, unique_bins_pos[:, 1].shape)
            })
            df = df.pivot("x", "y", "count")
            df = df.reset_index()
            ax = sns.heatmap(df, vmax=50, cmap="YlGnBu",mask=(df == 0),xticklabels="", yticklabels="")
            ax.set_xlim(0, nb_div)
            ax.set_ylim(0, nb_div)
            plt.title(f"nb_div {nb_div} - Gen {gen:07}")
            plt.savefig(os.path.join(path_folder_save, f"heatmap_{nb_div}_{gen:07}.{extension}"))
            plt.close()

        if do_save_scatter:
            fig = plt.figure()
            if np.all(array_fitness_values.flatten() == array_fitness_values.flatten()[0]):
                scatter_plot = plt.scatter(array_gt_positions[:, 0].flatten(), 600 - array_gt_positions[:, 1].flatten(), marker='o', s=1)
            else:
                cm = plt.cm.get_cmap('RdYlBu')
                scatter_plot = plt.scatter(array_gt_positions[:, 0].flatten(), 600 - array_gt_positions[:, 1].flatten(), c=array_fitness_values.flatten(), marker='o', s=1, cmap=cm)
                plt.colorbar(scatter_plot)
            try:
                dir_path = os.path.dirname(os.path.abspath(__file__))

                pbm_path = os.path.join(dir_path, os.pardir, "resources", "maze_hard.pbm")
                plt.imshow(cv2.resize(cv2.flip(cv2.imread(pbm_path), 0), (600, 600)))
            except:
                pass
            plt.xlim(0, 600)
            plt.ylim(0, 600)
            plt.title(f"Gen {gen:07}")
            fig.savefig(os.path.join(path_folder_save, f"scatter_pos_{gen:07}.{extension}"))
            plt.close()

    @classmethod
    def save_gif_coverage(cls, path_folder_load, path_folder_save, list_nb_div, do_save_scatter=False):
        list_valid_gens = []
        for gen in range(cls.get_max_number_generations() + 1):
            try:
                cls._save_heatmap(gen, path_folder_load, list_nb_div, path_folder_save, do_save_scatter=do_save_scatter, extension="png")
                list_valid_gens.append(gen)
            except FileNotFoundError:
                pass

        if list_valid_gens:
            last_gen = list_valid_gens[-1]
            cls._save_heatmap(last_gen, path_folder_load, list_nb_div, path_folder_save, do_save_scatter=do_save_scatter, extension="png")
            cls._save_heatmap(last_gen, path_folder_load, list_nb_div, path_folder_save, do_save_scatter=False, extension="eps")

        for nb_div in list_nb_div:
            images_heatmap = []
            for gen in list_valid_gens:
                file_path_heatmap = os.path.join(path_folder_save, f"heatmap_{nb_div}_{gen:07}.png")
                images_heatmap.append(imageio.imread(file_path_heatmap))
                os.remove(file_path_heatmap)
            imageio.mimsave(os.path.join(path_folder_save, f"heatmap_{nb_div}.gif"), images_heatmap, fps=2)

        if do_save_scatter:
            images_scatter_positions = []
            for gen in list_valid_gens:
                file_path_scatter_positions = os.path.join(path_folder_save, f"scatter_pos_{gen:07}.png")
                images_scatter_positions.append(imageio.imread(file_path_scatter_positions))
                os.remove(file_path_scatter_positions)
            imageio.mimsave(os.path.join(path_folder_save, f"scatter_pos.gif"), images_scatter_positions, fps=2)

    @classmethod
    def save_gif_offspring_analysis(cls, path_folder_load, path_folder_save, do_save_scatter=False):
        list_valid_gens = []
        for gen in range(cls.get_max_number_generations() + 1):
            try:
                _, array_gt_positions, array_fitness_values, array_novelty = get_data_proj(os.path.join(path_folder_load, f"proj_{gen}.dat"))
                _, array_gt_positions_offspring = get_data_offspring(os.path.join(path_folder_load, f"offspring_{gen}.dat"))
                array_gt_positions = array_gt_positions[:, 0:2]
                array_gt_positions_offspring = array_gt_positions_offspring[:, 0:2]

                if do_save_scatter:
                    fig = plt.figure()
                    plt.scatter(array_gt_positions[:, 0].flatten(), 600 - array_gt_positions[:, 1].flatten(), marker='o', c='blue', s=1)
                    plt.scatter(array_gt_positions_offspring[:, 0].flatten(), 600 - array_gt_positions_offspring[:, 1].flatten(), marker='o', c='red', s=1)
                    plt.scatter(array_gt_positions_offspring[-5:, 0].flatten(), 600 - array_gt_positions_offspring[-5:, 1].flatten(), marker='o', c='green', s=2)
                    try:
                        dir_path = os.path.dirname(os.path.abspath(__file__))

                        pbm_path = os.path.join(dir_path, os.pardir, "resources", "maze_hard.pbm")
                        plt.imshow(cv2.resize(cv2.flip(cv2.imread(pbm_path), 0), (600, 600)))
                    except:
                        pass
                    plt.xlim(0, 600)
                    plt.ylim(0, 600)
                    plt.title(f"Gen {gen:07}")
                    fig.savefig(os.path.join(path_folder_save, f"scatter_pos_offspring_{gen:07}.png"))
                    plt.close()

                list_valid_gens.append(gen)
            except FileNotFoundError:
                pass

        if do_save_scatter:
            images_scatter_positions = []
            for gen in list_valid_gens:
                file_path_scatter_positions = os.path.join(path_folder_save, f"scatter_pos_offspring_{gen:07}.png")
                images_scatter_positions.append(imageio.imread(file_path_scatter_positions))
                os.remove(file_path_scatter_positions)
            if images_scatter_positions:
                imageio.mimsave(os.path.join(path_folder_save, f"scatter_pos_offspring.gif"), images_scatter_positions, fps=2)

    @classmethod
    def calculate_coverage_position(cls, array_gt_positions, nb_div=40):

        start_lim = np.asarray([0, 0])
        stop_lim = np.asarray([600, 600])

        temp = np.floor(nb_div * (array_gt_positions - start_lim) / (stop_lim - start_lim))
        bins_array = start_lim + temp * (stop_lim - start_lim) / nb_div

        unique_bins_pos, count_per_bin_pos = np.unique(bins_array, axis=0, return_counts=True)
        count_per_bin_pos = np.reshape(count_per_bin_pos, unique_bins_pos[:, 1].shape)

        np.size(np.unique(bins_array, axis=0), axis=0) / (nb_div ** len(start_lim))

        return cls._calculate_coverage(array_gt_positions,
                                       nb_div,
                                       cls.x_lim, cls.y_lim)

    @classmethod
    def calculate_JDS_distance_with_uniform_across_same_space(cls, array_gt_positions, nb_div=40):
        return cls.JSD_dist_uniform_across_explored_space(array_gt_positions, nb_div, cls.x_lim, cls.y_lim)

    @classmethod
    def calculate_JDS_distance_with_uniform_across_entire_space(cls, array_gt_positions, nb_div=40):
        return cls.JSD_dist_uniform_across_entire_space(array_gt_positions, nb_div, cls.x_lim, cls.y_lim)

    @classmethod
    def calculate_entropy_normalised(cls, array_gt_positions, nb_div=40):
        return cls._calculate_entropy_normalised(array_gt_positions, nb_div, cls.x_lim, cls.y_lim)

    @classmethod
    def calculate_coverages(cls, path_folder):
        list_gens = []
        dict_list_coverage_pos = {
            nb_div: []
            for nb_div in cls.LIST_NB_DIV_COVERAGE_POS
        }
        list_entropy_normalised_10 = []
        list_JDS_with_uniform_across_explored_space_10 = []
        list_JDS_with_uniform_across_entire_space_10 = []
        list_mean_fitness = []
        list_max_fitness = []
        list_mean_novelty = []

        for gen in range(cls.get_max_number_generations() + 1):
            try:
                _, ground_truth_array, array_fitness_values, array_novelty = get_data_proj(os.path.join(path_folder, f"proj_{gen}.dat"))
                list_gens.append(gen)
                for nb_div in cls.LIST_NB_DIV_COVERAGE_POS:
                    dict_list_coverage_pos[nb_div].append(cls.calculate_coverage_position(ground_truth_array[:, 0:2], nb_div=nb_div)) # Considering only x,y coverage

                list_entropy_normalised_10.append(cls.calculate_entropy_normalised(ground_truth_array[:, 0:2], nb_div=10))
                list_JDS_with_uniform_across_explored_space_10.append(
                    cls.calculate_JDS_distance_with_uniform_across_same_space(ground_truth_array[:, 0:2], nb_div=10))
                list_JDS_with_uniform_across_entire_space_10.append(
                    cls.calculate_JDS_distance_with_uniform_across_entire_space(ground_truth_array[:, 0:2], nb_div=10))

                _, _, mean_fitness, best_fitness = cls.get_data_fit(ground_truth_array[:, 0:2], array_fitness_values, cls.x_lim, cls.y_lim, nb_div=40)

                list_mean_fitness.append(mean_fitness)
                list_max_fitness.append(best_fitness)

                list_mean_novelty.append(np.mean(array_novelty[np.isfinite(array_novelty)]))

            except FileNotFoundError:
                pass

        return list_gens, \
               dict_list_coverage_pos, \
               list_entropy_normalised_10, \
               list_JDS_with_uniform_across_explored_space_10, \
               list_JDS_with_uniform_across_entire_space_10, \
               list_mean_fitness, \
               list_max_fitness, \
               list_mean_novelty

    @classmethod
    def generate_figures(cls,
                         df,
                         dict_path_variants_legend,
                         folder_path_save,
                         prefix=None):
        if not prefix:
            prefix = ''
        else:
            prefix = f'{prefix}_'

        tmp_list_path_variants, list_legends = zip(*dict_path_variants_legend.items())
        list_path_variants = [os.path.basename(x)
                              for x in tmp_list_path_variants]

        df_legend = pd.DataFrame({
            cls.MAIN_FOLDER: list_path_variants,
            cls.LEGEND: list_legends,
        })

        df = pd.merge(df, df_legend, on=cls.MAIN_FOLDER)


        for NAME_LEGEND in cls.NAME_LEGENDS_TO_SAVE:
            cls.save_figure(folder_path_save, df, NAME_LEGEND, f'{prefix}{NAME_LEGEND}.png')

        return (os.path.join(folder_path_save, f'{prefix}{NAME_LEGEND}.png')
                for NAME_LEGEND in cls.NAME_LEGENDS_REPORT)

    @classmethod
    def generate_figures_per_variant(cls,
                                     df,
                                     variant_experiment: Experiment,
                                     path_variant_results_to_save):
        main_folder_results_variant_experiment = f"results_{variant_experiment.get_exec_name()}"

        df = df[df[cls.MAIN_FOLDER] == main_folder_results_variant_experiment]

        for NAME_LEGEND in cls.NAME_LEGENDS_TO_SAVE:
            cls.save_figure_one_variant(path_variant_results_to_save, df, NAME_LEGEND, f'{NAME_LEGEND}.png')

        return (os.path.join(path_variant_results_to_save, f'{NAME_LEGEND}.png')
                for NAME_LEGEND in cls.NAME_LEGENDS_REPORT)

    @classmethod
    def calculate_list_size_pop(cls, path_folder):
        list_gens = []
        list_size_pop = []

        for gen in range(cls.get_max_number_generations() + 1):
            try:
                _, ground_truth_array, array_fitness_values, array_novelty = get_data_proj(os.path.join(path_folder, f"proj_{gen}.dat"))
                list_gens.append(gen)
                list_size_pop.append(np.size(ground_truth_array, axis=0))
            except FileNotFoundError:
                pass

        return list_gens, list_size_pop

    @classmethod
    def calculate_list_l(cls, path_folder) -> Tuple[list, list, list]:
        try:
            array_gen, array_l, array_pop_size = get_data_modifier(os.path.join(path_folder, f"stat_modifier.dat"))
            return list(array_gen), list(array_l), list(array_pop_size)
        except FileNotFoundError:
            return [], [], []

    @classmethod
    def get_results_folder_path_load(cls, tuple_subdir):
        folder_path_load, subdir = tuple_subdir
        name_main_folder = os.path.basename(folder_path_load)
        subdir_path = os.path.join(folder_path_load, subdir)

        print(subdir_path)
        _list_gens, \
        _dict_list_coverage_pos, \
        _list_entropy_normalised, \
        _list_JDS_with_uniform_across_explored_space, \
        _list_JDS_with_uniform_across_entire_space, \
        _list_mean_fitness, \
        _list_max_fitness, \
        _list_mean_novelty = cls.calculate_coverages(subdir_path)

        _list_gens_l, _list_l, _list_size_pop = cls.calculate_list_l(subdir_path)
        return {
            cls.LIST_GENS: _list_gens,
            **{cls.DICT_LIST_COVERAGE_POS_STR[nb_div]: _dict_list_coverage_pos[nb_div]
               for nb_div in _dict_list_coverage_pos},
            cls.LIST_ENTROPY_NORMALISED: _list_entropy_normalised,
            cls.LIST_JDS_UNIFORM_EXPLORED: _list_JDS_with_uniform_across_explored_space,
            cls.LIST_JDS_UNIFORM_ENTIRE: _list_JDS_with_uniform_across_entire_space,
            cls.LIST_MEAN_FITNESS: _list_mean_fitness,
            cls.LIST_MAX_FITNESS: _list_max_fitness,
            cls.LIST_NOVELTY: _list_mean_novelty,
            cls.LIST_GENS_L: _list_gens_l,
            cls.LIST_L: _list_l,
            cls.LIST_SIZE_POP: _list_size_pop,
            cls.LIST_MAIN_FOLDER: [name_main_folder] * len(_list_gens),
            cls.LIST_SUBDIR: [subdir] * len(_list_gens),
            cls.LIST_MAIN_FOLDER_L: [name_main_folder] * len(_list_gens_l),
            cls.LIST_SUBDIR_L: [subdir] * len(_list_gens_l),
        }

    @classmethod
    def get_dfs(cls,
                list_paths_results: List[str],
                number_processes: int=1,
                ):

        list_tuple_load_subdir = [
            (folder_path_load, subdir)
            for folder_path_load in list_paths_results
            for subdir in cls.get_list_subdirs(folder_path_load)
        ]

        from multiprocessing import Pool
        from operator import itemgetter
        import itertools
        with Pool(processes=number_processes) as pool:
            results = pool.map(cls.get_results_folder_path_load, list_tuple_load_subdir)


        df = pd.DataFrame({
            cls.DICT_LIST_TO_NAME_ATTR[NAME_ATTR]: list(itertools.chain.from_iterable(map(itemgetter(NAME_ATTR), results)))
            for NAME_ATTR in [
                cls.LIST_GENS,
                *cls.DICT_LIST_COVERAGE_POS_STR.values(),
                cls.LIST_MAIN_FOLDER,
                cls.LIST_SUBDIR,
                cls.LIST_ENTROPY_NORMALISED,
                cls.LIST_JDS_UNIFORM_EXPLORED,
                cls.LIST_JDS_UNIFORM_ENTIRE,
                cls.LIST_MEAN_FITNESS,
                cls.LIST_MAX_FITNESS,
                cls.LIST_NOVELTY,
            ]
        })

        df_l = pd.DataFrame({
            cls.DICT_LIST_TO_NAME_ATTR[NAME_ATTR]: list(itertools.chain.from_iterable(map(itemgetter(NAME_ATTR), results)))
            for NAME_ATTR in [
                cls.LIST_GENS_L,
                cls.LIST_L,
                cls.LIST_SIZE_POP,
                cls.LIST_MAIN_FOLDER_L,
                cls.LIST_SUBDIR_L,
            ]
        })

        print(pd.merge(df, df_l, on=[cls.GEN, cls.MAIN_FOLDER, cls.SUBDIR], how='outer'))

        return pd.merge(df, df_l, on=[cls.GEN, cls.MAIN_FOLDER, cls.SUBDIR], how='outer')

if __name__ == '__main__':
    res = MetricsMaze.JSD_dist_uniform_across_entire_space(np.asarray([[1 , 1]]), 50, MetricsMaze.x_lim, MetricsMaze.y_lim)
    print(res)
    # MetricsWalls.generate_figures(dict_path_variants_legend={"/Users/looka/git/sferes2/exp/aurora/submodules/figure_generator/results/results_aurora_graphic_environment-hexa_walls_algorithm-novelty_search_latent_space_size-6_use_colors/": "test"},
    #                               folder_path_save='bouh/')
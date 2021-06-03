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

from analysis.metrics.metrics import Metrics

from data_reader import get_data_proj, get_data_modifier, get_data_offspring

class MetricsWalls(Metrics):
    GEN = 'gen'
    LEGEND = "legend"
    x_lim = (-1., 1.)
    y_lim = (-1., 1.)
    z_lim = (0., 0.4)
    pitch_lim = (-np.pi, np.pi)
    roll_lim = (-np.pi, np.pi)
    yaw_lim = (-np.pi, np.pi)

    COVERAGE_POS = "coverage_pos"
    COVERAGE_ROT = "coverage_rot"
    COVERAGE_GT = "coverage_gt"
    SIZE_POP = 'size_pop'
    MUTUAL_INFORMATION = 'mutual_information'
    L = 'l'
    SUBDIR = "subdir"
    MAIN_FOLDER = 'main_folder'

    LIST_GENS = 'list_gens'
    LIST_COVERAGE_POS = 'list_coverage_pos'
    LIST_COVERAGE_ROT = 'list_coverage_rot'
    LIST_COVERAGE_GT = 'list_coverage_gt'
    LIST_MUTUAL_INFORMATION = 'list_mutual_information'
    LIST_GENS_L = 'list_gens_l'
    LIST_L = 'list_l'
    LIST_SIZE_POP = 'list_size_pop'
    LIST_MAIN_FOLDER = 'list_main_folder'
    LIST_SUBDIR = 'list_subdir'
    LIST_MAIN_FOLDER_L = 'list_main_folder_l'
    LIST_SUBDIR_L = 'list_subdir_l'

    DICT_LIST_TO_NAME_ATTR = {
        LIST_GENS: GEN,
        LIST_GENS_L: GEN,
        LIST_COVERAGE_POS: COVERAGE_POS,
        LIST_COVERAGE_ROT: COVERAGE_ROT,
        LIST_COVERAGE_GT: COVERAGE_GT,
        LIST_MUTUAL_INFORMATION: MUTUAL_INFORMATION,
        LIST_L: L,
        LIST_SIZE_POP: SIZE_POP,
        LIST_MAIN_FOLDER: MAIN_FOLDER,
        LIST_MAIN_FOLDER_L: MAIN_FOLDER,
        LIST_SUBDIR: SUBDIR,
        LIST_SUBDIR_L: SUBDIR,
    }

    def __init__(self):
        super().__init__()

    @classmethod
    def get_number_generations(cls) -> int:
        return 15000

    @classmethod
    def calculate_coverage_position(cls, array_gt_positions, nb_div=10):
        return cls._calculate_coverage(array_gt_positions,
                                       nb_div,
                                       cls.x_lim, cls.y_lim, cls.z_lim)

    @classmethod
    def calculate_coverage_orientation(cls, array_gt_orientation, nb_div=10):
        return cls._calculate_coverage(array_gt_orientation,
                                       nb_div,
                                       cls.pitch_lim, cls.roll_lim, cls.yaw_lim)

    @classmethod
    def calculate_coverage_gt(cls, array_gt, nb_div=10):
        return cls._calculate_coverage(array_gt,
                                       nb_div,
                                       cls.x_lim, cls.y_lim, cls.z_lim,
                                       cls.pitch_lim, cls.roll_lim, cls.yaw_lim)

    @classmethod
    def calculate_mutual_information_position_orientation(cls, array_gt, nb_div=10):
        n_pop = np.size(array_gt, axis=0)

        lim_borders = [cls.x_lim, cls.y_lim, cls.z_lim, cls.pitch_lim, cls.roll_lim, cls.yaw_lim]
        start_lim = np.asarray(list(zip(*lim_borders))[0])
        stop_lim = np.asarray(list(zip(*lim_borders))[1])

        temp = np.floor(nb_div * (array_gt - start_lim) / (stop_lim - start_lim))
        bins_array = start_lim + temp * (stop_lim - start_lim) / nb_div

        unique_bins, count_per_bin = np.unique(bins_array, axis=0, return_counts=True)

        dict_bins_gt = dict(zip(map(tuple, unique_bins), count_per_bin))

        unique_bins_pos, count_per_bin_pos = np.unique(bins_array[:, 0:3], axis=0, return_counts=True)
        unique_bins_rot, count_per_bin_rot = np.unique(bins_array[:, 3:6], axis=0, return_counts=True)

        dict_bins_pos = dict(zip(map(tuple, unique_bins_pos), count_per_bin_pos))
        dict_bins_rot = dict(zip(map(tuple, unique_bins_rot), count_per_bin_rot))

        mutual_information = 0
        for tuple_bin_gt, count_bin_gt in dict_bins_gt.items():
            bin_pos = tuple_bin_gt[0:3]
            bin_rot = tuple_bin_gt[3:6]
            count_bin_pos = dict_bins_pos[bin_pos]
            count_bin_rot = dict_bins_rot[bin_rot]
            mutual_information += count_bin_gt * np.log(count_bin_gt * n_pop / (count_bin_pos * count_bin_rot))

        mutual_information /= n_pop

        return mutual_information

    @classmethod
    def calculate_coverages(cls, path_folder):
        list_gens = []
        list_coverage_pos = []
        list_coverage_rot = []
        list_coverage_gt = []
        list_mutual_information = []

        for gen in range(20001):
            try:
                _, ground_truth_array, array_fitness_values, array_novelty = get_data_proj(os.path.join(path_folder, f"proj_{gen}.dat"))
                list_gens.append(gen)
                list_coverage_pos.append(cls.calculate_coverage_position(ground_truth_array[:, 0:3]))
                list_coverage_rot.append(cls.calculate_coverage_orientation(ground_truth_array[:, 3:6]))
                list_coverage_gt.append(cls.calculate_coverage_gt(ground_truth_array[:, 0:6]))
                list_mutual_information.append(cls.calculate_mutual_information_position_orientation(ground_truth_array[:, 0:6]))
            except FileNotFoundError:
                pass

        return list_gens, list_coverage_pos, list_coverage_rot, list_coverage_gt, list_mutual_information

    @classmethod
    def calculate_list_size_pop(cls, path_folder):
        list_gens = []
        list_size_pop = []

        for gen in range(20001):
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

        _list_gens, _list_coverage_pos, _list_coverage_rot, _list_coverage_gt, _list_mutual_information = cls.calculate_coverages(subdir_path)
        _list_gens_l, _list_l, _list_size_pop = cls.calculate_list_l(subdir_path)

        return {
           cls.LIST_GENS: _list_gens,
           cls.LIST_COVERAGE_POS: _list_coverage_pos,
           cls.LIST_COVERAGE_ROT: _list_coverage_rot,
           cls.LIST_COVERAGE_GT: _list_coverage_gt,
           cls.LIST_MUTUAL_INFORMATION: _list_mutual_information,
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
        with Pool(processes=number_processes) as p:
            results = p.map(cls.get_results_folder_path_load, list_tuple_load_subdir)

        df = pd.DataFrame({
            cls.DICT_LIST_TO_NAME_ATTR[NAME_ATTR]: list(itertools.chain.from_iterable(map(itemgetter(NAME_ATTR), results)))
            for NAME_ATTR in [
                cls.LIST_GENS,
                cls.LIST_COVERAGE_POS,
                cls.LIST_COVERAGE_ROT,
                cls.LIST_COVERAGE_GT,
                cls.LIST_MUTUAL_INFORMATION,
                cls.LIST_MAIN_FOLDER,
                cls.LIST_SUBDIR,
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

        cls.save_figure(folder_path_save, df, cls.COVERAGE_POS, f'{prefix}{cls.COVERAGE_POS}.png')
        cls.save_figure(folder_path_save, df, cls.COVERAGE_ROT, f'{prefix}{cls.COVERAGE_ROT}.png')
        cls.save_figure(folder_path_save, df, cls.COVERAGE_GT, f'{prefix}{cls.COVERAGE_GT}.png')
        cls.save_figure(folder_path_save, df, cls.SIZE_POP, f'{prefix}{cls.SIZE_POP}.png')
        cls.save_figure(folder_path_save, df, cls.MUTUAL_INFORMATION, f'{prefix}{cls.MUTUAL_INFORMATION}.png')
        cls.save_figure(folder_path_save, df, cls.L, f'{prefix}{cls.L}.png')

        return os.path.join(folder_path_save, f'{prefix}{cls.COVERAGE_POS}.png'), \
               os.path.join(folder_path_save, f'{prefix}{cls.COVERAGE_ROT}.png'), \
               os.path.join(folder_path_save, f'{prefix}{cls.COVERAGE_GT}.png'), \
               os.path.join(folder_path_save, f'{prefix}{cls.SIZE_POP}.png'), \
               os.path.join(folder_path_save, f'{prefix}{cls.MUTUAL_INFORMATION}.png'), \
               os.path.join(folder_path_save, f'{prefix}{cls.L}.png')

    @classmethod
    def generate_figures_per_variant(cls,
                                     df,
                                     variant_experiment: Experiment,
                                     path_variant_results_to_save):
        main_folder_results_variant_experiment = f"results_{variant_experiment.get_exec_name()}"

        df = df[df[cls.MAIN_FOLDER] == main_folder_results_variant_experiment]

        cls.save_figure_one_variant(path_variant_results_to_save, df, cls.COVERAGE_POS, f'{cls.COVERAGE_POS}.png')
        cls.save_figure_one_variant(path_variant_results_to_save, df, cls.COVERAGE_ROT, f'{cls.COVERAGE_ROT}.png')
        cls.save_figure_one_variant(path_variant_results_to_save, df, cls.COVERAGE_GT, f'{cls.COVERAGE_GT}.png')
        cls.save_figure_one_variant(path_variant_results_to_save, df, cls.SIZE_POP, f'{cls.SIZE_POP}.png')
        cls.save_figure_one_variant(path_variant_results_to_save, df, cls.MUTUAL_INFORMATION, f'{cls.MUTUAL_INFORMATION}.png')
        cls.save_figure_one_variant(path_variant_results_to_save, df, cls.L, f'{cls.L}.png')

        return os.path.join(path_variant_results_to_save, f'{cls.COVERAGE_POS}.png'), \
               os.path.join(path_variant_results_to_save, f'{cls.COVERAGE_ROT}.png'), \
               os.path.join(path_variant_results_to_save, f'{cls.COVERAGE_GT}.png'), \
               os.path.join(path_variant_results_to_save, f'{cls.SIZE_POP}.png'), \
               os.path.join(path_variant_results_to_save, f'{cls.MUTUAL_INFORMATION}.png'), \
               os.path.join(path_variant_results_to_save, f'{cls.L}.png')
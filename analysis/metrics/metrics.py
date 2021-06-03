import abc
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

from data_reader import get_data_proj, get_data_modifier, get_data_offspring


def get_dict_coverage_pos_str(name_coverage, list_nb_div_coverage_pos):
    return {
        NB_DIV: f"{name_coverage}_{NB_DIV}"
        for NB_DIV in list_nb_div_coverage_pos
    }


def get_dict_list_to_attr_coverage_str(dict_list_coverage_pos_str,
                                       dict_coverage_pos_str):
    return {
        dict_list_coverage_pos_str[nb_div]: dict_coverage_pos_str[nb_div]
        for nb_div in dict_list_coverage_pos_str
    }


class Metrics(metaclass=abc.ABCMeta):
    LEGEND = "legend"
    GEN = "gen"
    SUBDIR = "subdir"
    MAIN_FOLDER = "main_folder"

    def __init__(self):
        pass



    @classmethod
    @abc.abstractmethod
    def get_number_generations(cls) -> int:
        raise NotImplementedError

    @classmethod
    def get_max_number_generations(cls) -> int:
        return 30000

    @classmethod
    def _calculate_coverage(cls, array_gt_positions, nb_div, *lim_borders):
        start_lim = np.asarray(list(zip(*lim_borders))[0])
        stop_lim = np.asarray(list(zip(*lim_borders))[1])

        temp = np.floor(nb_div * (array_gt_positions - start_lim) / (stop_lim - start_lim))
        bins_array = start_lim + temp * (stop_lim - start_lim) / nb_div
        return np.size(np.unique(bins_array, axis=0), axis=0) / (nb_div ** len(start_lim))

    @classmethod
    def x_log_x(cls, x):
        if x > 0.:
            return x * np.log(x)
        else:
            return 0

    @classmethod
    def _calculate_entropy_normalised(cls, array_data, nb_div, *lim_borders):
        n_pop = np.size(array_data, axis=0)
        n_dim = len(lim_borders)

        start_lim = np.asarray(list(zip(*lim_borders))[0])
        stop_lim = np.asarray(list(zip(*lim_borders))[1])

        temp = np.floor(nb_div * (array_data - start_lim) / (stop_lim - start_lim))
        bins_array = start_lim + temp * (stop_lim - start_lim) / nb_div

        unique_bins, count_per_bin = np.unique(bins_array, axis=0, return_counts=True)

        dict_bins_gt = dict(zip(map(tuple, unique_bins), count_per_bin))

        entropy = 0
        entropy_uniform = 0
        proba_bin_uniform = 1. / (nb_div ** n_dim)

        for tuple_bin_gt, count_bin_gt in dict_bins_gt.items():
            estimated_proba_bin = count_bin_gt / n_pop
            entropy += cls.x_log_x(estimated_proba_bin)

        entropy *= -1
        entropy_uniform = -1 * np.log(proba_bin_uniform)

        return entropy / entropy_uniform

    @classmethod
    def _get_uniform_dataset(cls, nb_div, *lim_borders):
        list_linspaces = [np.linspace(start, stop, num=nb_div, endpoint=False) + ((stop - start) / nb_div) / 2
                          for (start, stop) in lim_borders]
        return np.vstack(tuple(itertools.product(*list_linspaces)))

    @classmethod
    def JSD_dist(cls, array_data_1, array_data_2, nb_div, *lim_borders):
        def _kl(p, q):
            assert ((q > 0) and (p >= 0))
            if p == 0.:
                return 0.
            else:
                return p * np.log(p / q)

        n_pop_1 = np.size(array_data_1, axis=0)
        n_pop_2 = np.size(array_data_2, axis=0)

        start_lim = np.asarray(list(zip(*lim_borders))[0])
        stop_lim = np.asarray(list(zip(*lim_borders))[1])

        temp_1 = np.floor(nb_div * (array_data_1 - start_lim) / (stop_lim - start_lim))
        temp_2 = np.floor(nb_div * (array_data_2 - start_lim) / (stop_lim - start_lim))

        bins_array_1 = start_lim + temp_1 * (stop_lim - start_lim) / nb_div
        bins_array_2 = start_lim + temp_2 * (stop_lim - start_lim) / nb_div

        unique_bins_1, count_per_bin_1 = np.unique(bins_array_1, axis=0, return_counts=True)
        proba_per_bin_1 = count_per_bin_1 / n_pop_1
        unique_bins_2, count_per_bin_2 = np.unique(bins_array_2, axis=0, return_counts=True)
        proba_per_bin_2 = count_per_bin_2 / n_pop_2

        dict_proba_per_bin_1 = dict(zip(map(tuple, unique_bins_1), proba_per_bin_1))
        dict_proba_per_bin_2 = dict(zip(map(tuple, unique_bins_2), proba_per_bin_2))

        # Completing Dictionaries with missing bins:
        # So that both dictionaries share the same keys
        for tuple_bin_gt_1, _ in dict_proba_per_bin_1.items():
            if tuple_bin_gt_1 not in dict_proba_per_bin_2:
                dict_proba_per_bin_2[tuple_bin_gt_1] = 0.

        for tuple_bin_gt_2, _ in dict_proba_per_bin_2.items():
            if tuple_bin_gt_2 not in dict_proba_per_bin_1:
                dict_proba_per_bin_1[tuple_bin_gt_2] = 0.

        # Getting avg proba dict
        dict_proba_average = {}
        for tuple_bin_gt in dict_proba_per_bin_1.keys(): # They share the same keys with the other dict
            dict_proba_average[tuple_bin_gt] = (dict_proba_per_bin_1[tuple_bin_gt]
                                                + dict_proba_per_bin_2[tuple_bin_gt]) / 2.


        res_jds_dist_temp = 0.
        for tuple_bin_gt in dict_proba_per_bin_1.keys():
            proba_bin_1 = dict_proba_per_bin_1[tuple_bin_gt]
            proba_bin_2 = dict_proba_per_bin_2[tuple_bin_gt]
            proba_bin_avg = dict_proba_average[tuple_bin_gt]

            res_jds_dist_temp += _kl(proba_bin_1, proba_bin_avg) + _kl(proba_bin_2, proba_bin_avg)

        res_jds_dist = np.sqrt(res_jds_dist_temp / 2)

        return res_jds_dist

    @classmethod
    def JSD_dist_uniform_across_entire_space(cls, array_data, nb_div, *lim_borders):
        return cls.JSD_dist(array_data, cls._get_uniform_dataset(nb_div, *lim_borders), nb_div, *lim_borders)

    @classmethod
    def JSD_dist_uniform_across_explored_space(cls, array_data, nb_div, *lim_borders):
        def _kl(p, q):
            assert ((q > 0) and (p >= 0))
            if p == 0.:
                return 0.
            else:
                return p * np.log(p / q)

        n_pop_1 = np.size(array_data, axis=0)

        start_lim = np.asarray(list(zip(*lim_borders))[0])
        stop_lim = np.asarray(list(zip(*lim_borders))[1])

        temp_1 = np.floor(nb_div * (array_data - start_lim) / (stop_lim - start_lim))

        bins_array_1 = start_lim + temp_1 * (stop_lim - start_lim) / nb_div

        unique_bins_1, count_per_bin_1 = np.unique(bins_array_1, axis=0, return_counts=True)
        proba_per_bin_1 = count_per_bin_1 / n_pop_1

        dict_proba_per_bin_1 = dict(zip(map(tuple, unique_bins_1), proba_per_bin_1))

        # Completing Uniform Dict
        # So that both dictionaries share the same keys
        dict_proba_uniform_same_bins = {}
        for tuple_bin_gt in dict_proba_per_bin_1.keys(): # They share the same keys with the other dict
            dict_proba_uniform_same_bins[tuple_bin_gt] = 1. / float(len(dict_proba_per_bin_1))

        # Getting avg proba dict
        dict_proba_average = {}
        for tuple_bin_gt in dict_proba_per_bin_1.keys(): # They share the same keys with the other dict
            dict_proba_average[tuple_bin_gt] = (dict_proba_per_bin_1[tuple_bin_gt]
                                                + dict_proba_uniform_same_bins[tuple_bin_gt]) / 2.

        res_jds_dist_temp = 0.
        for tuple_bin_gt in dict_proba_per_bin_1.keys():
            proba_bin_1 = dict_proba_per_bin_1[tuple_bin_gt]
            proba_bin_2 = dict_proba_uniform_same_bins[tuple_bin_gt]
            proba_bin_avg = dict_proba_average[tuple_bin_gt]

            res_jds_dist_temp += _kl(proba_bin_1, proba_bin_avg) + _kl(proba_bin_2, proba_bin_avg)

        res_jds_dist = np.sqrt(res_jds_dist_temp / 2)

        return res_jds_dist

    @classmethod
    def save_figure(cls,
                    folder_path_save,
                    df: pd.DataFrame,
                    y,
                    name_file,
                    df_l: pd.DataFrame=None,
                    ):
        sns.set_style('whitegrid')

        if df_l is not None:
            if y in df_l.columns:
                df = df_l

        df_without_nans = df[~df[y].isnull()]

        def first_second_third_quartile(self_, vals, grouper, units=None):
            # Group and get the aggregation estimate
            grouped = vals.groupby(grouper, sort=self_.sort)
            est = grouped.agg('median')
            min_val = grouped.quantile(0.25)
            max_val = grouped.quantile(0.75)
            cis = pd.DataFrame(np.c_[min_val, max_val],
                               index=est.index,
                               columns=["low", "high"]).stack()

            # Unpack the CIs into "wide" format for plotting
            if cis.notnull().any():
                cis = cis.unstack().reindex(est.index)
            else:
                cis = None

            return est.index, est, cis

        plt.figure()
        my_lineplot=sns.lineplot
        _LinePlotter.aggregate = first_second_third_quartile
        sns.relplot(x='gen', y=y, hue=cls.LEGEND, data=df_without_nans, kind="line", facet_kws={"legend_out": True})
        plt.xlim(0, cls.get_number_generations())
        plt.xlabel('generation')
        # plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
        # plt.legend(loc='center left', bbox_to_anchor=(1.25, 0.5), ncol=1)
        plt.savefig(os.path.join(folder_path_save, name_file))
        plt.close()

    @classmethod
    def save_figure_one_variant(cls,
                                folder_path_save,
                                df: pd.DataFrame,
                                y,
                                name_file,
                                df_l: pd.DataFrame=None,
                                ) -> str:
        sns.set_style('whitegrid')
        # list_variants = list(set(df[cls.LEGEND].tolist()))

        if df_l is not None:
            if y in df_l.columns:
                df = df_l

        df_without_nans = df[~df[y].isnull()]
        sns.relplot(x=cls.GEN, y=y, hue=cls.SUBDIR, data=df_without_nans, kind="line", facet_kws={"legend_out": True})
        plt.xlim(0, cls.get_number_generations())
        plt.xlabel('generation')
        path_fig = os.path.join(folder_path_save, name_file)
        # plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
        # plt.legend(loc='center left', bbox_to_anchor=(1.25, 0.5), ncol=1)
        plt.savefig(path_fig)
        plt.close()

        return path_fig

    @classmethod
    def get_list_subdirs(cls, folder_path):
        return [d for d in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, d))]

    @classmethod
    def get_data_fit(cls, array_gt_positions, array_fitness_values, xlim, ylim, nb_div):
        array_gt_positions = array_gt_positions[:, 0:2]
        start_lim = np.asarray([xlim[0], ylim[0]])
        stop_lim = np.asarray([xlim[1], ylim[1]])

        temp = np.floor(nb_div * (array_gt_positions - start_lim) / (stop_lim - start_lim))
        bins_array = start_lim + temp * (stop_lim - start_lim) / nb_div
        bins_array = np.round(bins_array, 4)
        # print(start_lim, stop_lim, bins_array)

        unique_bins_pos, count_per_bin_pos = np.unique(bins_array, axis=0, return_counts=True)
        count_per_bin_pos = np.reshape(count_per_bin_pos, unique_bins_pos[:, 1].shape)
        # print(array_fitness_values)

        dict_best_fitness_per_bin = {}
        for bin, fitness in zip(bins_array, array_fitness_values):
            tuple_bin = tuple(bin)
            if tuple_bin not in dict_best_fitness_per_bin:
                dict_best_fitness_per_bin[tuple_bin] = fitness.item()
            else:
                if fitness.item() > dict_best_fitness_per_bin[tuple_bin]:
                    dict_best_fitness_per_bin[tuple_bin] = fitness.item()

        list_best_fitnesses_in_order = []
        for bin in unique_bins_pos:
            tuple_bin = tuple(bin)
            list_best_fitnesses_in_order.append(dict_best_fitness_per_bin[tuple_bin])

        array_best_fitnesses = np.asarray(list_best_fitnesses_in_order)
        array_best_fitnesses = np.reshape(array_best_fitnesses, unique_bins_pos[:, 1].shape)

        list_unique_bins_pos = unique_bins_pos.tolist()
        list_missing_bins = []
        #
        for i in np.round(np.linspace(*xlim, nb_div, endpoint=False), 4):
            for j in np.round(np.linspace(*ylim, nb_div, endpoint=False), 4):
                if [i, j] not in list_unique_bins_pos:
                    list_missing_bins.append([i, j])

        array_missing_bins = np.asarray(list_missing_bins)
        if not list_missing_bins:
            array_missing_bins = array_missing_bins.reshape((-1, 2))
        count_for_missing_bins = np.zeros_like(array_missing_bins[:, 0])
        fitness_for_missing_bins = np.zeros_like(array_missing_bins[:, 0])
        unique_bins_pos = np.vstack((unique_bins_pos, array_missing_bins))
        # unique_bins_pos[:, 1] = 1 - unique_bins_pos[:, 1]

        mean_fitness = np.mean(array_best_fitnesses)
        best_fitness = np.max(array_best_fitnesses)

        # count_per_bin_pos = np.vstack((count_per_bin_pos.reshape(-1,1), count_for_missing_bins.reshape(-1,1)))
        array_best_fitnesses = np.vstack((array_best_fitnesses.reshape(-1,1), fitness_for_missing_bins.reshape(-1,1)))

        return array_best_fitnesses, unique_bins_pos, mean_fitness, best_fitness
        # print(df)

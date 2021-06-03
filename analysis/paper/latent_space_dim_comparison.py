import os
import sys

from collections import OrderedDict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import ListedColormap

from seaborn.algorithms import bootstrap
from seaborn.categorical import _CategoricalStatPlotter
from seaborn.relational import _LinePlotter
from seaborn.utils import remove_na

sys.path.append('/Users/looka/git/sferes2/exp/aurora/submodules/figure_generator')
sys.path.append('/Users/looka/git/sferes2/exp/aurora/')

sys.path.append('/git/sferes2/exp/aurora/submodules/figure_generator')
sys.path.append('/git/sferes2/exp/aurora/')

import singularity.collections_experiments.maze as maze_experiments

from analysis.metrics.metrics import Metrics
import analysis.metrics.air_hockey
import analysis.metrics.maze

import analysis.paper.dataframe_preprocessor as dataframe_preprocessor


class LatentSpaceDimComparison(object):

    @classmethod
    def get_comparison_chosen_variants_custom_scripts(cls,
                                                      folder_path_save,
                                                      df: pd.DataFrame,
                                                      environment,
                                                      y,
                                                      y_lim,
                                                      name_file,
                                                      ax,
                                                      hue_order,
                                                      specified_hue=None,
                                                      no_y_log_scale=False,
                                                      rasterize=False,
                                                      linestyles=None,
                                                      list_colors=None,
                                                      list_markers=None,
                                                      markevery=1,
                                                      dict_legends=None,
                                                      alpha_median=0.7,
                                                      linewidth_median=1.,
                                                      ):
        if dict_legends is None:
            dict_legends = dict()

        # sns.set_style('whitegrid')

        df_without_nans = df[~df[y].isnull()]
        if dataframe_preprocessor.ENVIRONMENT in df_without_nans.columns:
            df_without_nans = df_without_nans[df_without_nans[dataframe_preprocessor.ENVIRONMENT] == environment]

        # f, ax = plt.subplots(figsize=(11, 7))
        # sns.boxplot(x=dataframe_preprocessor.LATENT_SPACE_DIM,
        #             y=y,
        #             hue=Metrics.LEGEND,
        #             data=df_without_nans,
        #             # kind="line",
        #             ax=ax,
        #             # facet_kws={"legend_out": True},
        #             )
        if list_colors is None:
            my_cmap = ListedColormap(sns.color_palette("colorblind").as_hex())
            list_colors = my_cmap.colors[::-1]

        if specified_hue is None:
            specified_hue = dataframe_preprocessor.NAME_VARIANT

        x = analysis.metrics.air_hockey.MetricsAirHockey.GEN

        df_without_nans = df_without_nans[[x, y, specified_hue]]
        if environment == "air_hockey":
            df_without_nans = df_without_nans[df_without_nans[x] % 100 == 0]
        else:
            df_without_nans = df_without_nans[df_without_nans[x] % 10 == 0]
        stats = df_without_nans.groupby([specified_hue, analysis.metrics.air_hockey.MetricsAirHockey.GEN]).describe()
        print(stats)

        medians = stats[(y, '50%')].unstack(level=0)
        quartiles1 = stats[(y, '25%')].unstack(level=0)
        quartiles3 = stats[(y, '75%')].unstack(level=0)

        print(medians)

        if linestyles is None:
            linestyles = ['-'] * len(hue_order)
        if list_markers is None:
            list_markers = [None] * len(hue_order)

        # ax.set_prop_cycle(cycler("color", my_cmap.colors))
        for index, name_variant in enumerate(hue_order):
            if name_variant in dict_legends:
                name_legend = dict_legends[name_variant]
            else:
                name_legend = name_variant
            print(name_legend)
            if linestyles[index] == "--":
                zorder = 2
            else:
                zorder = 1
            ax.plot(medians.index.values.tolist(),
                    medians[name_variant],
                    label=name_legend, alpha=alpha_median, color=list_colors[index], linestyle=linestyles[index], linewidth=linewidth_median,
                    marker=list_markers[index], markevery=markevery,
                    zorder=zorder)
            ax.fill_between(medians.index.values.tolist(), quartiles1[name_variant], quartiles3[name_variant], alpha=0.2, color=list_colors[index], zorder=zorder)

        print(df_without_nans)


        if (y == analysis.metrics.air_hockey.MetricsAirHockey.SIZE_POP) and (not no_y_log_scale):
            ax.set_yscale('log')

        if y_lim:
            ax.set_ylim(y_lim)

        return medians

        # plt.show()
        # plt.xlim(0, number_generations)
        # plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

        # plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
        # plt.legend(loc='center left', bbox_to_anchor=(1.25, 0.5), ncol=1)
        # plt.subplots_adjust(right=0.85, left=0.05)
        # plt.savefig("output.png", bbox_inches="tight")
        # plt.tight_layout()
        # plt.legend(loc='center left', bbox_to_anchor=(1.25, 0.5), ncol=1)
        # plt.show()
        # plt.savefig(os.path.join(folder_path_save, name_file))
        # plt.close()

    @classmethod
    def get_comparison_latent_space_metric_custom_script(cls,
                                                         folder_path_save,
                                                         df: pd.DataFrame,
                                                         environment,
                                                         y,
                                                         y_lim,
                                                         chosen_generation,
                                                         name_file,
                                                         ax,
                                                         hue_order,
                                                         ):
        def estimate_statistic(self, estimator, ci, n_boot, seed):

            if self.hue_names is None:
                statistic = []
                confint = []
            else:
                statistic = [[] for _ in self.plot_data]
                confint = [[] for _ in self.plot_data]

            for i, group_data in enumerate(self.plot_data):

                # Option 1: we have a single layer of grouping
                # --------------------------------------------

                if self.plot_hues is None:

                    if self.plot_units is None:
                        stat_data = remove_na(group_data)
                        unit_data = None
                    else:
                        unit_data = self.plot_units[i]
                        have = pd.notnull(np.c_[group_data, unit_data]).all(axis=1)
                        stat_data = group_data[have]
                        unit_data = unit_data[have]

                    # Estimate a statistic from the vector of data
                    if not stat_data.size:
                        statistic.append(np.nan)
                    else:
                        # statistic.append(estimator(stat_data))
                        statistic.append(np.median(stat_data))  # COUCOU - Median instead of mean

                    # Get a confidence interval for this estimate
                    if ci is not None:

                        if stat_data.size < 2:
                            confint.append([np.nan, np.nan])
                            continue

                        if ci == "sd":

                            estimate = estimator(stat_data)
                            sd = np.std(stat_data)
                            confint.append((estimate - sd, estimate + sd))

                        elif ci == 'interquartile':
                            print("COUCOU - interquartile")
                            confint.append((np.quantile(stat_data, 0.25),
                                            np.quantile(stat_data, 0.75)))

                        else:

                            boots = bootstrap(stat_data, func=estimator,
                                              n_boot=n_boot,
                                              units=unit_data,
                                              seed=seed)
                            confint.append(sns.utils.ci(boots, ci))

                # Option 2: we are grouping by a hue layer
                # ----------------------------------------

                else:
                    for j, hue_level in enumerate(self.hue_names):

                        if not self.plot_hues[i].size:
                            statistic[i].append(np.nan)
                            if ci is not None:
                                confint[i].append((np.nan, np.nan))
                            continue

                        hue_mask = self.plot_hues[i] == hue_level
                        if self.plot_units is None:
                            stat_data = remove_na(group_data[hue_mask])
                            unit_data = None
                        else:
                            group_units = self.plot_units[i]
                            have = pd.notnull(
                                np.c_[group_data, group_units]
                            ).all(axis=1)
                            stat_data = group_data[hue_mask & have]
                            unit_data = group_units[hue_mask & have]

                        # Estimate a statistic from the vector of data
                        if not stat_data.size:
                            statistic[i].append(np.nan)
                        else:
                            # statistic[i].append(estimator(stat_data))
                            statistic[i].append(np.median(stat_data))

                        # Get a confidence interval for this estimate
                        if ci is not None:

                            if stat_data.size < 2:
                                confint[i].append([np.nan, np.nan])
                                continue

                            if ci == "sd":

                                estimate = estimator(stat_data)
                                sd = np.std(stat_data)
                                confint[i].append((estimate - sd, estimate + sd))

                            elif ci == 'interquartile':
                                print("COUCOU - interquartile")
                                estimate = np.median(stat_data)
                                confint[i].append((np.quantile(stat_data, 0.25),
                                                   np.quantile(stat_data, 0.75)))

                            else:

                                boots = bootstrap(stat_data, func=estimator,
                                                  n_boot=n_boot,
                                                  units=unit_data,
                                                  seed=seed)
                                confint[i].append(sns.utils.ci(boots, ci))

            # Save the resulting values for plotting
            self.statistic = np.array(statistic)
            self.confint = np.array(confint)

        # sns.set_style('whitegrid')

        df_without_nans = df[~df[y].isnull()]
        df_without_nans = df_without_nans[df_without_nans[dataframe_preprocessor.ENVIRONMENT] == environment]
        if chosen_generation is not None:
            df_without_nans = df_without_nans[df_without_nans["gen"] == chosen_generation]

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

        # f, ax = plt.subplots(figsize=(11, 7))
        my_lineplot = sns.lineplot
        _LinePlotter.aggregate = first_second_third_quartile
        _CategoricalStatPlotter.estimate_statistic = estimate_statistic
        # sns.boxplot(x=dataframe_preprocessor.LATENT_SPACE_DIM,
        #             y=y,
        #             hue=Metrics.LEGEND,
        #             data=df_without_nans,
        #             # kind="line",
        #             ax=ax,
        #             # facet_kws={"legend_out": True},
        #             )
        sns.set_palette(sns.color_palette("colorblind", as_cmap=True))
        print(df_without_nans.columns)

        sns.lineplot(x=dataframe_preprocessor.LATENT_SPACE_DIM,
                     y=y,
                     hue=Metrics.LEGEND,
                     data=df_without_nans,
                     # kind="line",
                     marker="o",
                     # dodge=True,
                     # facet_kws={"legend_out": True},
                     # ci="interquartile"
                     markers=True,
                     ax=ax,
                     hue_order=hue_order,
                     )

        #
        # if y == analysis.metrics.air_hockey.MetricsAirHockey.SIZE_POP:
        #     ax.set_yscale('log')

        if y_lim:
            ax.set_ylim(y_lim)


        # plt.show()
        # plt.xlim(0, number_generations)
        # plt.xlabel("Latent Space Dimensions")
        # plt.grid(True, which="both", axis="y")
        # plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

        # plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
        # plt.legend(loc='center left', bbox_to_anchor=(1.25, 0.5), ncol=1)
        # plt.subplots_adjust(right=0.85, left=0.05)
        # plt.savefig("output.png", bbox_inches="tight")
        # plt.tight_layout()
        # plt.legend(loc='center left', bbox_to_anchor=(1.25, 0.5), ncol=1)
        # plt.show()
        # plt.savefig(os.path.join(folder_path_save, name_file))
        # plt.close()


    @classmethod
    def get_comparison_latent_space_metric(cls,
                                           folder_path_save,
                                           df: pd.DataFrame,
                                           environment,
                                           y,
                                           y_lim,
                                           chosen_generation,
                                           name_file,

                                           ):
        def estimate_statistic(self, estimator, ci, n_boot, seed):

            if self.hue_names is None:
                statistic = []
                confint = []
            else:
                statistic = [[] for _ in self.plot_data]
                confint = [[] for _ in self.plot_data]

            for i, group_data in enumerate(self.plot_data):

                # Option 1: we have a single layer of grouping
                # --------------------------------------------

                if self.plot_hues is None:

                    if self.plot_units is None:
                        stat_data = remove_na(group_data)
                        unit_data = None
                    else:
                        unit_data = self.plot_units[i]
                        have = pd.notnull(np.c_[group_data, unit_data]).all(axis=1)
                        stat_data = group_data[have]
                        unit_data = unit_data[have]

                    # Estimate a statistic from the vector of data
                    if not stat_data.size:
                        statistic.append(np.nan)
                    else:
                        # statistic.append(estimator(stat_data))
                        statistic.append(np.median(stat_data))  # COUCOU - Median instead of mean

                    # Get a confidence interval for this estimate
                    if ci is not None:

                        if stat_data.size < 2:
                            confint.append([np.nan, np.nan])
                            continue

                        if ci == "sd":

                            estimate = estimator(stat_data)
                            sd = np.std(stat_data)
                            confint.append((estimate - sd, estimate + sd))

                        elif ci == 'interquartile':
                            print("COUCOU - interquartile")
                            confint.append((np.quantile(stat_data, 0.25),
                                            np.quantile(stat_data, 0.75)))

                        else:

                            boots = bootstrap(stat_data, func=estimator,
                                              n_boot=n_boot,
                                              units=unit_data,
                                              seed=seed)
                            confint.append(sns.utils.ci(boots, ci))

                # Option 2: we are grouping by a hue layer
                # ----------------------------------------

                else:
                    for j, hue_level in enumerate(self.hue_names):

                        if not self.plot_hues[i].size:
                            statistic[i].append(np.nan)
                            if ci is not None:
                                confint[i].append((np.nan, np.nan))
                            continue

                        hue_mask = self.plot_hues[i] == hue_level
                        if self.plot_units is None:
                            stat_data = remove_na(group_data[hue_mask])
                            unit_data = None
                        else:
                            group_units = self.plot_units[i]
                            have = pd.notnull(
                                np.c_[group_data, group_units]
                            ).all(axis=1)
                            stat_data = group_data[hue_mask & have]
                            unit_data = group_units[hue_mask & have]

                        # Estimate a statistic from the vector of data
                        if not stat_data.size:
                            statistic[i].append(np.nan)
                        else:
                            # statistic[i].append(estimator(stat_data))
                            statistic[i].append(np.median(stat_data))

                        # Get a confidence interval for this estimate
                        if ci is not None:

                            if stat_data.size < 2:
                                confint[i].append([np.nan, np.nan])
                                continue

                            if ci == "sd":

                                estimate = estimator(stat_data)
                                sd = np.std(stat_data)
                                confint[i].append((estimate - sd, estimate + sd))

                            elif ci == 'interquartile':
                                print("COUCOU - interquartile")
                                estimate = np.median(stat_data)
                                confint[i].append((np.quantile(stat_data, 0.25),
                                                   np.quantile(stat_data, 0.75)))

                            else:

                                boots = bootstrap(stat_data, func=estimator,
                                                  n_boot=n_boot,
                                                  units=unit_data,
                                                  seed=seed)
                                confint[i].append(sns.utils.ci(boots, ci))

            # Save the resulting values for plotting
            self.statistic = np.array(statistic)
            self.confint = np.array(confint)

        sns.set_style('whitegrid')

        df_without_nans = df[~df[y].isnull()]
        df_without_nans = df_without_nans[df_without_nans[dataframe_preprocessor.ENVIRONMENT] == environment]
        df_without_nans = df_without_nans[df_without_nans["gen"] == chosen_generation]

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

        f, ax = plt.subplots(figsize=(11, 7))
        my_lineplot = sns.lineplot
        _LinePlotter.aggregate = first_second_third_quartile
        _CategoricalStatPlotter.estimate_statistic = estimate_statistic
        # sns.boxplot(x=dataframe_preprocessor.LATENT_SPACE_DIM,
        #             y=y,
        #             hue=Metrics.LEGEND,
        #             data=df_without_nans,
        #             # kind="line",
        #             ax=ax,
        #             # facet_kws={"legend_out": True},
        #             )
        sns.lineplot(x=dataframe_preprocessor.LATENT_SPACE_DIM,
                      y=y,
                      hue=Metrics.LEGEND,
                      data=df_without_nans,
                      # kind="line",
                      ax=ax,
                     marker="o",
                      # dodge=True,
                      # facet_kws={"legend_out": True},
                      # ci="interquartile"
                     markers=True,
                      )


        if y == analysis.metrics.air_hockey.MetricsAirHockey.SIZE_POP:
            ax.set_yscale('log')

        if y_lim:
            ax.set_ylim(y_lim)


        # plt.show()
        # plt.xlim(0, number_generations)
        plt.xlabel("Latent Space Dimensions")
        plt.grid(True, which="both", axis="y")
        # plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

        plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
        # plt.legend(loc='center left', bbox_to_anchor=(1.25, 0.5), ncol=1)
        # plt.subplots_adjust(right=0.85, left=0.05)
        # plt.savefig("output.png", bbox_inches="tight")
        plt.tight_layout()
        # plt.legend(loc='center left', bbox_to_anchor=(1.25, 0.5), ncol=1)
        # plt.show()
        plt.savefig(os.path.join(folder_path_save, name_file))
        plt.close()

    @classmethod
    def get_comparison_chosen_variants(cls,
                                       folder_path_save,
                                       df: pd.DataFrame,
                                       environment,
                                       y,
                                       y_lim,
                                       name_file,
                                       ):
        def estimate_statistic(self, estimator, ci, n_boot, seed):

            if self.hue_names is None:
                statistic = []
                confint = []
            else:
                statistic = [[] for _ in self.plot_data]
                confint = [[] for _ in self.plot_data]

            for i, group_data in enumerate(self.plot_data):

                # Option 1: we have a single layer of grouping
                # --------------------------------------------

                if self.plot_hues is None:

                    if self.plot_units is None:
                        stat_data = remove_na(group_data)
                        unit_data = None
                    else:
                        unit_data = self.plot_units[i]
                        have = pd.notnull(np.c_[group_data, unit_data]).all(axis=1)
                        stat_data = group_data[have]
                        unit_data = unit_data[have]

                    # Estimate a statistic from the vector of data
                    if not stat_data.size:
                        statistic.append(np.nan)
                    else:
                        # statistic.append(estimator(stat_data))
                        statistic.append(np.median(stat_data))  # COUCOU - Median instead of mean

                    # Get a confidence interval for this estimate
                    if ci is not None:

                        if stat_data.size < 2:
                            confint.append([np.nan, np.nan])
                            continue

                        if ci == "sd":

                            estimate = estimator(stat_data)
                            sd = np.std(stat_data)
                            confint.append((estimate - sd, estimate + sd))

                        elif ci == 'interquartile':
                            print("COUCOU - interquartile")
                            confint.append((np.quantile(stat_data, 0.25),
                                            np.quantile(stat_data, 0.75)))

                        else:

                            boots = bootstrap(stat_data, func=estimator,
                                              n_boot=n_boot,
                                              units=unit_data,
                                              seed=seed)
                            confint.append(sns.utils.ci(boots, ci))

                # Option 2: we are grouping by a hue layer
                # ----------------------------------------

                else:
                    for j, hue_level in enumerate(self.hue_names):

                        if not self.plot_hues[i].size:
                            statistic[i].append(np.nan)
                            if ci is not None:
                                confint[i].append((np.nan, np.nan))
                            continue

                        hue_mask = self.plot_hues[i] == hue_level
                        if self.plot_units is None:
                            stat_data = remove_na(group_data[hue_mask])
                            unit_data = None
                        else:
                            group_units = self.plot_units[i]
                            have = pd.notnull(
                                np.c_[group_data, group_units]
                            ).all(axis=1)
                            stat_data = group_data[hue_mask & have]
                            unit_data = group_units[hue_mask & have]

                        # Estimate a statistic from the vector of data
                        if not stat_data.size:
                            statistic[i].append(np.nan)
                        else:
                            # statistic[i].append(estimator(stat_data))
                            statistic[i].append(np.median(stat_data))

                        # Get a confidence interval for this estimate
                        if ci is not None:

                            if stat_data.size < 2:
                                confint[i].append([np.nan, np.nan])
                                continue

                            if ci == "sd":

                                estimate = estimator(stat_data)
                                sd = np.std(stat_data)
                                confint[i].append((estimate - sd, estimate + sd))

                            elif ci == 'interquartile':
                                print("COUCOU - interquartile")
                                estimate = np.median(stat_data)
                                confint[i].append((np.quantile(stat_data, 0.25),
                                                   np.quantile(stat_data, 0.75)))

                            else:

                                boots = bootstrap(stat_data, func=estimator,
                                                  n_boot=n_boot,
                                                  units=unit_data,
                                                  seed=seed)
                                confint[i].append(sns.utils.ci(boots, ci))

            # Save the resulting values for plotting
            self.statistic = np.array(statistic)
            self.confint = np.array(confint)

        sns.set_style('whitegrid')

        df_without_nans = df[~df[y].isnull()]
        df_without_nans = df_without_nans[df_without_nans[dataframe_preprocessor.ENVIRONMENT] == environment]

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

        f, ax = plt.subplots(figsize=(11, 7))
        my_lineplot = sns.lineplot
        _LinePlotter.aggregate = first_second_third_quartile
        _CategoricalStatPlotter.estimate_statistic = estimate_statistic
        # sns.boxplot(x=dataframe_preprocessor.LATENT_SPACE_DIM,
        #             y=y,
        #             hue=Metrics.LEGEND,
        #             data=df_without_nans,
        #             # kind="line",
        #             ax=ax,
        #             # facet_kws={"legend_out": True},
        #             )
        sns.lineplot(x=analysis.metrics.air_hockey.MetricsAirHockey.GEN,
                    y=y,
                    hue=dataframe_preprocessor.NAME_VARIANT,
                    data=df_without_nans,
                    # kind="line",
                    ax=ax,
                    # markers=True,
                     # marker="o",
                     # facet_kws={"legend_out": True},
                    )


        if y == analysis.metrics.air_hockey.MetricsAirHockey.SIZE_POP:
            ax.set_yscale('log')

        if y_lim:
            ax.set_ylim(y_lim)


        # plt.show()
        # plt.xlim(0, number_generations)
        plt.xlabel("Latent Space Dimensions")
        plt.grid(True, which="both", axis="y")
        # plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

        plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
        # plt.legend(loc='center left', bbox_to_anchor=(1.25, 0.5), ncol=1)
        # plt.subplots_adjust(right=0.85, left=0.05)
        # plt.savefig("output.png", bbox_inches="tight")
        plt.tight_layout()
        # plt.legend(loc='center left', bbox_to_anchor=(1.25, 0.5), ncol=1)
        # plt.show()
        plt.savefig(os.path.join(folder_path_save, name_file))
        plt.close()


def filter_and_add_supplementary_legend_comparison_latent_dim(df_old, do_get_list_legends=False):
    list_name_variant_paper = []
    list_legends = []

    list_dimensions_latent_space = list(maze_experiments.DICT_MAZE_AURORA_UNIFORM_n_COLORS.keys())

    # for dim_latent_space in list_dimensions_latent_space:
    #     list_name_variant_paper.append(f"aurora_uniform_{dim_latent_space}_psat")
    #     list_legends.append(f"aurora_uniform_$n$_psat")
    #
    # for dim_latent_space in list_dimensions_latent_space:
    #     list_name_variant_paper.append(f"aurora_uniform_{dim_latent_space}_vat")
    #     list_legends.append(f"aurora_uniform_$n$_vat")

    for dim_latent_space in list_dimensions_latent_space:
        list_name_variant_paper.append(f"aurora_uniform_{dim_latent_space}_no_norm")
        list_legends.append(f"AURORA-CSC-uniform-$n$")

    for dim_latent_space in list_dimensions_latent_space:
        list_name_variant_paper.append(f"aurora_uniform_{dim_latent_space}_vat_no_norm")
        list_legends.append(f"AURORA-VAT-uniform-$n$")

    df_legend = pd.DataFrame({
        dataframe_preprocessor.NAME_VARIANT: list_name_variant_paper,
        "legend": list_legends,
    })
    print(df_legend)

    df = pd.merge(df_old,
                  df_legend,
                  on=dataframe_preprocessor.NAME_VARIANT,
                  how="inner",
                  )
    if not do_get_list_legends:
        return df
    else:
        return df, list(OrderedDict.fromkeys(list_legends))  # list_legends without duplicates


def main():
    # df = dataframe_preprocessor.get_preprocessed_df()
    # dataframe_preprocessor.save_df(df)

    df = pd.read_csv("df.csv")

    df = filter_and_add_supplementary_legend_comparison_latent_dim(df)

    print(df[df["legend"] == "aurora_uniform_$n$_vat"][df["gen"] == 5000]["size_pop"])

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

    folder_path_save = os.path.join(os.path.dirname(os.path.abspath(__file__)), "latent_space_dim_comparison")
    if not os.path.exists(folder_path_save):
        os.mkdir(folder_path_save)

    y_coverage = analysis.metrics.air_hockey.MetricsAirHockey.COVERAGE_POS + "_40"
    for environment, chosen_generation in dict_environment_chosen_generation.items():
        list_y = [
            analysis.metrics.air_hockey.MetricsAirHockey.SIZE_POP,
            y_coverage,
            dataframe_preprocessor.UNIFORMITY,
        ]
        for y in list_y:
            print(environment, y)
            if y in [y_coverage,
                     analysis.metrics.air_hockey.MetricsAirHockey.JDS_UNIFORM_EXPLORED]:
                y_lim = (0., 1.)
            elif y == analysis.metrics.air_hockey.MetricsAirHockey.SIZE_POP:
                y_lim = (1e2, 1e4)
            else:
                y_lim = None

            LatentSpaceDimComparison.get_comparison_latent_space_metric(
                folder_path_save=folder_path_save,
                df=df,
                environment=environment,
                y=y,
                y_lim=y_lim,
                chosen_generation=chosen_generation,
                name_file=f"latent_space_dim_comparison_{environment}_{y}.png"
            )

    # ----------------------------------------

    del df
    df = pd.read_csv("df.csv")
    df = df[df["name_variant"].isin([
        "TAXONS_10",
        "TAXO_N_10",
        "TAXO_S_10",
        "NS",
    ])]
    list_y = [
        # analysis.metrics.air_hockey.MetricsAirHockey.SIZE_POP,
        y_coverage,
        dataframe_preprocessor.UNIFORMITY,
    ]

    folder_path_save_taxons = os.path.join(os.path.dirname(os.path.abspath(__file__)), "comparison_taxons")
    if not os.path.exists(folder_path_save_taxons):
        os.mkdir(folder_path_save_taxons)

    for environment in dict_environment_chosen_generation:
        for y in list_y:
            print(environment, y)
            if y in [y_coverage,
                     analysis.metrics.air_hockey.MetricsAirHockey.JDS_UNIFORM_EXPLORED]:
                y_lim = (0., 1.)
            elif y == analysis.metrics.air_hockey.MetricsAirHockey.SIZE_POP:
                y_lim = (1e2, 1e4)
            else:
                y_lim = None

            LatentSpaceDimComparison.get_comparison_chosen_variants(
                folder_path_save=folder_path_save_taxons,
                df=df,
                environment=environment,
                y=y,
                y_lim=y_lim,
                name_file=f"latent_space_dim_comparison_{environment}_{y}.png"
            )

    # ----------------------------------------

    del df
    df = pd.read_csv("df.csv")
    df = df[df["name_variant"].isin([
        "aurora_uniform_10_psat",
        "aurora_curiosity_10_psat",
        "aurora_novelty_10_psat",
        "aurora_nov_sur_10_psat",
        "TAXONS_10",
        "qd_uniform_psat",
        "qd_no_selection_psat",
    ])]
    list_y = [
        # analysis.metrics.air_hockey.MetricsAirHockey.SIZE_POP,
        y_coverage,
        dataframe_preprocessor.UNIFORMITY,
    ]

    folder_path_save_taxons = os.path.join(os.path.dirname(os.path.abspath(__file__)), "comparison_aurora")
    if not os.path.exists(folder_path_save_taxons):
        os.mkdir(folder_path_save_taxons)

    for environment in dict_environment_chosen_generation:
        for y in list_y:
            print(environment, y)
            if y in [y_coverage,
                     analysis.metrics.air_hockey.MetricsAirHockey.JDS_UNIFORM_EXPLORED]:
                y_lim = (0., 1.)
            elif y == analysis.metrics.air_hockey.MetricsAirHockey.SIZE_POP:
                y_lim = (1e2, 1e4)
            else:
                y_lim = None

            LatentSpaceDimComparison.get_comparison_chosen_variants(
                folder_path_save=folder_path_save_taxons,
                df=df,
                environment=environment,
                y=y,
                y_lim=y_lim,
                name_file=f"latent_space_dim_comparison_{environment}_{y}.png"
            )

    # ----------------------------------------

    del df
    df = pd.read_csv("df.csv")
    df = df[df["name_variant"].isin([
        "aurora_uniform_10_psat_fit",
        "qd_uniform_psat_fit",
    ])]
    list_y = [
        # analysis.metrics.air_hockey.MetricsAirHockey.SIZE_POP,
        analysis.metrics.maze.MetricsMaze.MEAN_FITNESS,
    ]

    folder_path_save_taxons = os.path.join(os.path.dirname(os.path.abspath(__file__)), "comparison_fit")
    if not os.path.exists(folder_path_save_taxons):
        os.mkdir(folder_path_save_taxons)

    for environment in dict_environment_chosen_generation:
        for y in list_y:
            print(environment, y)
            if y in [y_coverage,
                     analysis.metrics.air_hockey.MetricsAirHockey.JDS_UNIFORM_EXPLORED]:
                y_lim = (0., 1.)
            elif y == analysis.metrics.air_hockey.MetricsAirHockey.SIZE_POP:
                y_lim = (1e2, 1e4)
            else:
                y_lim = None

            LatentSpaceDimComparison.get_comparison_chosen_variants(
                folder_path_save=folder_path_save_taxons,
                df=df,
                environment=environment,
                y=y,
                y_lim=y_lim,
                name_file=f"latent_space_dim_comparison_{environment}_{y}.png"
            )


if __name__ == '__main__':
    main()

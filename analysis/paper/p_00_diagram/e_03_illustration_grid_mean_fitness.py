import os

import argparse

import itertools
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import sys
import pandas as pd
import seaborn as sns
from matplotlib import ticker, lines

sys.path.append('/Users/looka/git/sferes2/exp/aurora/submodules/figure_generator')
sys.path.append('/Users/looka/git/sferes2/exp/aurora/')

sys.path.append('/git/sferes2/exp/aurora/submodules/figure_generator')
sys.path.append('/git/sferes2/exp/aurora/')

from analysis.metrics.metrics import Metrics
import analysis.metrics.air_hockey
import analysis.metrics.maze
from data_reader import get_data_proj, get_data_modifier, get_data_offspring


import analysis.paper.plot_utils as pu
import analysis.paper.latent_space_dim_comparison as latent_space_dim_comparison
import analysis.paper.dataframe_preprocessor as dataframe_preprocessor

import analysis.paper.p_01_comparison_variants.e_04_plots_fitness as e_04_plots_fitness


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('-s', '--save')

    args = parser.parse_args()
    return args


def draw_heatmap(ax, path_proj_file, nb_div, xlim, ylim, array_gt_positions=None, array_fitness_values=None, vmax=0, vmin=None, linewidths=0., linecolor="white"):
    try:
        array_best_fitnesses, unique_bins_pos, mean_fitness, best_fitness = e_04_plots_fitness.get_data(path_proj_file, xlim, ylim, nb_div,
                                                                                                        array_gt_positions=array_gt_positions,
                                                                                                        array_fitness_values=array_fitness_values, )

        # print(np.reshape(array_best_fitnesses, unique_bins_pos[:, 1].shape))
        df = pd.DataFrame({
            "x": unique_bins_pos[:, 1],
            "y": unique_bins_pos[:, 0],
            "fitness": np.reshape(array_best_fitnesses, unique_bins_pos[:, 1].shape)
        })
        # print(df)

        df = df.pivot("x", "y", "fitness")
        # df = df.reset_index()
        # print(df)
        # sns.set_palette(sns.color_palette(CB_color_cycle))
        # ax.grid()
        im = sns.heatmap(df,
                         ax=ax,
                         # vmax=-1, vmin=0,
                         cmap="viridis",
                         cbar=False,
                         # cbar=True if col == 0 else False,
                         # cbar_ax=None if col > 0 else cbar_ax,
                         mask=(df == 0),
                         xticklabels=[str(xlim[0])] + int(nb_div - 2) * [""] + [str(xlim[1])],
                         yticklabels=[str(ylim[0])] + int(nb_div - 2) * [""] + [str(ylim[1])],
                         vmax=vmax,
                         linewidths=linewidths,
                         linecolor=linecolor,
                         vmin=vmin,
                         )

        ax.set_xlim(0, nb_div)
        ax.set_ylim(0, nb_div)


        # ax.set_xticks(list(range(0, nb_div+1, 10)))
        # ax.set_yticks(list(range(0, nb_div+1, 10)))
        ax.set_xticks([0, 2, 4, 6, 8])
        ax.set_yticks([0, 2, 4, 6, 8])

        ax.tick_params(axis='both', which='both',
                       bottom=True,      # ticks along the bottom edge are off
                       top=False,         # ticks along the top edge are off
                       labelbottom=False,
                       width=1,
                       length=1,
                       labelleft=False,
                       left=True,
                       )

        ax.spines["top"].set_visible(False)
        ax.spines["bottom"].set_visible(True)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_visible(True)

        ax.xaxis.label.set_visible(False)
        ax.yaxis.label.set_visible(False)




        # print(list(np.round(-1.25 + (np.arange(0, 51, 10) / 50) * 2.5, 2)))

        # ax.set_xticklabels(
        #     np.round(xlim[0] + (np.arange(0, nb_div+1, 10) / nb_div) * (xlim[1] - xlim[0]),2)
        # )


        # ax.set_yticklabels(
        #     np.round(ylim[0] + (np.arange(0, nb_div+1, 10) / nb_div) * (ylim[1] - ylim[0]), 2)
        # )

        return im
    except FileNotFoundError:
        pass

def generage_figure_axes(fig, axes, font_size=10):
    np.random.seed(42)
    # dataset 1
    NB_POINTS = 2000

    dataset_1 = np.random.rand(NB_POINTS, 2) * 1.95 - 1.95 / 2
    fitness_scores_1 = 1 - np.linalg.norm(dataset_1, axis=1) ** 2 / 2
    dist_to_center = np.linalg.norm(dataset_1, axis=1, ord=np.inf)
    # dataset_1 = dataset_1 / np.power(dist_to_center, 0.1).reshape(-1, 1)  # Increasing density of points at the center


    # dataset 2

    np.random.seed(42)
    dataset_2 = np.random.rand(NB_POINTS, 2) * 1.95 - 1.95 / 2
    fitness_scores_2 =  1 - np.linalg.norm(dataset_2, axis=1) ** 2 / 2

    dist_to_center = np.linalg.norm(dataset_2, axis=1, ord=np.inf)
    dataset_2 = dataset_2 * np.power(dist_to_center, 1.2).reshape(-1, 1)  # Increasing density of points at the center

    axes[0, 0].scatter(dataset_1[:, 0], dataset_1[:, 1], c=fitness_scores_1, s=1, vmin=0, vmax=1)
    axes[0, 1].scatter(dataset_2[:, 0], dataset_2[:, 1], c=fitness_scores_2, s=1, vmin=0, vmax=1)

    for ax in (axes[0, 0], axes[0, 1]):
        ax.tick_params(axis='both',
                       which='both',
                       bottom=True,      # ticks along the bottom edge are off
                       top=False,         # ticks along the top edge are off
                       labelbottom=False,
                       width=1,
                       length=1,
                       labelleft=False,
                       left=True,
                       right=False
                       )

        ax.xaxis.label.set_visible(False)
        ax.yaxis.label.set_visible(False)

        ax.spines["top"].set_visible(False)
        ax.spines["bottom"].set_visible(True)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_visible(True)
        ax.set_xlim((-1, 1))
        ax.set_ylim((-1, 1))

    NB_DIV = 8

    array_best_fitnesses, unique_bins_pos, mean_fitness_1, best_fitness = Metrics.get_data_fit(dataset_1, fitness_scores_1, xlim=(-1, 1), ylim=(-1, 1), nb_div=NB_DIV)
    print(fitness_scores_1.mean(), mean_fitness_1)

    array_best_fitnesses, unique_bins_pos, mean_fitness_2, best_fitness = Metrics.get_data_fit(dataset_2, fitness_scores_2, xlim=(-1, 1), ylim=(-1, 1), nb_div=NB_DIV)
    print(fitness_scores_2.mean(), mean_fitness_2)

    draw_heatmap(axes[1, 0], path_proj_file=None, nb_div=NB_DIV, xlim=(-1, 1), ylim=(-1, 1),
                 array_gt_positions=dataset_1,
                 array_fitness_values=fitness_scores_1,vmax=1,  linewidths=0.15, vmin=0)

    draw_heatmap(axes[1, 1], path_proj_file=None, nb_div=NB_DIV, xlim=(-1, 1), ylim=(-1, 1),
                 array_gt_positions=dataset_2,
                 array_fitness_values=fitness_scores_2, vmax=1., linewidths=0.15, vmin=0)

    # mean fitness for scatter plots
    axes[0, 0].set_xlabel(f"Mean Fitness $\\approx {fitness_scores_1.mean():.2f}$", fontsize=font_size, in_layout=True)
    axes[0, 1].set_xlabel(f"Mean Fitness $\\approx {fitness_scores_2.mean():.2f}$", fontsize=font_size, in_layout=True)

    # grid-based mean fitness for grid plots
    axes[1, 0].set_xlabel(f"Grid Mean Fitness\n$\\approx \\mathbf{{{mean_fitness_1:.2f}}}$", fontsize=font_size, in_layout=True)
    axes[1, 1].set_xlabel(f"Grid Mean Fitness\n$\\approx \\mathbf{{{mean_fitness_2:.2f}}}$", fontsize=font_size, in_layout=True)

    for (i, j) in itertools.product([0,1], [0,1]):
        axes[i, j].xaxis.label.set_visible(True)


    # for row in range(len(list_experiments)):
    #     for col in range(len(list_algorithms_folders)):
    #
    #         experiment = list_experiments[row]
    #         algorithm_folder = list_algorithms_folders[col]
    #         folder_proj_file = os.path.join(current_folder_path, dict_experiment_to_folder[experiment], algorithm_folder)
    #
    #         path_proj_file = get_last_gen_proj_file(folder_proj_file)
    #         print(path_proj_file)
    #
    #         if experiment == "maze":
    #             im = heatmap_maze(axes[row, col], path_proj_file, 50, xlim=(0, 600), ylim=(0, 600))
    #             axes[row, col].set_xlabel("$x_T$")
    #             axes[row, col].set_ylabel("$y_T$")
    #             # plot_comparison_maze(axes[row, col], path_proj_file)
    #         elif experiment == "hexapod_camera_vertical":
    #             im = heatmap_hexapod(axes[row, col], path_proj_file, 50, xlim=(-1.25, 1.25), ylim=(-1, 1))
    #             axes[row, col].set_xlabel("$x_T$")
    #             axes[row, col].set_ylabel("$y_T$")
    #         elif experiment == "air_hockey":
    #             im = heatmap_air_hockey(axes[row, col], path_proj_file, 50, xlim=(-1., 1.), ylim=(-1., 1.))
    #             axes[row, col].set_xlabel("$x_T^{\\text{puck}}$")
    #             axes[row, col].set_ylabel("$y_T^{\\text{puck}}$")
    #         print(im.get_images())
    #         # elif experiment == "air_hockey":
    #         #     plot_comparison_air_hockey(axes[row, col], path_proj_file)
    #
    #         # axes[row, col].get_legend().remove()
    #
    #         axes[row, col].tick_params(axis='both', which='both', length=1)
    #
    #
    #
    #         # ax.set_xlabel('$x$')
    #         # axes[row, col].set_xlabel(dict_algorithm_folder_to_title[algorithm_folder])
    #         # axes[row, col].set_ylabel(dict_experiment_to_title[experiment])
    #
    #
    #
    #         # axes[row, col].xaxis.label.set_visible(False)
    #
    #         # axes[row, col].set_axisbelow(True)
    #
    #         if row == 0:
    #             axes[row, col].set_title(dict_algorithm_folder_to_title[algorithm_folder])
    #
    #         axes[row, col].spines["top"].set_visible(False)
    #         axes[row, col].spines["bottom"].set_visible(True)
    #         axes[row, col].spines["right"].set_visible(False)
    #         axes[row, col].spines["left"].set_visible(True)
    #
    #         if col > 0:
    #             axes[row, col].yaxis.label.set_visible(False)
    #             axes[row, col].tick_params(axis='y',          # changes apply to the x-axis
    #                                        which='both',      # both major and minor ticks are affected
    #                                        right=False,      # ticks along the bottom edge are off
    #                                        left=True,         # ticks along the top edge are off
    #                                        labelleft=False)
    #
    # pad = 5  # in points
    # for ax, row in zip(axes[:,0], range(len(list_experiments))):
    #     ax.annotate(dict_experiment_to_title[list_experiments[row]], xy=(0, 0.5), xytext=(-ax.yaxis.labelpad - pad, 0),
    #                 xycoords=ax.yaxis.label, textcoords='offset points',
    #                 size='medium', ha='right', va='center', rotation=90)

    # ---------------

    # ax.set_xlabel('$x$')
    # plt.grid()
    # plt.tight_layout()

    # TODO

    #     for axis in axes.flat:
    #         print(axis.collections)

    # x0, y0 = fig.transFigure.inverted().transform((x0,y0))
    x0, y0 = (0.8 * np.array(axes[0, 0].transAxes.transform((1, 1.15))) + 0.2 * np.array(axes[0, 1].transAxes.transform((0, 1.15))))
    x1, y1 = (0.8 * np.array(axes[1, 0].transAxes.transform((1, -0.3))) + 0.2 * np.array(axes[1, 1].transAxes.transform((0, -0.3))))
    # x1, y1 = fig.transFigure.inverted().transform((x1,y1))
    x1, y1 = axes[0, 0].transAxes.inverted().transform((x1, y1))
    x0, y0 = axes[0, 0].transAxes.inverted().transform((x0, y0))
    print(x0,x1)
    print(y0,y1)
    line = lines.Line2D([x0, x1], [y0, y1], lw=0.5, color='black', alpha=1, linestyle='-', transform= axes[0, 0].transAxes, clip_on=True, in_layout=False)
    # axes[0, 0].add_line(line)
    fig.add_artist(line)

    cbar = plt.colorbar(mappable=axes[0,0].collections[0], ax=axes[:, :])
    cbar.ax.tick_params(axis='both',          # changes apply to the x-axis
                        which='both',      # both major and minor ticks are affected
                        width=0.5, length=3,
                        direction="in")
    cbar.ax.set_title("Fitness", fontsize=font_size)






def generate_figure(path_save):
    plt.clf()
    plt.cla()
    plt.close()

    matplotlib.rc_file_defaults()

    pu.figure_setup()

    fig_size = pu.get_fig_size(8, 8)

    # fig = plt.figure(figsize=fig_size)
    fig, axes = plt.subplots(2, 2, constrained_layout=True)
    fig.set_size_inches(*fig_size)
    # cbar_ax = fig.add_axes([.91, .3, .03, .4])

    current_folder_path = "/Users/looka/git/sferes2/exp/aurora/analysis/paper/p_01_comparison_variants/"

    generage_figure_axes(fig, axes, font_size=10)

    axes[0,0].text(-0.1, 1.05, '\\textbf{Illu}', ha='left', transform=axes[0,0].transAxes, fontsize=11)


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

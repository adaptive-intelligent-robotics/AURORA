import os

import argparse
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.lines as lines
import numpy as np
import sys
import pandas as pd
# import seaborn as sns
from matplotlib import ticker
from matplotlib.transforms import IdentityTransform

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


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('-s', '--save')

    args = parser.parse_args()
    return args


def get_all_fitness_values(list_path_proj_files):
    array_fitness_values = None
    for path_proj_file in list_path_proj_files:
        _, _, tmp_array_fitness_values, _ = get_data_proj(path_proj_file)
        if array_fitness_values is None:
            array_fitness_values = tmp_array_fitness_values
        else:
            array_fitness_values = np.vstack((array_fitness_values, tmp_array_fitness_values))
    return array_fitness_values


def get_min_max_fitness_values(list_path_proj_files):
    array_fitness_values = get_all_fitness_values(list_path_proj_files)
    return np.min(array_fitness_values), np.max(array_fitness_values)


def plot_comparison_maze(ax, path_proj_file, vmin, vmax):
    print(path_proj_file)
    # metric = analysis.metrics.air_hockey.MetricsAirHockey.COVERAGE_POS + "_50"
    _, array_gt_positions, array_fitness_values, array_novelty = get_data_proj(path_proj_file)

    # sns.set_style('whitegrid')
    # sns.set_palette(sns.color_palette("colorblind", as_cmap=True))

    argsort_fitness = np.argsort(array_fitness_values.flatten())

    # Shuffling indexes
    np.random.shuffle(argsort_fitness)

    array_gt_positions = array_gt_positions[argsort_fitness]
    array_fitness_values = array_fitness_values[argsort_fitness]

    ax.scatter(array_gt_positions[:, 0].flatten(),
               600 - array_gt_positions[:, 1].flatten(),
               c=array_fitness_values.flatten(),
               marker='o', s=1, linewidths=0, rasterized=True, cmap="viridis",
               vmin=vmin, vmax=vmax)

    ax.set_xlabel("$x_T$")
    ax.set_ylabel("$y_T$")

def plot_comparison_hexapod(ax, path_proj_file, vmin, vmax):
    # metric = analysis.metrics.air_hockey.MetricsAirHockey.COVERAGE_POS + "_50"
    _, array_gt_positions, array_fitness_values, array_novelty = get_data_proj(path_proj_file)

    # sns.set_style('whitegrid')
    # sns.set_palette(sns.color_palette("colorblind", as_cmap=True))

    print(array_gt_positions)

    argsort_fitness = np.argsort(array_fitness_values.flatten())

    # Shuffling indexes
    np.random.shuffle(argsort_fitness)

    array_gt_positions = array_gt_positions[argsort_fitness]
    array_fitness_values = array_fitness_values[argsort_fitness]

    ax.scatter(array_gt_positions[:, 1].flatten(),
               array_gt_positions[:, 0].flatten(),
               c=array_fitness_values.flatten(),
               marker='o', s=1, linewidths=0, rasterized=True, cmap="viridis",
               vmin=vmin, vmax=vmax)

    ax.set_xlim(-1.,1.)
    ax.set_ylim(-1.5, 1.5)

    ax.set_xlabel("$x_T$")
    ax.set_ylabel("$y_T$")
    # ax.set_aspect("equal", adjustable="box")

def plot_comparison_air_hockey(ax, path_proj_file, vmin, vmax):
    # metric = analysis.metrics.air_hockey.MetricsAirHockey.COVERAGE_POS + "_50"
    _, array_gt_positions, array_fitness_values, array_novelty = get_data_proj(path_proj_file)

    # sns.set_style('whitegrid')
    # sns.set_palette(sns.color_palette("colorblind", as_cmap=True))

    # ax.set_xlabel(r"$x_T^{\text{puck}}$")
    # ax.set_ylabel(r"$y_T^{\text{puck}}$")
    ax.set_xlabel("$x_T$")
    ax.set_ylabel("$y_T$")

    argsort_fitness = np.argsort(array_fitness_values.flatten())

    # Shuffling indexes
    np.random.shuffle(argsort_fitness)

    array_gt_positions = array_gt_positions[argsort_fitness]
    array_fitness_values = array_fitness_values[argsort_fitness]

    ax.scatter(array_gt_positions[:, 0].flatten(),
               array_gt_positions[:, 1].flatten(),
               c=array_fitness_values.flatten(), marker='o',
               s=1, linewidths=0, rasterized=True, cmap="viridis",
               vmin=vmin, vmax=vmax)

def get_last_gen_proj_file(path_folder_proj_file):
    for gen in np.arange(30000, 0, -1):
        path_supposed_proj_file = os.path.join(path_folder_proj_file, f"proj_{gen}.dat")
        if os.path.exists(path_supposed_proj_file):
            return path_supposed_proj_file

    raise FileNotFoundError


def generate_figure(path_save):
    plt.style.use('classic')
    plt.clf()
    plt.cla()
    plt.close()

    matplotlib.rc_file_defaults()

    pu.figure_setup()

    fig_size = pu.get_fig_size(31, 13)
    # fig = plt.figure(figsize=fig_size)
    fig, axes = plt.subplots(3, 5, constrained_layout=True)
    fig.set_size_inches(*fig_size)

    current_folder_path = "/Users/looka/git/sferes2/exp/aurora/analysis/paper/p_01_comparison_variants/"

    # ax = fig.add_subplot(131)
    # ax2 = fig.add_subplot(132)
    # ax3 = fig.add_subplot(133)


    # ax.plot(x, y, c='b', lw=pu.plot_lw())


    list_experiments = [
        "maze",
        "hexapod_camera_vertical",
        "air_hockey",
    ]

    dict_experiment_to_title = {
        "maze": "Maze",
        "hexapod_camera_vertical": "Hexapod",
        "air_hockey": "Air-Hockey",
    }

    dict_experiment_to_folder = {
        "maze": "data_archives/maze/",
        "hexapod_camera_vertical": "data_archives/hexapod/",
        "air_hockey": "data_archives/air_hockey/",
    }

    # name_folder -> title
    list_algorithms_folders = [
        "aurora_10",
        "taxons_10",
        "qd",
        "ns",
        "qd_no_sel",
    ]

    dict_algorithm_folder_to_title = {
        "aurora_10": "AURORA-CSC-uniform-10",
        "qd": "HC-CSC-uniform",
        "taxons_10": "TAXONS-10",
        "ns": "NS",
        "qd_no_sel": "Random Search",
    }

    for key in dict_algorithm_folder_to_title:
        dict_algorithm_folder_to_title[key] = dict_algorithm_folder_to_title[key].replace("_", "\_")


    # Adding lines
    x0 = 0
    y0 = 1.2

    # x0, y0 = fig.transFigure.inverted().transform((x0,y0))
    x1, y1 = axes[0,1].transAxes.transform((1, y0))
    # x1, y1 = fig.transFigure.inverted().transform((x1,y1))
    x1, y1 = axes[0,0].transAxes.inverted().transform((x1,y1))
    print(x1,y1)
    line = lines.Line2D([x0, x1], [y0, y1], lw=1., color='silver', alpha=1, linestyle='--', transform=axes[0, 0].transAxes, clip_on=False, in_layout=True, marker=3)
    # axes[0, 0].add_line(line)
    fig.add_artist(line)

    import matplotlib.text as text
    new_text = text.Annotation(r"\textit{Unsupervised BDs}", xy=((x0 + x1) / 2, y0), xytext=(0, 12),
                               xycoords=axes[0, 0].transAxes, textcoords='offset points',
                               size='medium', ha='center', va='top', clip_on=False, in_layout=True)

    fig.add_artist(new_text)

    x0 = 0
    y0 = 1.2

    # x0, y0 = fig.transFigure.inverted().transform((x0,y0))
    x1, y1 = axes[0,4].transAxes.transform((1, y0))
    # x1, y1 = fig.transFigure.inverted().transform((x1,y1))
    x1, y1 = axes[0,2].transAxes.inverted().transform((x1,y1))
    print(x1,y1)
    line = lines.Line2D([x0, x1], [y0, y1], lw=1., color='silver', alpha=1, linestyle='--', transform=axes[0, 2].transAxes, clip_on=False, in_layout=True, marker=3)
    # axes[0, 0].add_line(line)
    fig.add_artist(line)

    new_text = text.Annotation(r"\textit{Hand-coded BDs}", xy=((x0 + x1) / 2, y0), xytext=(0, 12),
                               xycoords=axes[0, 2].transAxes, textcoords='offset points',
                               size='medium', ha='center', va='top', clip_on=False, in_layout=True)
    fig.add_artist(new_text)

    # line.set_clip_on(False)

    # axes[0, 0].add_line(line)
    # ---------------


    for row in range(len(list_experiments)):
        experiment = list_experiments[row]

        list_all_path_proj_files = [
            get_last_gen_proj_file(os.path.join(current_folder_path, dict_experiment_to_folder[experiment], algorithm_folder))
            for algorithm_folder in list_algorithms_folders
        ]

        vmin, vmax = get_min_max_fitness_values(list_all_path_proj_files)

        for col in range(len(list_algorithms_folders)):
            algorithm_folder = list_algorithms_folders[col]
            folder_proj_file = os.path.join(current_folder_path, dict_experiment_to_folder[experiment], algorithm_folder)

            path_proj_file = get_last_gen_proj_file(folder_proj_file)
            print(path_proj_file)
            # axes[row, col].get_legend().remove()

            # axes[row, col].grid()

            # ax.set_xlabel('$x$')
            # axes[row, col].set_xlabel(dict_algorithm_folder_to_title[algorithm_folder])
            # axes[row, col].set_ylabel(dict_experiment_to_title[experiment])

            axes[row, col].spines["top"].set_visible(False)
            axes[row, col].spines["bottom"].set_visible(True)
            axes[row, col].spines["right"].set_visible(False)
            axes[row, col].spines["left"].set_visible(True)

            axes[row, col].tick_params(axis='both',          # changes apply to the x-axis
                                       which='both',      # both major and minor ticks are affected
                                       width=1, length=1)

            if col > 0:
                axes[row, col].yaxis.label.set_visible(False)

                axes[row, col].tick_params(axis='y',          # changes apply to the x-axis
                                           which='both',      # both major and minor ticks are affected
                                           right=False,      # ticks along the bottom edge are off
                                           left=True,         # ticks along the top edge are off
                                           labelleft=False, width=1, length=1)
            if row <= 1:
                axes[row, col].tick_params(axis='x',          # changes apply to the x-axis
                                           which='both',      # both major and minor ticks are affected
                                           bottom=True,      # ticks along the bottom edge are off
                                           top=False,         # ticks along the top edge are off
                                           labelbottom=True, width=1, length=1)
                axes[row, col].xaxis.label.set_visible(False)

            # axes[row, col].xaxis.label.set_visible(False)

            axes[row, col].set_axisbelow(True)

            if row == 0:
                axes[row, col].set_title(dict_algorithm_folder_to_title[algorithm_folder])

            if experiment == "maze":
                plot_comparison_maze(axes[row, col], path_proj_file, vmin, vmax=0)
            elif experiment == "hexapod_camera_vertical":
                plot_comparison_hexapod(axes[row, col], path_proj_file, vmin, vmax=0)
            elif experiment == "air_hockey":
                plot_comparison_air_hockey(axes[row, col], path_proj_file, vmin, vmax=0)



    pad = 10 # in points
    for ax, row in zip(axes[:,0], range(len(list_experiments))):
        ax.annotate(dict_experiment_to_title[list_experiments[row]], xy=(0, 0.5), xytext=(-35, 0),
                    xycoords="axes fraction", textcoords='offset points',
                    size='medium', ha='right', va='center', rotation=90)


    # def fmt(x, pos):
    #     a, b = '{:.2e}'.format(x).split('e')
    #     b = int(b)
    #     return r'${} \times 10^{{{}}}$'.format(a, b)

    list_cbar = []


    for i in range(3):
        cbar = plt.colorbar(mappable=axes[i,0].collections[0], ax=axes[i,:], aspect=10)
        cbar.ax.tick_params(axis='both',          # changes apply to the x-axis
                            which='both',      # both major and minor ticks are affected
                            width=0.5, length=3,
                            direction="in")
        list_cbar.append(cbar)
    list_cbar[0].ax.set_title("Fitness")


    # list_cbar[0].formatter.set_powerlimits((0, 0))
    # list_cbar[0].update_ticks()

    # ax.set_xlabel('$x$')
    # plt.grid()
    # plt.tight_layout()

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

import os
import random
import sys

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

sys.path.append('/Users/looka/git/sferes2/exp/aurora/submodules/figure_generator')
sys.path.append('/Users/looka/git/sferes2/exp/aurora/')

sys.path.append('/git/sferes2/exp/aurora/submodules/figure_generator')
sys.path.append('/git/sferes2/exp/aurora/')

import analysis.paper.plot_utils as pu

from analysis.paper.p_01_comparison_variants.e_02_scatter_plots_variants import get_last_gen_proj_file, get_args

from data_reader import convert_to_rgb, get_data_proj


def create_scatter_plot(ax,
                        plot_component: np.ndarray,
                        color_component: np.ndarray,
                        indexes_plot_component: tuple,
                        indexes_color_component: tuple,
                        indexes_subset_points: np.ndarray = None,
                        gray = False,
                        ):
    assert len(indexes_plot_component) in (2, 3)
    assert len(indexes_color_component) in (2, 3)

    rgb_color_component, list_str_colors = convert_to_rgb(color_component, indexes_color_component)

    print(rgb_color_component)

    if indexes_subset_points is not None:
        plot_component = plot_component[indexes_subset_points, :]
        rgb_color_component = rgb_color_component[indexes_subset_points, :]
    if gray:
        ax.scatter(plot_component[:, indexes_plot_component[0]],
                   plot_component[:, indexes_plot_component[1]],
                   c="lightgray", s=50, marker="o")
    else:
        ax.scatter(plot_component[:, indexes_plot_component[0]],
                   plot_component[:, indexes_plot_component[1]],
                   c=rgb_color_component, s=300, marker="o", edgecolors=None, linewidths=0.2)


def generate_figure(path_save: str):
    plt.clf()
    plt.cla()
    plt.close()

    matplotlib.rc_file_defaults()

    pu.figure_setup()

    fig_size = pu.get_fig_size(10, 6)
    # fig = plt.figure(figsize=fig_size)
    fig, axes = plt.subplots(1, 1, constrained_layout=True)
    fig.set_size_inches(*fig_size)

    current_folder_path = "/Users/looka/git/sferes2/exp/aurora/analysis/paper/p_02_analysis_learned_behavioural_space/"

    # ax = fig.add_subplot(131)
    # ax2 = fig.add_subplot(132)
    # ax3 = fig.add_subplot(133)

    # ax.plot(x, y, c='b', lw=pu.plot_lw())

    list_experiments = [
        # "maze",
        "hexapod",
        # "air_hockey",
    ]

    dict_experiment_to_title = {
        # "maze": "Maze",
        "hexapod": "Hexapod",
        # "air_hockey": "Air-Hockey",
    }

    dict_experiment_to_folder = {
        "maze": "../p_02_analysis_learned_behavioural_space/results_dim_2/maze/retrain",
        "hexapod": "../p_02_analysis_learned_behavioural_space/results_dim_2/hexapod/",
        "air_hockey": "../p_02_analysis_learned_behavioural_space/results_dim_2/air_hockey/",
    }

    # name_folder -> title

    # print("load dataframe")
    # df = pd.read_csv("/Users/looka/git/sferes2/exp/aurora/analysis/paper/dataframes/df.csv")
    # print("finished loading dataframe")

    experiment = "hexapod"
    folder_proj_file = os.path.join(current_folder_path, dict_experiment_to_folder[experiment])
    path_proj_file = get_last_gen_proj_file(folder_proj_file)
    array_latent_space, array_gt_positions, array_fitness_values, array_novelty = get_data_proj(path_proj_file)

    seed = 51
    np.random.seed(seed)
    random.seed(seed)

    number_samples = 400
    other_samples = 50

    indexes = [40, 2525, 2574, 3790]

    indexes_subset_points = np.random.choice(array_latent_space.shape[0], number_samples, replace=False)
    # index_other_points = np.array(indexes)
    index_other_points = np.array(indexes)
    col = 1
    create_scatter_plot(axes, array_latent_space, array_latent_space, (0, 1), (0, 1), indexes_subset_points, gray=True)
    print("index_other_points", index_other_points)
    create_scatter_plot(axes, array_latent_space, array_latent_space, (0, 1), (0, 1), index_other_points, gray=False)
    # axes.get_legend().remove()

    # axes.grid()

    # ax.set_xlabel('$x$')
    # if row == 0 and col == 0:
    #     axes.set_ylabel("\\textbf{Task Behavioural Descriptor}\n$y_T$")
    # elif row == 0:
    #     axes.xaxis.label.set_visible(True)
    #     axes.yaxis.label.set_visible(True)
    # elif col > 0:
    #     axes.set_xlabel("Latent dimension $1$")
    #     axes.set_ylabel("Latent dimension $2$")


    # axes.set_xlabel('Number of dimensions')
    # axes.xaxis.set_ticks(np.arange(2, 20+1, 2))
    # axes.xaxis.set_major_formatter(ticker.FormatStrFormatter('%d'))
    axes.set_axisbelow(True)

    axes.spines["top"].set_visible(False)
    axes.spines["bottom"].set_visible(False)
    axes.spines["right"].set_visible(False)
    axes.spines["left"].set_visible(False)
    axes.tick_params(axis='both',          # changes apply to the x-axis
                     which='both',      # both major and minor ticks are affected
                     bottom=False,      # ticks along the bottom edge are off
                     top=False,         # ticks along the top edge are off
                     labelbottom=False, width=1, length=1, labelleft=False, left=False)
    # if col == 0:
    #     axes.set_title(f"\\textbf{{Task Behavioural}}\n\\textbf{{Descriptor}}")

    # elif col == 1:
    #     axes.set_title(f"\\textbf{{Latent Space}}")
    #     axes.set_xlabel("Latent dimension 1")
    #     axes.set_ylabel("Latent dimension 2")
    # ---------------

    # ax.set_xlabel('$x$')

    # handles, labels = axes[2, 2].get_legend_handles_labels()
    # fig.legend(handles, labels, bbox_to_anchor=(0.5, -0.02), ncol=2, loc="upper center")
    # plt.grid()
    # plt.tight_layout()

    if path_save:
        pu.save_fig(fig, path_save, transparent=True)
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

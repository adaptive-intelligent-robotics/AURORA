import os
import sys

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import lines
from matplotlib.transforms import IdentityTransform


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
                        ):
    assert len(indexes_plot_component) in (2, 3)
    assert len(indexes_color_component) in (2, 3)

    rgb_color_component, list_str_colors = convert_to_rgb(color_component, indexes_color_component)

    print(rgb_color_component)

    ax.scatter(plot_component[:, indexes_plot_component[0]],
               plot_component[:, indexes_plot_component[1]],
               c=rgb_color_component, s=1, marker=".",
               rasterized=True)


def generate_figure(path_save: str):
    plt.clf()
    plt.cla()
    plt.close()

    matplotlib.rc_file_defaults()

    pu.figure_setup()
    pu.set_font_size(new_font_size=12)

    fig_size = pu.get_fig_size(22, 11)
    # fig = plt.figure(figsize=fig_size)
    fig, axes = plt.subplots(2, 4, constrained_layout=True)
    fig.set_size_inches(*fig_size)

    current_folder_path = "/Users/looka/git/sferes2/exp/aurora/analysis/paper/p_02_analysis_learned_behavioural_space/"

    # ax = fig.add_subplot(131)
    # ax2 = fig.add_subplot(132)
    # ax3 = fig.add_subplot(133)

    # ax.plot(x, y, c='b', lw=pu.plot_lw())

    list_experiments = [
        "hexapod",
        "air_hockey",
        "maze",
        "maze_retrain",
    ]

    dict_experiment_to_title = {
        "maze": "Maze",
        "hexapod": "Hexapod",
        "air_hockey": "Air-Hockey",
        "maze_retrain": "Maze (re-train)",
    }

    dict_experiment_to_folder = {
        "maze": "results_dim_2/maze/",
        "hexapod": "results_dim_2/hexapod/",
        "air_hockey": "results_dim_2/air_hockey/",
        "maze_retrain": "results_dim_2/maze/retrain/",
    }


    # name_folder -> title

    # print("load dataframe")
    # df = pd.read_csv("/Users/looka/git/sferes2/exp/aurora/analysis/paper/dataframes/df.csv")
    # print("finished loading dataframe")

    for row in range(2):
        for col in range(len(list_experiments)):
            experiment = list_experiments[col]
            folder_proj_file = os.path.join(current_folder_path, dict_experiment_to_folder[experiment])
            path_proj_file = get_last_gen_proj_file(folder_proj_file)
            array_latent_space, array_gt_positions, _, _ = get_data_proj(path_proj_file)

            if row == 0:

                indexes = np.arange(np.size(array_gt_positions, axis=0))
                np.random.shuffle(indexes)
                array_gt_positions = array_gt_positions[indexes]
                array_latent_space = array_latent_space[indexes]

                if experiment.startswith("maze"):
                    array_gt_positions[:, 1] = 600 - array_gt_positions[:, 1]
                if experiment == "hexapod":
                    new_array = np.zeros_like(array_gt_positions)
                    new_array[:, 1] = array_gt_positions[:, 0]
                    new_array[:, 0] = array_gt_positions[:, 1]
                    array_gt_positions = new_array


                create_scatter_plot(axes[row, col], array_gt_positions, array_latent_space, (0, 1), (0, 1))
                axes[row, col].set_xlabel(r"$x_T$")
                axes[row, col].set_ylabel(r"$y_T$")



            elif row == 1:
                create_scatter_plot(axes[row, col], array_latent_space, array_latent_space, (0, 1), (0, 1))
            # axes[row, col].get_legend().remove()

            # axes[row, col].grid()

            # ax.set_xlabel('$x$')
            if row == 0 and col == 0:
                axes[row, col].set_ylabel("$y_T$")
                axes[row, col].annotate("\\textbf{Task BDs} ($\\boldsymbol{b_{\\mathcal{B}_\\mathcal{T}}}$)", xy=(0, 0.5), xytext=(-45, 0),
                                        xycoords="axes fraction", textcoords='offset points',
                                        size='medium', ha='right', va='center', rotation=90)
            elif row == 1 and col == 0:
                axes[row, col].annotate("\\textbf{Unsupervised BDs}", xy=(0, 0.5), xytext=(-45, 0),
                                        xycoords="axes fraction", textcoords='offset points',
                                        size='medium', ha='right', va='center', rotation=90)

                axes[row, col].set_xlabel("Latent dimension $1$")
                axes[row, col].set_ylabel("Latent dimension $2$")

            elif row == 0:
                axes[row, col].xaxis.label.set_visible(True)
                axes[row, col].yaxis.label.set_visible(False)
            elif col > 0:
                axes[row, col].set_xlabel("Latent dimension $1$")
                axes[row, col].set_ylabel("Latent dimension $2$")
                axes[row, col].yaxis.label.set_visible(False)



        # axes[row, col].set_xlabel('Number of dimensions')
            # axes[row, col].xaxis.set_ticks(np.arange(2, 20+1, 2))
            # axes[row, col].xaxis.set_major_formatter(ticker.FormatStrFormatter('%d'))
            axes[row, col].set_axisbelow(True)

            axes[row, col].spines["top"].set_visible(False)
            # axes[row, col].spines["bottom"].set_visible(False)
            axes[row, col].spines["right"].set_visible(False)
            # axes[row, col].spines["left"].set_visible(False)

            if row == 0:
                axes[row, col].set_title(dict_experiment_to_title[list_experiments[col]])

            pu.archive_ticks_params(axes[row, col])
    # ---------------

    # ax.set_xlabel('$x$')

    # handles, labels = axes[2, 2].get_legend_handles_labels()
    # fig.legend(handles, labels, bbox_to_anchor=(0.5, -0.02), ncol=2, loc="upper center")
    # plt.grid()
    plt.tight_layout()

    # Adding lines
    x0 = 1
    y0 = 1

    # x0, y0 = fig.transFigure.inverted().transform((x0,y0))
    x0, y0 = (0.75 * np.array(axes[0, 2].transAxes.transform((1, 1.15))) + 0.25 * np.array(axes[0, 3].transAxes.transform((0, 1.15))))
    x1, y1 = (0.75 * np.array(axes[1, 2].transAxes.transform((1, -0.3))) + 0.25 * np.array(axes[1, 3].transAxes.transform((0, -0.3))))
    # x1, y1 = fig.transFigure.inverted().transform((x1,y1))
    x1, y1 = fig.transFigure.inverted().transform((x1, y1))
    x0, y0 = fig.transFigure.inverted().transform((x0, y0))
    print(x0,x1)
    print(y0,y1)
    line = lines.Line2D([x0, x1], [y0, y1], lw=1., color='black', alpha=1, linestyle='-', transform=fig.transFigure, clip_on=False, in_layout=True)
    # axes[0, 0].add_line(line)
    fig.add_artist(line)

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

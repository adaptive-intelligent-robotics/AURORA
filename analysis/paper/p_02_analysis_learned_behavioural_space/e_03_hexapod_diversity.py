import os
import sys

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import gridspec, lines
from matplotlib.colors import ListedColormap
from matplotlib.patches import Polygon
import matplotlib.image as mpimg

import pandas as pd
import seaborn as sns


sys.path.append('/Users/looka/git/sferes2/exp/aurora/submodules/figure_generator')
sys.path.append('/Users/looka/git/sferes2/exp/aurora/')

sys.path.append('/git/sferes2/exp/aurora/submodules/figure_generator')
sys.path.append('/git/sferes2/exp/aurora/')


from analysis.metrics.hexapod_camera_vertical import MetricsHexapodCameraVertical
from analysis.paper import plot_utils, latent_space_dim_comparison

import analysis.paper.plot_utils as pu

from analysis.paper.p_01_comparison_variants.e_02_scatter_plots_variants import get_last_gen_proj_file, get_args

from data_reader import convert_to_rgb, get_data_proj


def adjust_lightness(color, amount=0.5):
    # https://stackoverflow.com/questions/37765197/darken-or-lighten-a-color-in-matplotlib
    import matplotlib.colors as mc
    import colorsys
    try:
        c = mc.cnames[color]
    except:
        c = color
    c = colorsys.rgb_to_hls(*mc.to_rgb(c))
    return colorsys.hls_to_rgb(c[0], max(0, min(1, amount * c[1])), c[2])


def create_scatter_plot(ax1,
                        plot_component: np.ndarray,
                        color_component: np.ndarray,
                        indexes_plot_component: tuple,
                        indexes_color_component: tuple,
                        center=None,
                        numpy_max=None,
                        rank_based=False,
                        max_l=None,
                        ):
    assert len(indexes_plot_component) in (2, 3)
    assert len(indexes_color_component) in (2, 3)

    rgb_color_component, list_str_colors = convert_to_rgb(color_component, indexes_color_component, center=center, numpy_max=numpy_max, rank_based_coloring=rank_based, max_l=max_l)

    # list_str_colors = [
    #     f'rgb({rgb_color_component[i, 0]}, '
    #     f'{rgb_color_component[i, 1]}, '
    #     f'{rgb_color_component[i, 2]})' for i
    #     in range(rgb_color_component.shape[0])]

    # color_component = np.hstack((
    #     plot_component[:, indexes_color_component[0]].reshape(-1, 1),
    #     plot_component[:, indexes_color_component[1]].reshape(-1, 1)
    # ))
    #
    # rgb_color_component = np.asarray((color_component + np.pi) / (2 * np.pi), dtype=np.float)
    # print(np.zeros((np.size(rgb_color_component, axis=0), 1)))
    # rgb_color_component = np.hstack((rgb_color_component, np.zeros((np.size(rgb_color_component, axis=0), 1))))
    #
    #
    # print(rgb_color_component)
    print(indexes_color_component)
    ax1.scatter(plot_component[:, indexes_plot_component[0]],
                plot_component[:, indexes_plot_component[1]],
                c=rgb_color_component, s=3, marker="o",
                rasterized=True)


def plot_illustration(ax):
    img = mpimg.imread('/Users/looka/git/sferes2/exp/aurora/analysis/paper/'
                       'p_02_analysis_learned_behavioural_space/data_obs_hexapod/illu_hexa.png')
    ax.imshow(img)

    ax.spines["top"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)

    ax.tick_params(axis='y',          # changes apply to the x-axis
                   which='both',      # both major and minor ticks are affected
                   right=False,      # ticks along the bottom edge are off
                   left=False,         # ticks along the top edge are off
                   labelleft=False)
    ax.tick_params(axis='x',          # changes apply to the x-axis
                   which='both',      # both major and minor ticks are affected
                   top=False,      # ticks along the bottom edge are off
                   bottom=False,         # ticks along the top edge are off
                   labelbottom=False)


def plot_comparison_evolution(ax, aurora_angle_coverage, hand_coded_angle_coverage, fig):
    df = pd.read_csv("/Users/looka/git/sferes2/exp/aurora/analysis/paper/p_02_analysis_learned_behavioural_space/data_obs_hexapod/df_hexapod_camera_vertical_processed.csv")
    order_legend = [
        "qd_uniform_psat",

        *[
            f"aurora_uniform_{dim}_psat"
            for dim in [2, 3, 5, 10]
        ],
    ]

    dict_order_replacement = {
        **{
            f"aurora_uniform_{dim}_psat": str(dim)
            for dim in [3, 5, 10]
        },
        "aurora_uniform_2_psat": "$n$=2",
        "qd_uniform_psat": "HC",
    }

    df = df[df["name_variant"].isin(order_legend)]

    for current_name_variant, updated_name_variant in dict_order_replacement.items():
        df["name_variant"] = df["name_variant"].replace(current_name_variant, updated_name_variant, regex=True)

    # add escape character for latex rendering
    for i in range(len(order_legend)):
        order_legend[i] = dict_order_replacement[order_legend[i]]
        order_legend[i] = order_legend[i].replace('_', '\_')

    df["name_variant"] = df["name_variant"].replace('_', '\_', regex=True)

    # y_lim = (0., 1.)
    y_lim = None
    environment = "hexapod_camera_vertical"

    # Color palette chosen
    my_cmap = ListedColormap(sns.color_palette("colorblind").as_hex())
    list_colors = [
        my_cmap.colors[index_color] for index_color in [4, 0]
    ]
    list_colors = list_colors[::-1]

    df = df[df["gen"] == 15000]
    data = [
        df[df["name_variant"] == legend]["angle_coverage"]
        for legend in order_legend
    ]
    print(data[0])
    bplot = ax.boxplot(data,
               labels=order_legend,
               # notch=True,
               patch_artist=True) # fill with color

    ax.tick_params(axis='x', which='major', labelsize=10)
    for index, patch in enumerate(bplot["boxes"]):
        if index == 0:
            patch.set_facecolor(list_colors[1])
        else:
            patch.set_facecolor(adjust_lightness(list_colors[0], amount=2.5 - 1.75 * index / 4))

    ax.set_ylim((0.02, 1))
    ax.set_yscale("log")


    # medians = latent_space_dim_comparison.LatentSpaceDimComparison.get_comparison_chosen_variants_custom_scripts(
    #     folder_path_save=None,
    #     df=df,
    #     environment=environment,
    #     y="angle_coverage",
    #     y_lim=y_lim,
    #     name_file=None,
    #     ax=ax,
    #     hue_order=order_legend,
    #     list_colors=list_colors,
    #     alpha_median=1.,
    #     linewidth_median=2.,
    # )
    ax.grid()
    ax.set_title("Final Orientation Coverage")

    # print(medians["AURORA-CSC-uniform-20"].iloc[-1])
    print("aurora_angle_coverage", aurora_angle_coverage)
    ax.annotate("$\\bigstar$", xy=(5.40, aurora_angle_coverage), xycoords="data", horizontalalignment="center", verticalalignment="center")
    ax.annotate("$\\blacksquare$", xy=(1.45, hand_coded_angle_coverage), xycoords="data", horizontalalignment="center", verticalalignment="center")

    line = lines.Line2D([1.6, 5.4], [0.012, 0.012], lw=1., color='silver', alpha=1, linestyle='--', transform=ax.transData, clip_on=False, in_layout=True)
    plt.annotate("AURORA-CSC-uniform-$n$", xy=(3.5, 0.0080), xycoords=ax.transData, horizontalalignment="center", verticalalignment="bottom",)

    # axes[0, 0].add_line(line)
    fig.add_artist(line)

    handles, labels = ax.get_legend_handles_labels()
    # ax.set_xticklabels(order_legend, rotation=70, horizontalalignment="right")
    # ax.legend(handles, labels, bbox_to_anchor=(-0.02, 1.02), ncol=1, loc="upper left", prop={'size': 8})



def plot_scatter_orientations(ax1, ax2, array_gt_positions, array_gt_positions_2):
    create_scatter_plot(ax1, array_gt_positions, array_gt_positions, (3, 4), (0, 1),
                        rank_based=True)
    create_scatter_plot(ax2, array_gt_positions_2, array_gt_positions_2, (3, 4), (0, 1),
                        rank_based=True)

    for ax in (ax1, ax2):
        ax.add_patch(Polygon([[-np.pi, -np.pi / 2], [-np.pi, np.pi / 2], [0, np.pi / 2], [0, - np.pi / 2]],
                                      closed=True,
                                      fill=False,
                                      hatch='/',
                                      zorder=-1,
                                      alpha=1,
                                      color="silver")
                              )
        ax.add_patch(Polygon([[0, np.pi / 2], [0, 3 * np.pi / 2], [np.pi, 3 * np.pi / 2], [np.pi, np.pi / 2]],
                                      closed=True,
                                      fill=False,
                                      hatch='/',
                                      zorder=-1,
                                      alpha=1,
                                      color="silver"))
        ax.set_xticks([-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi])
        ax.set_xticklabels([r"$-\pi$", r"$-\pi / 2$", "0", r"$\pi / 2$", "$\pi$"])


        ax.set_yticks([-np.pi / 2, 0, np.pi / 2, np.pi, 3 * np.pi / 2])
        ax.set_yticklabels([r"$-\frac{\pi}{2}$", "0", r"$\frac{\pi}{2}$", "$\pi$", r"$\frac{3\pi}{2}$"])

        for i in range(2):
            ax.set_xlim((-np.pi, np.pi))
            ax.set_ylim((-np.pi / 2, 3 * np.pi / 2))



        ax.set_axisbelow(True)

        ax.spines["top"].set_visible(False)
        # axes[col].spines["bottom"].set_visible(False)
        ax.spines["right"].set_visible(False)
        # axes[col].spines["left"].set_visible(False)

    ax1.set_title(None)
    ax1.set_xlabel("Pitch (rad)")
    ax1.set_ylabel("Roll (rad)")

    ax2.set_title(None)
    ax2.set_xlabel("Pitch (rad)")
    ax2.set_ylabel("Roll (rad)")

    plot_utils.archive_ticks_params(ax1)
    plot_utils.archive_ticks_params(ax2)

    ax2.yaxis.label.set_visible(False)
    ax2.tick_params(axis='y',          # changes apply to the x-axis
                    which='both',      # both major and minor ticks are affected
                    right=False,      # ticks along the bottom edge are off
                    left=True,         # ticks along the top edge are off
                    labelleft=False, width=1, length=1)



def generate_figure(path_save: str):
    plt.clf()
    plt.cla()
    plt.close()

    matplotlib.rc_file_defaults()

    pu.figure_setup()
    pu.set_font_size(new_font_size=12)

    plt.rcParams.update({
        'axes.labelsize':12,
    })
    fig_size = pu.get_fig_size(22, 12)
    # fig = plt.figure(figsize=fig_size)
    fig = plt.figure()
    # fig, axes = plt.subplots(2, 3, constrained_layout=True)
    fig.set_size_inches(*fig_size)

    current_folder_path = "/Users/looka/git/sferes2/exp/aurora/analysis/paper/p_02_analysis_learned_behavioural_space/"


    # path_aurora = "data_obs_hexapod/aurora_20/"
    path_aurora = "data_obs_hexapod/aurora_10/"
    path_hand_coded = "../p_01_comparison_variants/data_archives/hexapod/qd/"

    # name_folder -> title

    # print("load dataframe")
    # df = pd.read_csv("/Users/looka/git/sferes2/exp/aurora/analysis/paper/dataframes/df.csv")
    # print("finished loading dataframe")
    dummy = np.random.rand(10000, 2) * 2 * np.pi - np.pi

    experiment = "hexapod"
    folder_proj_file_1 = os.path.join(current_folder_path, path_aurora)
    folder_proj_file_2 = os.path.join(current_folder_path, path_hand_coded)
    path_proj_file = get_last_gen_proj_file(folder_proj_file_1)
    path_proj_file_2 = get_last_gen_proj_file(folder_proj_file_2)
    _, array_gt_positions, _, _ = get_data_proj(path_proj_file)

    aurora_angle_coverage = 2 * MetricsHexapodCameraVertical.calculate_angle_coverage(array_gt_positions[:, 3:5], nb_div=16)
    # print("BOUH ANGLE COVERAGE", 2 * MetricsHexapodCameraVertical.calculate_angle_coverage(array_gt_positions[:, 3:5], nb_div=10))


    _, array_gt_positions_2, _, _ = get_data_proj(path_proj_file_2)
    hand_coded_angle_coverage = 2 * MetricsHexapodCameraVertical.calculate_angle_coverage(array_gt_positions_2[:, 3:5], nb_div=16)


    print(array_gt_positions[array_gt_positions[:, 0] < -np.pi / 2])
    if experiment == "maze":
        array_gt_positions[:, 1] = 600 - array_gt_positions[:, 1]
    if experiment == "hexapod":
        new_array = np.zeros_like(array_gt_positions)
        new_array[:, 1] = array_gt_positions[:, 0]
        new_array[:, 0] = array_gt_positions[:, 1]
        new_array[:, 2:] = array_gt_positions[:, 2:]
        indexes = np.arange(np.size(new_array, axis=0))
        np.random.shuffle(indexes)

        array_gt_positions = new_array[indexes, :]

        new_array = np.zeros_like(array_gt_positions_2)
        new_array[:, 1] = array_gt_positions_2[:, 0]
        new_array[:, 0] = array_gt_positions_2[:, 1]
        new_array[:, 2:] = array_gt_positions_2[:, 2:]
        indexes = np.arange(np.size(new_array, axis=0))
        np.random.shuffle(indexes)
        array_gt_positions_2 = new_array[indexes, :]

        array_gt_positions[array_gt_positions[:, 4] < -np.pi / 2, 4] += 2 * np.pi
        array_gt_positions_2[array_gt_positions_2[:, 4] < -np.pi / 2, 4] += 2 * np.pi
        print(array_gt_positions[array_gt_positions[:, 3] < -np.pi / 2][:, 3])


    # ---------------

    gs1 = gridspec.GridSpec(2, 1)
    ax0 = fig.add_subplot(gs1[0])
    ax1 = fig.add_subplot(gs1[1])

    plot_illustration(ax0)


    plot_comparison_evolution(ax1, aurora_angle_coverage, hand_coded_angle_coverage, fig)


    gs1.tight_layout(fig, rect=[0, 0, 0.333, 1.])

    gs2 = gridspec.GridSpec(1, 2)
    gs2_ax1 = fig.add_subplot(gs2[0])
    gs2_ax2 = fig.add_subplot(gs2[1])




    plot_scatter_orientations(gs2_ax1, gs2_ax2, array_gt_positions, array_gt_positions_2)

    gs2.tight_layout(fig, rect=[0.333, 0.25, 1, 0.75])

    plt.text(0., 0.55, '\\textbf{Cov}', ha='left', transform=fig.transFigure, fontsize=12)

    plt.text(0., 1., '\\textbf{Illu}', ha='left', transform=fig.transFigure, fontsize=12)
    plt.text(0.35, 1., "\\textbf{AU}", ha='left', transform=fig.transFigure, fontsize=12)
    plt.text(0.7, 1., "\\textbf{HC}", ha='left', transform=fig.transFigure, fontsize=12)

    # plt.tight_layout()

    x0, _ = gs2_ax1.transAxes.transform((0.5, 1.))
    _, y0 = fig.transFigure.transform((0.5, 0.95))

    x0, y0 = fig.transFigure.inverted().transform((x0, y0))

    plt.annotate("AURORA-CSC-uniform-10 $\\bigstar$", xy=(x0, y0), xycoords=fig.transFigure, ha="center")

    x0, _ = gs2_ax2.transAxes.transform((0.5, 1.))
    _, y0 = fig.transFigure.transform((0.5, 0.95))

    x0, y0 = fig.transFigure.inverted().transform((x0, y0))

    plt.annotate("HC-CSC-uniform $\\blacksquare$", xy=(x0, y0), xycoords=fig.transFigure, ha="center")


    # x0, y0 = fig.transFigure.inverted().transform((x0,y0))
    # x0, y0 = (0.75 * np.array(axes[0, 2].transAxes.transform((1, 1.15))) + 0.25 * np.array(axes[0, 3].transAxes.transform((0, 1.15))))
    # x1, y1 = (0.75 * np.array(axes[1, 2].transAxes.transform((1, -0.3))) + 0.25 * np.array(axes[1, 3].transAxes.transform((0, -0.3))))
    # x1, y1 = fig.transFigure.inverted().transform((x1,y1))
    # x1, y1 = fig.transFigure.inverted().transform((x1, y1))
    # x0, y0 = fig.transFigure.inverted().transform((x0, y0))
    # print(x0,x1)
    # print(y0,y1)
    # line = lines.Line2D([x0, x1], [y0, y1], lw=1., color='black', alpha=1, linestyle='-', transform=fig.transFigure, clip_on=False, in_layout=True)
    # axes[0, 0].add_line(line)
    # fig.add_artist(line)


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

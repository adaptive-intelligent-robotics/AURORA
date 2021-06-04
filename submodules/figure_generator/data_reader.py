import os
from enum import Enum
from typing import List, Union, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import seaborn as sns

from airl_2D_colormap import AIRL_ColorMap


class OptionsColor(Enum):
    COLORMAP = 0
    RGB = 1


class Components(Enum):
    GROUND_TRUTH = 0
    LATENT_COMPONENT = 1
    VALUE_FITNESS = 2
    VALUE_FITNESS_IMPLICIT = 3
    NOVELTY = 4


def read_data(file_path: str,
              list_names_components: List[str],
              convert_to_float_numpy_array: bool = True
              ) -> dict:
    dict_data_per_component = {name_data: [] for name_data in list_names_components}

    with open(file_path, 'r') as file:
        for line in file.readlines():
            list_different_components = filter(bool, line.split('  '))
            list_different_components = list(map(str.strip, list_different_components))
            list_different_components = [component.split(' ') for component in list_different_components]

            for index_component, name_component in enumerate(list_names_components):
                dict_data_per_component[name_component].append(list_different_components[index_component])

    try:
        if convert_to_float_numpy_array:
            for name_component in list_names_components:
                dict_data_per_component[name_component] = np.asarray(dict_data_per_component[name_component],
                                                                     dtype=np.float)
    except:
        dict_data_per_component = {name_data: np.asarray([]).reshape(0,len(dict_data_per_component[name_component][0])) for name_data in list_names_components}
    return dict_data_per_component


def project_axes(array: np.array, index_axes: Union[int, np.array]) -> np.array:
    if isinstance(index_axes, int):
        return array[:, index_axes]
    elif isinstance(index_axes, np.ndarray):
        return array.T[index_axes].T


def pca():
    # TODO
    pass


def save_figure(figure,
                name_without_format: str,
                list_formats: List[str]
                ) -> None:
    for _format in list_formats:
        figure.savefig(f"{name_without_format}.{_format}", format=_format)


def get_colors_array(option_color: OptionsColor,
                     ground_truth_component: np.ndarray,
                     indexes=None,
                     ) -> np.ndarray:
    if not indexes:
        indexes = [0, 1, 2]
    if option_color == OptionsColor.COLORMAP:
        if ground_truth_component.shape[1] == 2:
            return AIRL_ColorMap.get_2D_colormap(ground_truth_component)
        elif ground_truth_component.shape[1] > 2:
            return AIRL_ColorMap.get_2D_colormap(ground_truth_component[:, np.array([indexes[0], indexes[1]])])
    elif option_color == OptionsColor.RGB:
        if ground_truth_component.shape[1] == 3:
            return convert_to_rgb(ground_truth_component)
        elif ground_truth_component.shape[1] > 3:
            return convert_to_rgb(ground_truth_component[:, np.array(indexes)])
    else:
        raise ValueError("option_color is supposed to be an instance of OptionsColor")


def _create_joint_plot(name_saved_plot: str,
                       data_frame,
                       list_formats: List[str],
                       show_plot: bool = False,
                       do_put_title: bool = False
                       ):
    sns.set(style="white")

    hexplot = sns.jointplot("x", "y", data=data_frame, kind="hex", joint_kws={})
    plt.subplots_adjust(right=0.8, left=0.2, top=0.8, bottom=0.2)  # shrink fig so cbar is visible
    # make new ax object for the cbar
    cbar_ax = hexplot.fig.add_axes([0.85, .25, .05, .4])  # x, y, width, height
    plt.colorbar(cax=cbar_ax)

    figure = hexplot.fig

    if show_plot:
        plt.show()

    if do_put_title:
        figure.suptitle(name_saved_plot)

    save_figure(figure, name_saved_plot, list_formats)  # TODO
    figure.clear()
    plt.cla()
    plt.clf()
    plt.close()


def _create_color_plot(path_saved_plot: str,
                       data_frame: pd.DataFrame,
                       ground_truth_component: np.ndarray,
                       option_color: OptionsColor,
                       list_formats: List[str],
                       show_plot: bool = False,
                       do_put_title: bool = False,
                       indexes: tuple = None,
                       ):
    if not indexes:
        indexes = (0, 1)
    sns.set(style="darkgrid")
    color = sns.color_palette()[5]
    g = sns.jointplot("x", "y", data=data_frame, kind="reg", stat_func=None,
                      color='k', height=7, xlim=(-np.pi - 0.1, np.pi + 0.1), ylim=(-np.pi - 0.1, np.pi + 0.1))
    g.ax_joint.cla()
    colors_array = get_colors_array(option_color, ground_truth_component, indexes=indexes)
    if np.max(colors_array) > 2:
        colors_array = colors_array / 255.
    for i, row in enumerate(data_frame.values):
        plt.plot(row[0], row[1], color=colors_array[i], marker='o')
    g.set_axis_labels('x', 'y', fontsize=16)
    if show_plot:
        plt.show()

    fig = g.fig

    if do_put_title:
        fig.suptitle(path_saved_plot)

    save_figure(fig, path_saved_plot, list_formats)  # TODO
    fig.clear()
    plt.cla()
    plt.clf()
    plt.close()


def convert_to_rgb(dataset: np.ndarray, indexes_color_component, center=None, numpy_max=None, rank_based_coloring=True, max_l=None):
    assert len(indexes_color_component) in (2, 3)

    if len(indexes_color_component) == 2:
        rgb_color_component = AIRL_ColorMap.get_2D_colormap(dataset[:, np.array(list(indexes_color_component))], center=center, numpy_max=numpy_max, rank_based_coloring=rank_based_coloring, max_l=max_l)
    elif len(indexes_color_component) == 3:
        sub_component = dataset[:, np.array(list(indexes_color_component))]

        rgb_max = np.max(sub_component, axis=0)
        rgb_min = np.min(sub_component, axis=0)
        rgb_color_component = np.asarray(255 * (sub_component - rgb_min) / (rgb_max - rgb_min), dtype=np.int)
    else:
        raise ValueError

    list_str_colors = [
        f'rgb({rgb_color_component[i, 0]}, '
        f'{rgb_color_component[i, 1]}, '
        f'{rgb_color_component[i, 2]})' for i
        in range(rgb_color_component.shape[0])]

    return rgb_color_component, list_str_colors


def create_html_plot(path_saved_plot,
                     plot_component: np.ndarray,
                     color_component: np.ndarray,
                     indexes_plot_component: tuple,
                     indexes_color_component: tuple,
                     compare_component:np.ndarray=None,
                     latent_component:np.ndarray=None,
                     compare_latent_component: np.ndarray=None,
                     star=False,
                     added_metric_component=None
                     ):
    assert len(indexes_plot_component) in (2, 3)
    assert len(indexes_color_component) in (2, 3)

    rgb_color_component, list_str_colors = convert_to_rgb(color_component, indexes_color_component)

    if added_metric_component is None:
        dict_marker = dict(
            size=7,
            color=list_str_colors,  # set color to an array/list of desired values
            opacity=1
        )
    else:
        dict_marker = dict(
            size=7,
            color=added_metric_component,  # set color to an array/list of desired values
            opacity=1,
            colorscale='Viridis',
            colorbar=dict(thickness=20),
            cmin=0,
            cmax=400
        )

    if len(indexes_plot_component) == 3:
        fig = go.Figure(data=[go.Scatter3d(
            x=plot_component[:, indexes_plot_component[0]],
            y=plot_component[:, indexes_plot_component[1]],
            z=plot_component[:, indexes_plot_component[2]],
            text=[f'index: {index}' for index in np.arange(np.size(plot_component, axis=0))],
            mode='markers',
            marker=dict_marker
        )])
    elif len(indexes_plot_component) == 2:
        fig = go.Figure(data=[go.Scatter(
            x=plot_component[:, indexes_plot_component[0]],
            y=plot_component[:, indexes_plot_component[1]],
            text=[f'index: {index}' for index in np.arange(np.size(plot_component, axis=0))],
            mode='markers',
            marker=dict_marker
        )])
    else:
        raise ValueError
    if latent_component is not None:
        d = {}
        d_compare = {}
        X,Y,Z,color = [], [], [], []
        for i in range(np.size(latent_component, axis=0)):
            index_closer = np.argmin(np.linalg.norm(compare_latent_component - latent_component[i, :], axis=1), axis=0).item()
            # compare_point = compare_component[index_closer, :]
            # component_point = plot_component[index_closer, :]
            if index_closer not in d:
                d[index_closer] = [plot_component[i, :]]
            else:
                d[index_closer].append(plot_component[i, :])
            d_compare[index_closer] = compare_component[index_closer, :]
            # fig.add_shape(
            #     # Line Diagonal
            #     type="line",
            #     x0=compare_point[0],
            #     y0=compare_point[1],
            #     x1=component_point[0],
            #     y1=component_point[1],
            #     line=dict(
            #         color=list_str_colors[index_closer],
            #         width=4,
            #         dash="dashdot",
            #     )
            # )
        for index_closer, list_neighbours in d.items():
            if len(list_neighbours) >= 20:
                for neighbour in list_neighbours:
                    X.append(neighbour[indexes_plot_component[0]])
                    Y.append(neighbour[indexes_plot_component[1]])
                    Z.append(neighbour[indexes_plot_component[2]])
                    color.append(str(index_closer))
                    if star:
                        X.append(d_compare[index_closer][indexes_plot_component[0]])
                        Y.append(d_compare[index_closer][indexes_plot_component[1]])
                        Z.append(d_compare[index_closer][indexes_plot_component[2]])
                        color.append(str(index_closer))
        df = pd.DataFrame(dict(X=X, Y=Y, Z=Z, color=color))
        import plotly.express as px
        fig = px.line_3d(df, x='X', y='Y', z='Z', color="color")
        # fig.update_shapes(dict(xref='x', yref='y'))
    try:
        print(f"Saving new figure there: {path_saved_plot}.html")
        fig.write_html(f"{path_saved_plot}.html")
        print("Saving Succeeded")
    except:
        print(f"Saving of {path_saved_plot} failed, maybe the file already exists")


def get_data_proj(file_path_projected_archive: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    try:
        dict_data_per_component = read_data(file_path_projected_archive,
                                            ["_1",
                                             "_2",
                                             Components.LATENT_COMPONENT,
                                             Components.GROUND_TRUTH,
                                             Components.VALUE_FITNESS,
                                             Components.VALUE_FITNESS_IMPLICIT,
                                             Components.NOVELTY,
                                             ])
        implicit_fitness_component = dict_data_per_component[Components.VALUE_FITNESS_IMPLICIT]  # TODO
        novelty_component = dict_data_per_component[Components.NOVELTY]

    except:
        try:
            dict_data_per_component = read_data(file_path_projected_archive,
                                                ["_1",
                                                 "_2",
                                                 Components.LATENT_COMPONENT,
                                                 Components.GROUND_TRUTH,
                                                 Components.VALUE_FITNESS,
                                                 Components.NOVELTY,
                                                 ])
            novelty_component = dict_data_per_component[Components.NOVELTY]

            implicit_fitness_component = np.zeros_like(novelty_component)
        except:
            try:
                dict_data_per_component = read_data(file_path_projected_archive,
                                                    ["_1",
                                                     "_2",
                                                     Components.LATENT_COMPONENT,
                                                     Components.GROUND_TRUTH,
                                                     Components.VALUE_FITNESS,
                                                     ])
                value_fitness_component = dict_data_per_component[Components.VALUE_FITNESS]

                implicit_fitness_component = np.zeros_like(value_fitness_component)
                novelty_component = np.zeros_like(value_fitness_component)
            except:
                # print("[ERROR] PROBLEM IN PROJ FILE: ", file_path_projected_archive)
                raise

    latent_component = dict_data_per_component[Components.LATENT_COMPONENT]
    ground_truth_component = dict_data_per_component[Components.GROUND_TRUTH]
    value_fitness_component = dict_data_per_component[Components.VALUE_FITNESS]

    return latent_component, \
           ground_truth_component, \
           implicit_fitness_component, \
           novelty_component



def get_data_offspring(file_path_projected_archive: str) -> Tuple[np.ndarray, np.ndarray]:
    dict_data_per_component = read_data(file_path_projected_archive,
                                        ["_1",
                                         "_2",
                                         Components.LATENT_COMPONENT,
                                         Components.GROUND_TRUTH])
    latent_component = dict_data_per_component[Components.LATENT_COMPONENT]
    ground_truth_component = dict_data_per_component[Components.GROUND_TRUTH]
    return latent_component, \
           ground_truth_component


def get_data_modifier(file_path_projected_archive: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    dict_data_per_component = read_data(file_path_projected_archive,
                                        ["1"])
    gen_component = dict_data_per_component['1'][:, 0]
    l_component = dict_data_per_component['1'][:, 1]
    pop_size_component = dict_data_per_component['1'][:, 2]

    gen_component_without_duplicates = gen_component
    l_component_without_duplicates = l_component
    pop_size_component_without_duplicates = pop_size_component

    for gen in set(gen_component.flatten()):
        index_gen = np.where(gen_component_without_duplicates == gen)[0]
        if index_gen.size > 1:
            gen_component_without_duplicates = np.delete(gen_component_without_duplicates, index_gen[:-1].astype(int), axis=0)
            l_component_without_duplicates = np.delete(l_component_without_duplicates, index_gen[:-1].astype(int), axis=0)
            pop_size_component_without_duplicates = np.delete(pop_size_component_without_duplicates, index_gen[:-1].astype(int), axis=0)

    return gen_component_without_duplicates, l_component_without_duplicates, pop_size_component_without_duplicates


def get_data_frame(component, dim):
    if dim == 2:
        return pd.DataFrame(component, columns=["x", "y"])
    elif dim == 3:
        return pd.DataFrame(component, columns=["x", "y", "z"])
    else:
        return pd.DataFrame(component)


def create_archive_figures(file_path_projected_archive: str,
                           path_save: str,
                           generation: int,
                           list_formats: List[str],
                           do_put_title: bool = False,
                           show_plots: bool = False,
                           exp: str = '',
                           ) -> None:
    if exp:
        prefix_exp = f"{exp}_"
    else:
        prefix_exp = ""

    latent_component, ground_truth_component, *_ = get_data_proj(file_path_projected_archive)
    # ground_truth_component = project_axes(ground_truth_component, np.array([0, 1]))
    # ground_truth_component = np.hstack((ground_truth_component, np.random.rand(ground_truth_component.shape[0]).reshape((-1, 1))))

    if ground_truth_component.shape[1] == 2:
        df_latent = get_data_frame(latent_component, dim=2)
        df_ground_truth = get_data_frame(ground_truth_component, dim=2)

        _create_color_plot(path_saved_plot=os.path.join(path_save, f"archive_desc_gen_{generation:07}"),
                           data_frame=df_latent,
                           latent_component=latent_component,
                           ground_truth_component=ground_truth_component,
                           option_color=OptionsColor.COLORMAP,
                           list_formats=list_formats,
                           show_plot=show_plots,
                           do_put_title=do_put_title)

        _create_color_plot(path_saved_plot=os.path.join(path_save, f"archive_gt_gen_{generation:07}"),
                           data_frame=df_ground_truth,
                           latent_component=latent_component,
                           ground_truth_component=ground_truth_component,
                           option_color=OptionsColor.COLORMAP,
                           list_formats=list_formats,
                           show_plot=show_plots,
                           do_put_title=do_put_title)

        _create_joint_plot(name_saved_plot=os.path.join(path_save, f"archive_desc_joint_plot_gen_{generation:07}"),
                           data_frame=df_latent,
                           list_formats=list_formats,
                           show_plot=show_plots,
                           do_put_title=do_put_title)

        _create_joint_plot(
            name_saved_plot=os.path.join(path_save, f"archive_ground_truth_joint_plot_gen_{generation:07}"),
            data_frame=df_ground_truth,
            list_formats=list_formats,
            show_plot=show_plots,
            do_put_title=do_put_title)

    elif ground_truth_component.shape[1] >= 3:
        print(latent_component)
        # reducer = umap.UMAP(n_components=3)
        # embedding = reducer.fit_transform(latent_component)

        create_html_plot(path_saved_plot=os.path.join(path_save, f"{prefix_exp}gt_rot_gen_{generation:07}"),
                         plot_component=ground_truth_component,
                         color_component=ground_truth_component,
                         indexes_plot_component=(3, 4, 5),
                         indexes_color_component=(0, 1, 2),
                         )

        create_html_plot(path_saved_plot=os.path.join(path_save, f"{prefix_exp}gt_rot_gen_{generation:07}_comp"),
                         plot_component=ground_truth_component,
                         color_component=ground_truth_component,
                         indexes_plot_component=(0, 1, 2),
                         indexes_color_component=(0, 1, 2),
                         )

        create_html_plot(path_saved_plot=os.path.join(path_save, f"{prefix_exp}gt_pos_gen_{generation:07}"),
                         plot_component=ground_truth_component,
                         color_component=ground_truth_component,
                         indexes_plot_component=(0, 1, 2),
                         indexes_color_component=(3, 4, 5),
                         )

        create_html_plot(path_saved_plot=os.path.join(path_save, f"{prefix_exp}gt_pos_gen_{generation:07}_comp"),
                         plot_component=ground_truth_component,
                         color_component=ground_truth_component,
                         indexes_plot_component=(3, 4, 5),
                         indexes_color_component=(3, 4, 5),
                         )

        create_html_plot(path_saved_plot=os.path.join(path_save, f"{prefix_exp}gt_pos_gen_{generation:07}_test"),
                         plot_component=ground_truth_component,
                         color_component=ground_truth_component,
                         indexes_plot_component=(3, 4),
                         indexes_color_component=(0, 1),
                         )

        # create_html_plot(path_saved_plot=os.path.join(path_save, f"{prefix_exp}latent_gen_{generation:07}"),
        #                  plot_component=embedding,
        #                  color_component=ground_truth_component,
        #                  indexes_plot_component=(0, 1, 2),
        #                  indexes_color_component=(3, 4),
        #                  )


if __name__ == '__main__':
    # create_archive_figures("example/proj_1900.dat",
    #                        "images/",
    #                        generation=1900,
    #                        list_formats=["png"])
    latent_aurora, gt_aurora, *_ = get_data_proj('/Users/looka/git/sferes2/exp/aurora/submodules/figure_generator/example/other/proj_5000_aurora.dat')
    latent_ns, gt_ns, *_ = get_data_proj('/Users/looka/git/sferes2/exp/aurora/submodules/figure_generator/example/other/proj_5000_ns.dat')
    create_html_plot("/Users/looka/git/sferes2/exp/aurora/submodules/figure_generator/example/other/plot_rot",
                     plot_component=gt_ns,
                     color_component=gt_ns,
                     indexes_plot_component=(3, 4, 5),
                     indexes_color_component=(3, 4, 5),
                     latent_component=latent_ns,
                     compare_latent_component=latent_aurora,
                     compare_component=gt_aurora,
                     )

    create_html_plot("/Users/looka/git/sferes2/exp/aurora/submodules/figure_generator/example/other/plot_pos",
                     plot_component=gt_ns,
                     color_component=gt_ns,
                     indexes_plot_component=(0, 1, 2),
                     indexes_color_component=(0, 1, 2),
                     latent_component=latent_ns,
                     compare_latent_component=latent_aurora,
                     compare_component=gt_aurora,
                     star=True,
                     )

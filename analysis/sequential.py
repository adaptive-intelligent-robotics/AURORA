import glob
import os
import sys
import traceback
from typing import Tuple

import cv2
import jinja2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import k_means

sys.path.append('/git/sferes2/exp/aurora/submodules/figure_generator')
sys.path.append('/git/sferes2/exp/aurora/')

from mosaic import Mosaic

from data_reader import create_html_plot, read_data, get_data_proj
from analysis.metrics.hexapod_walls import MetricsWalls
from analysis.metrics.maze import MetricsMaze
from singularity.experiment import *

def read_data_sequential_obs(file_observation: str) -> Tuple[np.ndarray, np.ndarray]:
    dict_data_per_component = read_data(file_observation,
                                        ["1", "2"])
    pop_index = dict_data_per_component['1'][:, 0]
    sequential_conponent = dict_data_per_component['2']
    return pop_index, sequential_conponent

def write_graphs_sequences(path_folder_load, path_folder_save, gen, size_one_input: int):
    NUMBER_K_MEANS = 20
    PERIOD_CONSIDER_MEAN = 2

    path_observation = os.path.join(path_folder_load, f"observation_gen_{gen:07}.dat")
    path_reconstruction = os.path.join(path_folder_load, f"reconstruction_obs_gen_{gen:07}.dat")
    path_proj = os.path.join(path_folder_load, f"proj_{gen}.dat")

    if (not os.path.exists(path_observation)) \
            or (not os.path.exists(path_reconstruction)) \
            or (not os.path.exists(path_proj)):
        return

    pop_index, obs_component  = read_data_sequential_obs(path_observation)
    _, reconst_component = read_data_sequential_obs(path_reconstruction)

    latent_component, ground_truth_component, *_ = get_data_proj(path_proj)
    array_means = k_means(latent_component, NUMBER_K_MEANS)[0]

    list_indexes = []
    for mean in array_means:
        temp = np.linalg.norm(latent_component - mean, axis=1)
        list_indexes.append(np.argmin(temp))

    for index in list_indexes[::PERIOD_CONSIDER_MEAN]:
        save_figures_index(size_one_input,
                           obs_component,
                           reconst_component,
                           index,
                           pop_index,
                           gen,
                           path_folder_save)

def write_graphs_sequences_indexes(path_folder_load, path_folder_save, gen, size_one_input: int, indexes_indiv_show, len_running_mean=None, unique_offset=None):

    path_observation = os.path.join(path_folder_load, f"observation_gen_{gen:07}.dat")
    path_reconstruction = os.path.join(path_folder_load, f"reconstruction_obs_gen_{gen:07}.dat")
    # path_reconstruction = os.path.join(path_folder_load, f"reconstruction_obs_gen_{gen:07}_use_cell_state.dat")
    # path_observation = os.path.join(path_folder_load, f"scaled_data.dat")
    # path_reconstruction = os.path.join(path_folder_load, f"scaled_reconstruction.dat")

    if (not os.path.exists(path_observation)) \
            or (not os.path.exists(path_reconstruction)):
        return

    pop_index, obs_component  = read_data_sequential_obs(path_observation)
    _, reconst_component = read_data_sequential_obs(path_reconstruction)

    if len_running_mean:
        new_component = obs_component.copy()
        for i in range(len_running_mean, np.size(new_component, axis=0)):
            new_component[i, :] = obs_component[i - len_running_mean:i, :].mean(axis=0)
        obs_component = new_component[len_running_mean:,:].copy()


    for index in indexes_indiv_show:
        save_figures_index(size_one_input,
                            obs_component,
                            reconst_component,
                            index,
                            pop_index,
                            gen,
                            path_folder_save,
                           unique_offset=unique_offset)

def save_figures_index(size_one_input: int,
                       obs_component,
                       reconst_component,
                       index,
                       pop_index,
                       gen,
                       path_folder_save,
                       unique_offset=None,
                       ):

    fig, axs = plt.subplots(size_one_input)

    # reconst_component = reconst_component[1: np.size(reconst_component, 0),:] # TODO: To remove
    print(np.size(obs_component, axis=0), np.size(reconst_component, axis=0))
    assert (np.size(obs_component, axis=0) == np.size(reconst_component, axis=0))
    # assert (np.size(obs_component, axis=1) == np.size(reconst_component, axis=1))
    assert (index in pop_index)

    index_row_pop_array = np.where(index == pop_index.flatten())[0][0]

    if unique_offset is None:
        list_offsets = list(range(size_one_input))
    else:
        list_offsets = [unique_offset]
    for index_offset, offset in enumerate(list_offsets):
        list_one_light_obs = obs_component[index_row_pop_array, offset::size_one_input]
        list_one_light_reconst = reconst_component[index_row_pop_array][offset::size_one_input]
        number_one_light_obs = len(list_one_light_obs)
        number_one_light_obs_reconst = len(list_one_light_reconst)

        axs[offset].plot(np.linspace(0, 2000, number_one_light_obs), list_one_light_obs)
        axs[offset].plot(np.linspace(0, 2000, number_one_light_obs_reconst), list_one_light_reconst)
        axs[offset].set_ylim(0, 600)
        if index_offset == 0:
            position_sensor = "Left"
        elif index_offset == 1:
            position_sensor = "Middle"
        elif index_offset == 2:
            position_sensor = "Right"
        axs[offset].set_ylabel(f"{position_sensor} sensor")
        if index_offset == 2:
            axs[offset].set_xlabel(f"time-step")

    plt.tight_layout()
    plt.savefig(os.path.join(path_folder_save, f"sequence_{gen:07}_{index:06}.png"))
    plt.close()

if __name__ == '__main__':
    write_graphs_sequences_indexes("/Volumes/home/raw_results/2020-08-16_12_38_56_15_l0X_conv_seq_ae/",
                           "./RPC/",
                           gen=20000,
                           size_one_input=3,
                           indexes_indiv_show=range(0, 4500, 250),
                           len_running_mean=None,
                           unique_offset=None)

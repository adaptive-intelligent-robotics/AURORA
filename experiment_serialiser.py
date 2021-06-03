

import argparse
import pickle

import os.path
import sys

# Adding Module paths (to take additional waf_tools from subdirs into account)

sys.path.append('/Users/looka/git/sferes2/exp/aurora/submodules/figure_generator')
sys.path.append('/Users/looka/git/sferes2/exp/aurora/')

sys.path.append('/git/sferes2/exp/aurora/submodules/figure_generator')
sys.path.append('/git/sferes2/exp/aurora/')

import singularity.collections_experiments.maze as maze
import singularity.collections_experiments.hexapod_camera_vertical as hexapod_camera_vertical
import singularity.collections_experiments.air_hockey as air_hockey
import singularity.factory_experiments as factory_experiments

NAME_FILE_SERIALISED_LIST_EXPERIMENTS = "experiments.pkl"


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--chosen", required=False, type=str, default="all")
    parser.add_argument("--container", required=True, type=str)
    return parser.parse_args()


def get_chosen_experiments(str_chosen_experiments):
    list_experiments_all = factory_experiments.get_all_experiments_from_dict(factory_experiments.DICT_ALL_EXPERIMENTS)

    if str_chosen_experiments in factory_experiments.DICT_ALL_EXPERIMENTS:
        print(f'Building {str_chosen_experiments} experiments')
        chosen_experiments = factory_experiments.DICT_ALL_EXPERIMENTS[str_chosen_experiments]
    elif str_chosen_experiments.lower() == 'all':
        print(f'Building All experiments')
        chosen_experiments = list_experiments_all
    else:
        print(f'No chosen experiments to build! -> Building all experiments')
        chosen_experiments = list_experiments_all
    return chosen_experiments


def serialise_experiments(list_chosen_experiments, path):
    for experiment in list_chosen_experiments:
        print(experiment.get_exec_name(add_arguments=True))

    with open(path, "wb") as file_serialised_exp:
        pickle.dump(list_chosen_experiments, file_serialised_exp)


def load_serialised_experiments(path):
    with open(path, "rb") as file_serialised_exp:
        list_chosen_experiments = pickle.load(file_serialised_exp)

    print("Loaded Experiments:")
    for experiment in list_chosen_experiments:
        print(experiment.get_exec_name(add_arguments=True))

    return list_chosen_experiments


def main():
    args = get_args()

    str_chosen_experiments = args.chosen
    str_container = args.container

    path = os.path.abspath(
        os.path.join("./",
                     str_container[:-4],
                     NAME_FILE_SERIALISED_LIST_EXPERIMENTS)
    )

    list_chosen_experiments = get_chosen_experiments(str_chosen_experiments)
    serialise_experiments(list_chosen_experiments, path)


if __name__ == '__main__':
    main()

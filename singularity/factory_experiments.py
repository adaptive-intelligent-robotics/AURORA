from singularity.collections_experiments import maze, \
    hexapod_camera_vertical, \
    air_hockey

import singularity.experiment as experiment


DICT_ALL_EXPERIMENTS = {
    experiment.STR_MAZE: maze.LIST_EXPERIMENTS,

    experiment.STR_HEXAPOD_CAMERA_VERTICAL: hexapod_camera_vertical.LIST_EXPERIMENTS,

    experiment.STR_AIR_HOCKEY: air_hockey.LIST_EXPERIMENTS,
}

DICT_EXPERIMENTS_TO_LAUNCH = {
    experiment.STR_MAZE: maze.LIST_EXPERIMENTS,
    experiment.STR_HEXAPOD_CAMERA_VERTICAL: hexapod_camera_vertical.LIST_EXPERIMENTS,
    experiment.STR_AIR_HOCKEY: air_hockey.LIST_EXPERIMENTS
}

DICT_NUMBER_RESTARTS_PER_EXPERIMENT = {
    experiment.STR_MAZE: 2,

    experiment.STR_HEXAPOD_CAMERA_VERTICAL: 6,

    experiment.STR_AIR_HOCKEY: 1,
}


def get_all_experiments_from_dict(dict_experiments):
    list_experiments = []
    for _dict_list_exp in dict_experiments.values():
        list_experiments.extend(_dict_list_exp)
    return list_experiments


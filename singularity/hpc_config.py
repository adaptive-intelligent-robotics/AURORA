import itertools
import json
import sys
import argparse
import os

from singularity.experiment import STR_HEXAPOD_CAMERA_VERTICAL, DICT_EXPERIMENT_TO_CONFIG
from singularity.factory_experiments import DICT_ALL_EXPERIMENTS, DICT_NUMBER_RESTARTS_PER_EXPERIMENT

sys.path.append('/git/sferes2/exp/aurora/')

NUMBER_CORES_HPC = 32
NUMBER_CORES_HPC_GPU = 4

NAME_JSON = "hpc_config.json"


def create_json(path_folder_json: str) -> None:

    for name_exp, list_experiments in DICT_ALL_EXPERIMENTS.items():

        hpc_config_exp = {
            "wall_time": DICT_EXPERIMENT_TO_CONFIG[name_exp].walltime,

            "nb_cores": f"{DICT_EXPERIMENT_TO_CONFIG[name_exp].number_cpus_hpc_node}",
            "mem": f"{DICT_EXPERIMENT_TO_CONFIG[name_exp].memory}gb",

            "nb_runs": f"{DICT_EXPERIMENT_TO_CONFIG[name_exp].nb_runs}",
            "vnc": "yes",
            "apps": [""],

            "analysis": "analysis",
            "wall_time_analysis": "23:59:00",
            "nb_cores_analysis": "32",
            "mem_analysis": "62gb",

            # "sync": {
            #     "period_seconds": "3600",
            #     "interrupt_main": "yes"
            # },

            "identifier_json": name_exp,

            "args": [
                x.get_exec_name(add_arguments=True) for x in list_experiments
            ],

            "resume_args": [
                "" for _ in range(DICT_NUMBER_RESTARTS_PER_EXPERIMENT[name_exp])
            ]
        }

        if name_exp == STR_HEXAPOD_CAMERA_VERTICAL:
            hpc_config_exp.update({})
            # hpc_config_exp.update({
            #     "wall_time": "71:59:00",
            # })

        with open(os.path.join(path_folder_json, f'hpc_config_{name_exp}.json'), "w") as f:
            json.dump(hpc_config_exp, f, indent=1)


if __name__ == '__main__':
    ALL = 'all'

    parser = argparse.ArgumentParser()
    parser.add_argument('--path-folder-json', type=str, required=True)

    arg = parser.parse_args()
    if not os.path.exists(os.path.abspath(arg.path_folder_json)):
        raise FileNotFoundError()

    create_json(path_folder_json=arg.path_folder_json)

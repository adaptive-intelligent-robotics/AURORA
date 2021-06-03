import sys

from operator import itemgetter

sys.path.append('/git/sferes2/exp/aurora/')

import singularity.resources.fix_build as fix_build


class Config(object):
    def __init__(self, number_cpus_hpc_node, number_cpus_to_use, memory, walltime="23:59:00", nb_runs=5):
        self.number_cpus_hpc_node = number_cpus_hpc_node
        self.number_cpus_to_use = number_cpus_to_use
        self.memory = memory
        self.walltime = walltime
        self.nb_runs = nb_runs


class SpecificConfig(object):
    THROUGHPUT = Config(number_cpus_hpc_node=8, number_cpus_to_use=8, memory=96, walltime="5:59:00")
    GENERAL = Config(number_cpus_hpc_node=32, number_cpus_to_use=30, memory=62, walltime="23:59:00")
    SINGLENODE = Config(number_cpus_hpc_node=48, number_cpus_to_use=46, memory=124, walltime="23:59:00")

    @classmethod
    def get_thoughput(cls, walltime="23:59:00", nb_runs=5):
        return Config(number_cpus_hpc_node=8, number_cpus_to_use=8, memory=96, walltime=walltime, nb_runs=nb_runs)

    @classmethod
    def get_general(cls, walltime="23:59:00", nb_runs=5):
        return Config(number_cpus_hpc_node=32, number_cpus_to_use=30, memory=62, walltime=walltime, nb_runs=nb_runs)

    @classmethod
    def get_singlenode(cls, walltime="23:59:00", nb_runs=5):
        return Config(number_cpus_hpc_node=48, number_cpus_to_use=46, memory=124, walltime=walltime, nb_runs=nb_runs)


# NUMBER_CORES = 28
# NUMBER_CORES_HPC = 32

LIST_UPDATE_CONTAINER_PERIOD = [1, 5, 10, 20, 50]
LIST_COEFFICIENT_PROPORTIONAL_CONTROL_L = [5e-8, 5e-6, 5e-4, 5e-2]

STR_MAZE = "maze"

STR_HEXAPOD_CAMERA_VERTICAL = "hexapod_camera_vertical"

STR_AIR_HOCKEY = "air_hockey"

DICT_EXPERIMENT_TO_CONFIG = {
    # launched
    STR_MAZE: SpecificConfig.get_general(nb_runs=20),
    STR_HEXAPOD_CAMERA_VERTICAL: SpecificConfig.get_general(nb_runs=20),
    STR_AIR_HOCKEY: SpecificConfig.get_general(nb_runs=20),
}


class EncoderType:
    lstm_ae = "\"lstm_ae\""
    cnn_ae = "\"cnn_ae\""
    strg_cnn = "\"strg_cnn\""
    conv_seq_ae = "\"conv_seq_ae\""
    mlp_ae = "\"mlp_ae\""
    pca = "\"pca\""
    none = "\"none\""


class Experiment(object):
    def __init__(self,
                 name_exec: str,
                 algo: str,
                 env: str,
                 latent_space: int,
                 use_colors: bool,
                 arguments: str,
                 fixed_l: float = None,
                 use_videos: bool = False,
                 encoder_type = EncoderType.none,
                 lstm_latent_size_per_layer: int = None,
                 lstm_number_layers: int = None,
                 do_consider_bumpers: bool = None,
                 lp_norm: float = 2,
                 has_fit: bool = False,
                 # adapts distance threshold using original AURORA's volume estimation of the convex hull of the BD
                 use_volume_adaptive_threshold: bool = False,
                 # Normalisation of the latent space projections before setting them as BD
                 no_normalisation: bool = True,
                 taxons_elitism: bool = False,
                 update_container_period: int = None,
                 coefficient_proportional_control_l: float = None,
                 ):

        self._exec = name_exec
        self._algo = algo
        self._env = env
        self._latent_space = latent_space
        self._use_colors = use_colors
        self._arguments = arguments
        self._fixed_l = fixed_l
        self._use_videos = use_videos
        self._encoder_type = encoder_type
        self._lp_norm = lp_norm
        self._has_fit = has_fit
        self._use_volume_adaptive_threshold = use_volume_adaptive_threshold
        self._no_normalisation = no_normalisation
        self._taxons_elitism = taxons_elitism
        self._update_container_period = update_container_period
        self._coefficient_proportional_control_l = coefficient_proportional_control_l

        if lstm_latent_size_per_layer:
            assert lstm_number_layers
            assert lstm_latent_size_per_layer == latent_space
            assert do_consider_bumpers is not None

            self._lstm_number_layers = lstm_number_layers
            self._lstm_latent_size_per_layer = lstm_latent_size_per_layer
            self._do_consider_bumpers = do_consider_bumpers
        elif lstm_number_layers:
            assert lstm_latent_size_per_layer
            assert do_consider_bumpers is not None
            self._lstm_number_layers = lstm_number_layers
            self._lstm_latent_size_per_layer = lstm_latent_size_per_layer
            self._do_consider_bumpers = do_consider_bumpers
        else:
            self._lstm_number_layers = None
            self._lstm_latent_size_per_layer = None
            self._do_consider_bumpers = None

    @property
    def exec(self):
        return self._exec

    @property
    def algo(self):
        return self._algo

    @property
    def env(self):
        return self._env

    @property
    def latent_space(self):
        return self._latent_space

    @property
    def use_colors(self):
        return self._use_colors

    @property
    def use_videos(self):
        return self._use_videos

    @property
    def arguments(self):
        return self._arguments

    @property
    def fixed_l(self):
        return self._fixed_l

    @property
    def encoder_type(self):
        return self._encoder_type

    @property
    def lstm_number_layers(self):
        return self._lstm_number_layers

    @property
    def lstm_latent_size_per_layer(self):
        return self._lstm_latent_size_per_layer

    @property
    def do_consider_bumpers(self):
        return self._do_consider_bumpers

    @property
    def lp_norm(self):
        return self._lp_norm

    @property
    def has_fit(self):
        return self._has_fit

    @property
    def use_volume_adaptive_threshold(self):
        return self._use_volume_adaptive_threshold

    @property
    def no_normalisation(self):
        return self._no_normalisation

    @property
    def taxons_elitism(self):
        return self._taxons_elitism

    @property
    def update_container_period(self):
        return self._update_container_period

    @property
    def coefficient_proportional_control_l(self):
        return self._coefficient_proportional_control_l

    def get_exec_name(self, add_arguments=False):
        name = self.get_str_variables_run_experiment()
        name = name.lower().replace(' ', '_')
        name = fix_build.get_fixed_name(name)
        if not add_arguments:
            return f'{self.exec}_{name}'
        else:
            return f'{self.exec}_{name} {self.arguments}'

    def get_results_folder_name(self):
        return f"results_{self.get_exec_name(add_arguments=False)}"

    def get_str_variables_run_experiment(self):
        list_variables = [
            f'ENVIRONMENT="{self.env}"',
            f'ALGORITHM="{self.algo}"',
            f'LATENT_SPACE_SIZE={self.latent_space}',
        ]
        if self.encoder_type is not None:
            list_variables.append(f"ENCODER_TYPE={self.encoder_type}")
        if self.use_colors:
            list_variables.append("USE_COLORS")
        if self.use_videos:
            list_variables.append("USE_VIDEOS")
        if self.fixed_l is not None:
            list_variables.append(f"PARAMS_FIXED_L={self.fixed_l}")
        if self.lstm_latent_size_per_layer:
            list_variables.append(f"LSTM_LATENT_SIZE_PER_LAYER={self.lstm_latent_size_per_layer}")
        if self.lstm_number_layers:
            list_variables.append(f"LSTM_NUMBER_LAYERS={self.lstm_number_layers}")
        if self.do_consider_bumpers:
            list_variables.append(f"DO_CONSIDER_BUMPERS_MAZE")
        if self.lp_norm != 2: # LP NORM variable added only if using non-euclidian norm
            if self.lp_norm == float("inf"):
                list_variables.append(f'LP_NORM="{self.lp_norm}"') # if norm-inf, send a string variable
            else:
                list_variables.append(f'LP_NORM={self.lp_norm}')
        if self.has_fit:
            list_variables.append('HAS_FIT')
        if self.use_volume_adaptive_threshold:
            list_variables.append('VAT')
        if self.no_normalisation:
            list_variables.append('NO_NORMALISE')
        if self.taxons_elitism:
            list_variables.append('TAX_ELI')
        if self.coefficient_proportional_control_l is not None:
            list_variables.append(f'ALPHA_L={self.coefficient_proportional_control_l}')
        if self.update_container_period is not None:
            list_variables.append(f'T_UPDATE={self.update_container_period}')
        return ' '.join(list_variables)

    def get_str_variables_project_to_latent_space(self):
        list_variables = [
            f'ENVIRONMENT="{self.env}"',
            f'ALGORITHM="None"',
            f'LATENT_SPACE_SIZE={self.latent_space}',
        ]
        if self.encoder_type is not None:
            list_variables.append(f"ENCODER_TYPE={self.encoder_type}")
        if self.use_colors:
            list_variables.append("USE_COLORS")
        if self.use_videos:
            list_variables.append("USE_VIDEOS")
        if self.lstm_latent_size_per_layer:
            list_variables.append(f"LSTM_LATENT_SIZE_PER_LAYER={self.lstm_latent_size_per_layer}")
        if self.lstm_number_layers:
            list_variables.append(f"LSTM_NUMBER_LAYERS={self.lstm_number_layers}")
        if self.do_consider_bumpers:
            list_variables.append(f"DO_CONSIDER_BUMPERS_MAZE")
        if self.lp_norm != 2: # LP NORM variable added only if using non-euclidian norm
            if self.lp_norm == float("inf"):
                list_variables.append(f'LP_NORM="{self.lp_norm}"') # if norm-inf, send a string variable
            else:
                list_variables.append(f'LP_NORM={self.lp_norm}')
        if self.has_fit:
            list_variables.append('HAS_FIT')
        if self.use_volume_adaptive_threshold:
            list_variables.append('VAT')
        if self.no_normalisation:
            list_variables.append('NO_NORMALISE')
        if self.taxons_elitism:
            list_variables.append('TAX_ELI')
        if self.coefficient_proportional_control_l is not None:
            list_variables.append(f'ALPHA_L={self.coefficient_proportional_control_l}')
        if self.update_container_period is not None:
            list_variables.append(f'T_UPDATE={self.update_container_period}')
        return ' '.join(list_variables)

    def get_exec_command(self):
        return f'{self.get_exec_name()} {self.arguments}'

    def __repr__(self):
        return f'| Attribute | Value |\n' \
               f'| --------- | ----- |\n' \
               + f'\n'.join(f'| {attribute} | {value} |'
                            for attribute, value in vars(self).items()) \
               + '\n' \
               + f'| hash | {self.__hash__()} |'

    def __hash__(self):
        return hash(tuple(map(itemgetter(1), sorted(vars(self).items(), key=itemgetter(0)))))

    def __eq__(self, other):
        if isinstance(other, Experiment):
            return tuple(map(itemgetter(1), sorted(vars(self).items(), key=itemgetter(0)))) == \
                   tuple(map(itemgetter(1), sorted(vars(other).items(), key=itemgetter(0))))
        return NotImplemented

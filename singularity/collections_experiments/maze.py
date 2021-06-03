from singularity.experiment import Experiment, EncoderType, \
    LIST_COEFFICIENT_PROPORTIONAL_CONTROL_L, LIST_UPDATE_CONTAINER_PERIOD, DICT_EXPERIMENT_TO_CONFIG, STR_MAZE

NUMBER_CORES = DICT_EXPERIMENT_TO_CONFIG[STR_MAZE].number_cpus_to_use

UNIFORM = 'uniform'
CURIOSITY = 'curiosity'
NOVELTY = 'novelty'
SURPRISE = 'surprise'
NOVELTY_SURPRISE = 'novelty_surprise'

DICT_SELECTOR_TO_ALGO_AURORA = {
    UNIFORM: 'aurora_uniform',
    CURIOSITY: 'aurora_curiosity',
    NOVELTY: 'aurora_novelty',
    SURPRISE: 'aurora_surprise',
    NOVELTY_SURPRISE: 'aurora_nov_sur',
}

DICT_SELECTOR_TO_ALGO_TAXONS = {
    NOVELTY_SURPRISE: 'taxons',
    NOVELTY: 'taxo_n',
    SURPRISE: 'taxo_s',
}


def get_aurora_n_colors_maze(latent_space,
                             algo,
                             has_fit=False,
                             sticky_walls=False,
                             use_volume_adaptive_threshold=False,
                             no_normalisation=True,
                             update_container_period: int = None,
                             coefficient_proportional_control_l: float = None,
                             ):
    assert algo in DICT_SELECTOR_TO_ALGO_AURORA.values()

    if not sticky_walls:
        environment = 'hard_maze'
    else:
        environment = 'hard_maze_sticky'

    return Experiment(name_exec='aurora',
                      algo=algo,
                      env=environment,
                      latent_space=latent_space,
                      use_colors=True,
                      arguments=f"--number-threads {NUMBER_CORES}",
                      encoder_type=EncoderType.cnn_ae,
                      has_fit=has_fit,
                      use_volume_adaptive_threshold=use_volume_adaptive_threshold,
                      no_normalisation=no_normalisation,
                      update_container_period=update_container_period,
                      coefficient_proportional_control_l=coefficient_proportional_control_l,
                      )


def get_taxons_n_colors_maze(latent_space, algo, sticky_walls=False, taxons_elitism=False,):
    assert algo in DICT_SELECTOR_TO_ALGO_TAXONS.values()

    if not sticky_walls:
        environment = 'hard_maze'
    else:
        environment = 'hard_maze_sticky'

    return Experiment(name_exec='aurora',
                      algo=algo,
                      env=environment,
                      latent_space=latent_space,
                      use_colors=True,
                      arguments=f"--number-threads {NUMBER_CORES}",
                      encoder_type=EncoderType.cnn_ae,
                      taxons_elitism=taxons_elitism,
                      )


def get_hand_coded_qd_maze(fixed_l=None, has_fit=False):
    return Experiment(name_exec='aurora', algo='hand_coded_qd', env='hard_maze', latent_space=2,
                      use_colors=True, arguments=f"--number-threads {NUMBER_CORES}", fixed_l=fixed_l,
                      has_fit=has_fit)


LIST_DIMS_LATENT_SPACES_AURORA = [2, 5, 8, 10, 15, 20]
LIST_DIMS_LATENT_SPACES_TAXONS = [10]

# AURORA in a uniform setting (different latent spaces) # TODO: Do it with official AURORA (curiosity based)

DICT_MAZE_AURORA_UNIFORM_n_COLORS = {
    latent_dim: get_aurora_n_colors_maze(latent_space=latent_dim, algo=DICT_SELECTOR_TO_ALGO_AURORA[UNIFORM], has_fit=True)

    for latent_dim in LIST_DIMS_LATENT_SPACES_AURORA
}

# AURORA VARIANTS (NOVELTY/SURPRISE/UNIFORM/BOTH)
MAZE_AURORA_CURIOSITY_10_COLORS = get_aurora_n_colors_maze(latent_space=10, algo=DICT_SELECTOR_TO_ALGO_AURORA[CURIOSITY], has_fit=True)
MAZE_AURORA_NOVELTY_10_COLORS = get_aurora_n_colors_maze(latent_space=10, algo=DICT_SELECTOR_TO_ALGO_AURORA[NOVELTY], has_fit=True)
MAZE_AURORA_SURPRISE_10_COLORS = get_aurora_n_colors_maze(latent_space=10, algo=DICT_SELECTOR_TO_ALGO_AURORA[SURPRISE], has_fit=True)
MAZE_AURORA_NOVELTY_SURPRISE_10_COLORS = get_aurora_n_colors_maze(latent_space=10, algo=DICT_SELECTOR_TO_ALGO_AURORA[NOVELTY_SURPRISE], has_fit=True)

# AURORA using the ground truth as a behavioural descriptor
# (NOT equivalent to NS - equivalent to standard QD with adaptive distance threshold)
MAZE_AURORA_HARD_CODED_POS = Experiment(name_exec='aurora', algo='hand_coded_qd', env='hard_maze', latent_space=2,
                                        use_colors=True, arguments=f"--number-threads {NUMBER_CORES}", has_fit=True)

MAZE_AURORA_HARD_CODED_GEN_DESC = Experiment(name_exec='aurora', algo='hand_coded_qd', env='hard_maze_gen_desc', latent_space=59,
                                             use_colors=True, arguments=f"--number-threads {NUMBER_CORES}", has_fit=True)

MAZE_AURORA_HARD_CODED_POS_NO_SELECTION = Experiment(name_exec='aurora', algo='hand_coded_qd_no_sel', env='hard_maze', latent_space=2,
                                                     use_colors=True, arguments=f"--number-threads {NUMBER_CORES}", has_fit=True)

MAZE_AURORA_HARD_CODED_POS_L_FIXED_5 = get_hand_coded_qd_maze(fixed_l=5, has_fit=True)

# TAXONS in a classical setting (different latent spaces)

DICT_MAZE_TAXONS_n_COLORS = {
    latent_dim: get_taxons_n_colors_maze(latent_space=latent_dim, algo=DICT_SELECTOR_TO_ALGO_TAXONS[NOVELTY_SURPRISE])

    for latent_dim in LIST_DIMS_LATENT_SPACES_TAXONS
}

# TAXONS VARIANTS (NOVELTY ONLY/SURPRISE ONLY)
MAZE_TAXO_NOVELTY_10_COLORS = get_taxons_n_colors_maze(latent_space=10, algo=DICT_SELECTOR_TO_ALGO_TAXONS[NOVELTY])
MAZE_TAXO_SURPRISE_10_COLORS = get_taxons_n_colors_maze(latent_space=10, algo=DICT_SELECTOR_TO_ALGO_TAXONS[SURPRISE])

# A TAXONS using the ground truth as a behavioural descriptor (should be equivalent to NS without elitism)
MAZE_TAXONS_HARD_CODED_POS = Experiment(name_exec='aurora', algo='hand_coded_taxons', env='hard_maze', latent_space=2,
                                        use_colors=True, arguments=f"--number-threads {NUMBER_CORES}")

# Experiments using Fitness

# # AURORA
# MAZE_AURORA_UNIFORM_10_COLORS_FITNESS = get_aurora_n_colors_maze(latent_space=10, algo=DICT_SELECTOR_TO_ALGO_AURORA[UNIFORM], has_fit=True)

# # Hard-coded
# MAZE_AURORA_HARD_CODED_POS_FITNESS = Experiment(name_exec='aurora', algo='hand_coded_qd', env='hard_maze', latent_space=2,
#                                                 use_colors=True, arguments=f"--number-threads {NUMBER_CORES}", has_fit=True)


# Experiments using Sticky Walls
MAZE_AURORA_UNIFORM_10_COLORS_STICKY = get_aurora_n_colors_maze(latent_space=10, algo=DICT_SELECTOR_TO_ALGO_AURORA[UNIFORM], sticky_walls=True, has_fit=True)

MAZE_AURORA_HARD_CODED_POS_STICKY = Experiment(name_exec='aurora', algo='hand_coded_qd', env='hard_maze_sticky', latent_space=2,
                                               use_colors=True, arguments=f"--number-threads {NUMBER_CORES}", has_fit=True)

MAZE_TAXONS_10_COLORS_STICKY = get_taxons_n_colors_maze(latent_space=10, algo=DICT_SELECTOR_TO_ALGO_TAXONS[NOVELTY_SURPRISE], sticky_walls=True)

MAZE_TAXONS_HARD_CODED_POS_STICKY = Experiment(name_exec='aurora', algo='hand_coded_taxons', env='hard_maze_sticky', latent_space=2,
                                               use_colors=True, arguments=f"--number-threads {NUMBER_CORES}")

# For analysis effect adaptive l compared to fixed l in case of normal QD (with fitness value)
LIST_FIXED_L = [4.00, 4.25, 4.50, 4.75, 5]
DICT_MAZE_AURORA_HARD_CODED_POS_L_FIXED_l_HAS_FIT = {
    l: get_hand_coded_qd_maze(fixed_l=l, has_fit=True)

    for l in LIST_FIXED_L
}

# For analysis effect no Normalisation on Aurora
DICT_MAZE_AURORA_UNIFORM_n_COLORS_NORMALISATION = {
    latent_dim: get_aurora_n_colors_maze(latent_space=latent_dim,
                                         algo=DICT_SELECTOR_TO_ALGO_AURORA[UNIFORM],
                                         no_normalisation=False, has_fit=True)

    for latent_dim in LIST_DIMS_LATENT_SPACES_AURORA
}

# For analysis effect Volume Adaptive Threshold (VAT) on Aurora
DICT_MAZE_AURORA_UNIFORM_n_COLORS_VAT = {
    latent_dim: get_aurora_n_colors_maze(latent_space=latent_dim,
                                         algo=DICT_SELECTOR_TO_ALGO_AURORA[UNIFORM],
                                         use_volume_adaptive_threshold=True, has_fit=True)

    for latent_dim in LIST_DIMS_LATENT_SPACES_AURORA
}

# For analysis effect Volume Adaptive Threshold (VAT) + no Normalisation on Aurora
DICT_MAZE_AURORA_UNIFORM_n_COLORS_VAT_NORMALISATION= {
    latent_dim: get_aurora_n_colors_maze(latent_space=latent_dim,
                                         algo=DICT_SELECTOR_TO_ALGO_AURORA[UNIFORM],
                                         use_volume_adaptive_threshold=True,
                                         no_normalisation=False, has_fit=True)

    for latent_dim in LIST_DIMS_LATENT_SPACES_AURORA
}


# Taxons Elitism variants
MAZE_TAXONS_10_COLORS_ELITISM = get_taxons_n_colors_maze(latent_space=10, algo=DICT_SELECTOR_TO_ALGO_TAXONS[NOVELTY_SURPRISE], taxons_elitism=True)

MAZE_TAXONS_HARD_CODED_POS_ELITISM = Experiment(name_exec='aurora', algo='hand_coded_taxons', env='hard_maze', latent_space=2,
                                        use_colors=True, arguments=f"--number-threads {NUMBER_CORES}", taxons_elitism=True)

MAZE_TAXONS_10_COLORS_STICKY_ELITISM = get_taxons_n_colors_maze(latent_space=10, algo=DICT_SELECTOR_TO_ALGO_TAXONS[NOVELTY_SURPRISE], taxons_elitism=True, sticky_walls=True)

MAZE_TAXONS_HARD_CODED_POS_STICKY_ELITISM = Experiment(name_exec='aurora', algo='hand_coded_taxons', env='hard_maze_sticky', latent_space=2,
                                                       use_colors=True, arguments=f"--number-threads {NUMBER_CORES}", taxons_elitism=True)


# For analysis effect update_container_period
DICT_MAZE_AURORA_UNIFORM_n_COLORS_UPDATE_CONTAINER_PERIOD = {
    update_container_period: get_aurora_n_colors_maze(latent_space=10,
                                                      algo=DICT_SELECTOR_TO_ALGO_AURORA[UNIFORM],
                                                      update_container_period=update_container_period,
                                                      has_fit=True
                                                      )
    for update_container_period in LIST_UPDATE_CONTAINER_PERIOD
}

# For analysis effect coefficient_proportional_control_l
DICT_MAZE_AURORA_UNIFORM_n_COLORS_COEFFICIENT_PROPORTIONAL_CONTROL_L = {
    coefficient_proportional_control_l: get_aurora_n_colors_maze(
        latent_space=10,
        algo=DICT_SELECTOR_TO_ALGO_AURORA[UNIFORM],
        coefficient_proportional_control_l=coefficient_proportional_control_l,
        has_fit=True
    )
    for coefficient_proportional_control_l in LIST_COEFFICIENT_PROPORTIONAL_CONTROL_L
}

LIST_EXPERIMENTS = [
    # AURORA in a Uniform setting (different latent spaces)
    *DICT_MAZE_AURORA_UNIFORM_n_COLORS.values(),

    # AURORA VARIANTS (NOVELTY/SURPRISE/UNIFORM/BOTH)
    # MAZE_AURORA_CURIOSITY_10_COLORS,
    MAZE_AURORA_NOVELTY_10_COLORS,
    MAZE_AURORA_SURPRISE_10_COLORS,
    # MAZE_AURORA_NOVELTY_SURPRISE_10_COLORS,

    # AURORA using the ground truth as a behavioural descriptor
    # (NOT equivalent to NS - equivalent to standard QD with adaptive distance threshold)
    MAZE_AURORA_HARD_CODED_POS,
    # MAZE_AURORA_HARD_CODED_GEN_DESC,
    MAZE_AURORA_HARD_CODED_POS_NO_SELECTION,

    # MAZE_AURORA_HARD_CODED_POS_L_FIXED_5,

    # TAXONS in a classical setting (different latent spaces)

    *DICT_MAZE_TAXONS_n_COLORS.values(),

    # TAXONS VARIANTS (NOVELTY ONLY/SURPRISE ONLY)
    # MAZE_TAXO_NOVELTY_10_COLORS,
    # MAZE_TAXO_SURPRISE_10_COLORS,

    # A TAXONS using the ground truth as a behavioural descriptor (should be equivalent to NS without elitism)
    # MAZE_TAXONS_HARD_CODED_POS,

    # Using Fitness

    # MAZE_AURORA_UNIFORM_10_COLORS_FITNESS,
    # MAZE_AURORA_HARD_CODED_POS_FITNESS,

    # # Using Fitness with fixed l
    # *DICT_MAZE_AURORA_HARD_CODED_POS_L_FIXED_l_HAS_FIT.values(),

    # Experiments using Sticky Walls
    # MAZE_AURORA_UNIFORM_10_COLORS_STICKY,
    # MAZE_AURORA_HARD_CODED_POS_STICKY,
    # MAZE_TAXONS_10_COLORS_STICKY,
    # MAZE_TAXONS_HARD_CODED_POS_STICKY,

    # For analysis effect no Normalisation on Aurora
    # *DICT_MAZE_AURORA_UNIFORM_n_COLORS_NORMALISATION.values(),

    # For analysis effect Volume Adaptive Threshold (VAT) on Aurora
    *DICT_MAZE_AURORA_UNIFORM_n_COLORS_VAT.values(),

    # For analysis effect Volume Adaptive Threshold (VAT) + no Normalisation on Aurora
    # *DICT_MAZE_AURORA_UNIFORM_n_COLORS_VAT_NORMALISATION.values(),

    # ------------------------------

    # Taxons Elitism variants
    # MAZE_TAXONS_10_COLORS_ELITISM,
    MAZE_TAXONS_HARD_CODED_POS_ELITISM,
    # MAZE_TAXONS_10_COLORS_STICKY_ELITISM,
    # MAZE_TAXONS_HARD_CODED_POS_STICKY_ELITISM,

    # ------------------------------

    # For analysis effect update_container_period
    # *DICT_MAZE_AURORA_UNIFORM_n_COLORS_UPDATE_CONTAINER_PERIOD.values(),

    # For analysis effect coefficient_proportional_control_l
    # *DICT_MAZE_AURORA_UNIFORM_n_COLORS_COEFFICIENT_PROPORTIONAL_CONTROL_L.values(),
]


if __name__ == '__main__':
    # Outputting table of executables
    d = dict()
    d["executable"] = []
    d.update({
        key: []
        for key in list(vars(MAZE_AURORA_HARD_CODED_POS).keys())[:-5]
        if key != "_arguments"
    })
    for exp in LIST_EXPERIMENTS: # type: Experiment
        for key, value in vars(exp).items():
            if key in d:
                d[key].append(value)
        d["executable"].append(exp.get_exec_name())
    import pandas as pd
    df = pd.DataFrame(d)
    table_summary = df.to_markdown()
    df.to_csv("maze.csv", index=False)
    print(table_summary)

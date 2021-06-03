from singularity.collections_experiments.maze import DICT_SELECTOR_TO_ALGO_AURORA, DICT_SELECTOR_TO_ALGO_TAXONS, \
    UNIFORM, CURIOSITY, NOVELTY, SURPRISE, NOVELTY_SURPRISE
from singularity.experiment import Experiment, EncoderType, \
    LIST_COEFFICIENT_PROPORTIONAL_CONTROL_L, LIST_UPDATE_CONTAINER_PERIOD, DICT_EXPERIMENT_TO_CONFIG, STR_HEXAPOD_CAMERA_VERTICAL


NUMBER_CORES = DICT_EXPERIMENT_TO_CONFIG[STR_HEXAPOD_CAMERA_VERTICAL].number_cpus_to_use

############################
# Camera vertical AURORA
############################


def get_aurora_n_colors_hexapod_vertical(latent_space: int,
                                         algo: str,
                                         has_fit=True,
                                         use_volume_adaptive_threshold=False,
                                         no_normalisation=True,
                                         update_container_period: int = None,
                                         coefficient_proportional_control_l: float = None,
                                         encoder_type: EncoderType = EncoderType.cnn_ae,
                                         use_colors: bool = True,
                                         ):
    return Experiment(name_exec='aurora',
                      algo=algo,
                      env='hexa_cam_vertical',
                      latent_space=latent_space,
                      use_colors=use_colors,
                      arguments=f"--number-threads {NUMBER_CORES}",
                      encoder_type=encoder_type,
                      has_fit=has_fit,
                      use_volume_adaptive_threshold=use_volume_adaptive_threshold,
                      no_normalisation=no_normalisation,
                      update_container_period=update_container_period,
                      coefficient_proportional_control_l=coefficient_proportional_control_l,
                      )


def get_taxons_n_colors_hexapod_vertical(latent_space: int,
                                         algo: str,
                                         taxons_elitism: bool = False) -> Experiment:

    assert algo in DICT_SELECTOR_TO_ALGO_TAXONS.values()

    return Experiment(name_exec='aurora',
                      algo=algo,
                      env='hexa_cam_vertical',
                      latent_space=latent_space,
                      use_colors=True,
                      arguments=f"--number-threads {NUMBER_CORES}",
                      encoder_type=EncoderType.cnn_ae,
                      taxons_elitism=taxons_elitism,
                      has_fit=False
                      )


def get_hand_coded_qd_hexapod_camera_vertical(fixed_l: float = None, has_fit: bool = True) -> Experiment:
    return Experiment(name_exec='aurora',
                      algo='hand_coded_qd',
                      env='hexa_cam_vertical',
                      latent_space=2,
                      use_colors=True,
                      arguments=f"--number-threads {NUMBER_CORES}",
                      fixed_l=fixed_l,
                      has_fit=has_fit
                      )


LIST_DIMS_LATENT_SPACES_AURORA = [2, 5, 8, 10, 15, 20]
LIST_DIMS_LATENT_SPACES_TAXONS = [10]

# Various selectors

CAMERA_VERTICAL_AURORA_10_COLORS_CURIOSITY = get_aurora_n_colors_hexapod_vertical(latent_space=10, algo=DICT_SELECTOR_TO_ALGO_AURORA[CURIOSITY], has_fit=True)
CAMERA_VERTICAL_AURORA_10_COLORS_NOVELTY = get_aurora_n_colors_hexapod_vertical(latent_space=10, algo=DICT_SELECTOR_TO_ALGO_AURORA[NOVELTY], has_fit=True)
CAMERA_VERTICAL_AURORA_10_COLORS_SURPRISE = get_aurora_n_colors_hexapod_vertical(latent_space=10, algo=DICT_SELECTOR_TO_ALGO_AURORA[SURPRISE], has_fit=True)
CAMERA_VERTICAL_AURORA_10_COLORS_NOVELTY_SURPRISE = get_aurora_n_colors_hexapod_vertical(latent_space=10, algo=DICT_SELECTOR_TO_ALGO_AURORA[NOVELTY_SURPRISE], has_fit=True)

# Various Latent sizes

DICT_HEXAPOD_CAMERA_VERTICAL_AURORA_n_COLORS_UNIFORM = {
    latent_dim: get_aurora_n_colors_hexapod_vertical(latent_space=latent_dim, algo=DICT_SELECTOR_TO_ALGO_AURORA[UNIFORM], has_fit=True)

    for latent_dim in LIST_DIMS_LATENT_SPACES_AURORA
}

DICT_HEXAPOD_CAMERA_VERTICAL_AURORA_n_COLORS_UNIFORM[3] = get_aurora_n_colors_hexapod_vertical(latent_space=3, algo=DICT_SELECTOR_TO_ALGO_AURORA[UNIFORM], has_fit=True)
DICT_HEXAPOD_CAMERA_VERTICAL_AURORA_n_COLORS_UNIFORM[4] = get_aurora_n_colors_hexapod_vertical(latent_space=4, algo=DICT_SELECTOR_TO_ALGO_AURORA[UNIFORM], has_fit=True)

# AURORA using the ground truth as a behavioural descriptor
# (NOT equivalent to NS - equivalent to standard QD with adaptive distance threshold)

CAMERA_VERTICAL_HAND_CODED_GT = Experiment(name_exec='aurora', algo='hand_coded_qd', env='hexa_cam_vertical',
                                           latent_space=2, use_colors=True, arguments=f"--number-threads {NUMBER_CORES}", has_fit=True)

CAMERA_VERTICAL_HAND_CODED_GT_COLORS_NO_SELECTION = Experiment(name_exec='aurora', algo='hand_coded_qd_no_sel', env='hexa_cam_vertical',
                                                               latent_space=2, use_colors=True, arguments=f"--number-threads {NUMBER_CORES}", has_fit=True)

# TAXONS

# # TAXONS in a classical setting (different latent spaces)
DICT_CAMERA_VERTICAL_TAXONS_n_COLORS = {
    latent_dim: get_taxons_n_colors_hexapod_vertical(latent_space=latent_dim, algo=DICT_SELECTOR_TO_ALGO_TAXONS[NOVELTY_SURPRISE])
    for latent_dim in LIST_DIMS_LATENT_SPACES_TAXONS
}

# # TAXONS VARIANTS (NOVELTY ONLY/SURPRISE ONLY)
CAMERA_VERTICAL_TAXO_NOVELTY_10_COLORS = get_taxons_n_colors_hexapod_vertical(latent_space=10, algo=DICT_SELECTOR_TO_ALGO_TAXONS[NOVELTY])
CAMERA_VERTICAL_TAXO_SURPRISE_10_COLORS = get_taxons_n_colors_hexapod_vertical(latent_space=10, algo=DICT_SELECTOR_TO_ALGO_TAXONS[SURPRISE])

# # A TAXONS using the ground truth as a behavioural descriptor (should be equivalent to NS)
CAMERA_VERTICAL_TAXONS_HARD_CODED_POS = Experiment(name_exec='aurora', algo='hand_coded_taxons',
                                                   env='hexa_cam_vertical', latent_space=2,
                                                   use_colors=True, arguments=f"--number-threads {NUMBER_CORES}")


# Using other hard-coded behavioural descriptors

CAMERA_VERTICAL_HAND_CODED_PIXELS = Experiment(name_exec='aurora', algo='hand_coded_qd', env='hexa_cam_vert_hc_pix',
                                               latent_space=3 * 32 * 32, use_colors=True,  # Latent space * 3 as using colors
                                               arguments=f"--number-threads {NUMBER_CORES}", has_fit=True)

HAND_CODED_GENOTYPE_DESCRIPTORS = Experiment(name_exec='aurora', algo='hand_coded_qd', env='hexa_gen_desc',
                                             latent_space=36, use_colors=True, arguments=f"--number-threads {NUMBER_CORES}", has_fit=True)


# # Non-constant fitness function
#
# # # AURORA
# CAMERA_VERTICAL_AURORA_UNIFORM_10_COLORS_FITNESS = get_aurora_n_colors_hexapod_vertical(latent_space=10,
#                                                                                         algo=DICT_SELECTOR_TO_ALGO_AURORA[UNIFORM],
#                                                                                         has_fit=True)
#
# # # Hand-coded
# CAMERA_VERTICAL_HAND_CODED_GT_FITNESS = Experiment(name_exec='aurora', algo='hand_coded_qd', env='hexa_cam_vertical', latent_space=2,
#                                                    use_colors=True, arguments=f"--number-threads {NUMBER_CORES}", has_fit=True)

# TODO: See TODO below
# CAMERA_VERTICAL_HAND_CODED_GT_COLORS_FIXED_L = Experiment(name_exec='aurora', algo='hand_coded_qd', env='hexa_cam_vertical',
#                                                           latent_space=2, use_colors=True, arguments=f"--number-threads {NUMBER_CORES}",
#                                                           fixed_l=3.8)

# For analysis effect adaptive l compared to fixed l in case of normal QD (with fitness value)
LIST_FIXED_L = [0.16, 0.017, 0.018, 0.019, 0.020, 0.021]
DICT_HEXAPOD_CAMERA_VERTICAL_AURORA_HARD_CODED_POS_L_FIXED_l = {
    l: get_hand_coded_qd_hexapod_camera_vertical(fixed_l=l,
                                                 has_fit=True)
    for l in LIST_FIXED_L
}

# For analysis effect no Normalisation on Aurora
DICT_HEXAPOD_CAMERA_VERTICAL_AURORA_UNIFORM_n_COLORS_NORMALISATION = {
    latent_dim: get_aurora_n_colors_hexapod_vertical(latent_space=latent_dim,
                                                     algo=DICT_SELECTOR_TO_ALGO_AURORA[UNIFORM],
                                                     no_normalisation=False,
                                                     has_fit=True)

    for latent_dim in LIST_DIMS_LATENT_SPACES_AURORA
}

# For analysis effect Volume Adaptive Threshold (VAT) on Aurora
DICT_HEXAPOD_CAMERA_VERTICAL_AURORA_UNIFORM_n_COLORS_VAT = {
    latent_dim: get_aurora_n_colors_hexapod_vertical(latent_space=latent_dim,
                                                     algo=DICT_SELECTOR_TO_ALGO_AURORA[UNIFORM],
                                                     use_volume_adaptive_threshold=True,
                                                     has_fit=True)

    for latent_dim in LIST_DIMS_LATENT_SPACES_AURORA
}

# For analysis effect Volume Adaptive Threshold (VAT) + no Normalisation on Aurora
DICT_HEXAPOD_CAMERA_VERTICAL_AURORA_UNIFORM_n_COLORS_VAT_NORMALISATION = {
    latent_dim: get_aurora_n_colors_hexapod_vertical(latent_space=latent_dim,
                                                     algo=DICT_SELECTOR_TO_ALGO_AURORA[UNIFORM],
                                                     use_volume_adaptive_threshold=True,
                                                     no_normalisation=False,
                                                     has_fit=True)

    for latent_dim in LIST_DIMS_LATENT_SPACES_AURORA
}

# Taxons Elitism variants
CAMERA_VERTICAL_TAXONS_n_COLORS_ELITISM = get_taxons_n_colors_hexapod_vertical(latent_space=10, algo=DICT_SELECTOR_TO_ALGO_TAXONS[NOVELTY_SURPRISE], taxons_elitism=True)

CAMERA_VERTICAL_TAXONS_HARD_CODED_POS_ELITISM = Experiment(name_exec='aurora', algo='hand_coded_taxons',
                                                           env='hexa_cam_vertical', latent_space=2,
                                                           use_colors=True,
                                                           arguments=f"--number-threads {NUMBER_CORES}",
                                                           taxons_elitism=True)


# For analysis effect update_container_period
DICT_HEXAPOD_CAMERA_VERTICAL_AURORA_UNIFORM_n_COLORS_UPDATE_CONTAINER_PERIOD = {
    update_container_period: get_aurora_n_colors_hexapod_vertical(latent_space=10,
                                                                  algo=DICT_SELECTOR_TO_ALGO_AURORA[UNIFORM],
                                                                  update_container_period=update_container_period,
                                                                  has_fit=True,
                                                                  )
    for update_container_period in LIST_UPDATE_CONTAINER_PERIOD
}

# For analysis effect coefficient_proportional_control_l
DICT_HEXAPOD_CAMERA_VERTICAL_AURORA_UNIFORM_n_COLORS_COEFFICIENT_PROPORTIONAL_CONTROL_L = {
    coefficient_proportional_control_l: get_aurora_n_colors_hexapod_vertical(
        latent_space=10,
        algo=DICT_SELECTOR_TO_ALGO_AURORA[UNIFORM],
        coefficient_proportional_control_l=coefficient_proportional_control_l,
        has_fit=True,
    )
    for coefficient_proportional_control_l in LIST_COEFFICIENT_PROPORTIONAL_CONTROL_L
}

# AURORA using too strong autoencoder
HEXAPOD_CAMERA_VERTICAL_AURORA_n_COLORS_UNIFORM_STRONG_AE = get_aurora_n_colors_hexapod_vertical(latent_space=20,
                                                                                                 algo=DICT_SELECTOR_TO_ALGO_AURORA[UNIFORM],
                                                                                                 encoder_type=EncoderType.strg_cnn,
                                                                                                 use_colors=False,
                                                                                                 has_fit=True)


# TODO: Fit use of BD/Ground Truth

LIST_EXPERIMENTS = [
    # -----------------------------
    # AURORA
    # -----------------------------

    # Various selectors
    # CAMERA_VERTICAL_AURORA_10_COLORS_CURIOSITY,
    CAMERA_VERTICAL_AURORA_10_COLORS_NOVELTY,
    CAMERA_VERTICAL_AURORA_10_COLORS_SURPRISE,
    # CAMERA_VERTICAL_AURORA_10_COLORS_NOVELTY_SURPRISE,

    # Various Latent sizes
    *DICT_HEXAPOD_CAMERA_VERTICAL_AURORA_n_COLORS_UNIFORM.values(),

    # -----------------------------
    # Hard Coded BD
    # -----------------------------

    CAMERA_VERTICAL_HAND_CODED_GT,
    CAMERA_VERTICAL_HAND_CODED_GT_COLORS_NO_SELECTION,

    # Using other hard-coded behavioural descriptors

    # CAMERA_VERTICAL_HAND_CODED_PIXELS,
    # HAND_CODED_GENOTYPE_DESCRIPTORS,

    # -----------------------------
    # TAXONS
    # -----------------------------

    # # TAXONS in a classical setting (different latent spaces)
    *DICT_CAMERA_VERTICAL_TAXONS_n_COLORS.values(),

    # # TAXONS VARIANTS (NOVELTY ONLY/SURPRISE ONLY)
    # CAMERA_VERTICAL_TAXO_NOVELTY_10_COLORS,
    # CAMERA_VERTICAL_TAXO_SURPRISE_10_COLORS,

    # # A TAXONS using the ground truth as a behavioural descriptor (should be equivalent to NS)
    # CAMERA_VERTICAL_TAXONS_HARD_CODED_POS,  # NS without elitism

    # -----------------------------
    # Non-constant fitness function
    # -----------------------------

    # CAMERA_VERTICAL_AURORA_UNIFORM_10_COLORS_FITNESS,
    # CAMERA_VERTICAL_HAND_CODED_GT_FITNESS,

    # ------------------------------

    # Using Fixed l
    # *DICT_HEXAPOD_CAMERA_VERTICAL_AURORA_HARD_CODED_POS_L_FIXED_l.values(),

    # For analysis effect no Normalisation on Aurora
    # *DICT_HEXAPOD_CAMERA_VERTICAL_AURORA_UNIFORM_n_COLORS_NORMALISATION.values(),

    # For analysis effect Volume Adaptive Threshold (VAT) on Aurora
    *DICT_HEXAPOD_CAMERA_VERTICAL_AURORA_UNIFORM_n_COLORS_VAT.values(),

    # For analysis effect Volume Adaptive Threshold (VAT) + no Normalisation on Aurora
    # *DICT_HEXAPOD_CAMERA_VERTICAL_AURORA_UNIFORM_n_COLORS_VAT_NORMALISATION.values(),

    # ------------------------------

    # TAXONS elitism variants
    # CAMERA_VERTICAL_TAXONS_n_COLORS_ELITISM,
    CAMERA_VERTICAL_TAXONS_HARD_CODED_POS_ELITISM,  # Equivalent to NS

    # ------------------------------

    # For analysis effect update_container_period
    # *DICT_HEXAPOD_CAMERA_VERTICAL_AURORA_UNIFORM_n_COLORS_UPDATE_CONTAINER_PERIOD.values(),

    # For analysis effect coefficient_proportional_control_l
    # *DICT_HEXAPOD_CAMERA_VERTICAL_AURORA_UNIFORM_n_COLORS_COEFFICIENT_PROPORTIONAL_CONTROL_L.values(),

    # Strong Autoencoder
    # HEXAPOD_CAMERA_VERTICAL_AURORA_n_COLORS_UNIFORM_STRONG_AE,
]

if __name__ == '__main__':
    # Outputting table of executables
    d = dict()
    d["executable"] = []
    d.update({
        key: []
        for key in list(vars(HAND_CODED_GENOTYPE_DESCRIPTORS).keys())[:-5]
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
    df.to_csv("hexapod.csv", index=False)
    print(table_summary)

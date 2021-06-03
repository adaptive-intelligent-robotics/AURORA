import sys

sys.path.append('/git/sferes2/exp/aurora/submodules/figure_generator')
sys.path.append('/git/sferes2/exp/aurora/')

from analysis.metrics.hexapod_walls import MetricsWalls
from analysis.metrics.maze import MetricsMaze
from analysis.metrics.hexapod_camera_vertical import MetricsHexapodCameraVertical
from analysis.metrics.air_hockey import MetricsAirHockey
from singularity.collections_experiments import maze, hexapod_camera_vertical, air_hockey

UPLOAD = '/UPLOAD/'

COMPARISONS_HEXAPOD_CAMERA_VERTICAL = {
    "Hexapod Camera Vertical - QD vs AURORA": {
        hexapod_camera_vertical.DICT_HEXAPOD_CAMERA_VERTICAL_AURORA_n_COLORS_UNIFORM[10]: "AURORA - 10",
        hexapod_camera_vertical.CAMERA_VERTICAL_HAND_CODED_GT: f"Standard QD",
    },
    "Hexapod Camera Vertical - QD - PSAT": {
        **{
            hexapod_camera_vertical.DICT_HEXAPOD_CAMERA_VERTICAL_AURORA_HARD_CODED_POS_L_FIXED_l[l]: f"l={l}"
            for l in hexapod_camera_vertical.DICT_HEXAPOD_CAMERA_VERTICAL_AURORA_HARD_CODED_POS_L_FIXED_l
        },
        # hexapod_camera_vertical.CAMERA_VERTICAL_HAND_CODED_GT_FITNESS: f"PSAT"
    },
    "Coefficient Proportional Control": {
        **{
            hexapod_camera_vertical.DICT_HEXAPOD_CAMERA_VERTICAL_AURORA_UNIFORM_n_COLORS_COEFFICIENT_PROPORTIONAL_CONTROL_L[coeff]: f"coeff={coeff}"
            for coeff in hexapod_camera_vertical.DICT_HEXAPOD_CAMERA_VERTICAL_AURORA_UNIFORM_n_COLORS_COEFFICIENT_PROPORTIONAL_CONTROL_L
        }
    },
    "Update Container Period": {
        **{
            hexapod_camera_vertical.DICT_HEXAPOD_CAMERA_VERTICAL_AURORA_UNIFORM_n_COLORS_UPDATE_CONTAINER_PERIOD[period]: f"period={period}"
            for period in hexapod_camera_vertical.DICT_HEXAPOD_CAMERA_VERTICAL_AURORA_UNIFORM_n_COLORS_UPDATE_CONTAINER_PERIOD
        }
    },

    "Hexapod Camera Vertical - AURORA - PSAT": {
        **{
            hexapod_camera_vertical.DICT_HEXAPOD_CAMERA_VERTICAL_AURORA_n_COLORS_UNIFORM[latent_dim]: f"PSAT {latent_dim}"
            for latent_dim in hexapod_camera_vertical.DICT_HEXAPOD_CAMERA_VERTICAL_AURORA_n_COLORS_UNIFORM
        },
        **{
            hexapod_camera_vertical.DICT_HEXAPOD_CAMERA_VERTICAL_AURORA_UNIFORM_n_COLORS_NORMALISATION[latent_dim]: f"no-norm {latent_dim}"
            for latent_dim in hexapod_camera_vertical.DICT_HEXAPOD_CAMERA_VERTICAL_AURORA_UNIFORM_n_COLORS_NORMALISATION
        },
        **{
            hexapod_camera_vertical.DICT_HEXAPOD_CAMERA_VERTICAL_AURORA_UNIFORM_n_COLORS_VAT[latent_dim]: f"VAT {latent_dim}"
            for latent_dim in hexapod_camera_vertical.DICT_HEXAPOD_CAMERA_VERTICAL_AURORA_UNIFORM_n_COLORS_VAT
        },
        **{
            hexapod_camera_vertical.DICT_HEXAPOD_CAMERA_VERTICAL_AURORA_UNIFORM_n_COLORS_VAT_NORMALISATION[latent_dim]: f"no-norm VAT {latent_dim}"
            for latent_dim in hexapod_camera_vertical.DICT_HEXAPOD_CAMERA_VERTICAL_AURORA_UNIFORM_n_COLORS_VAT_NORMALISATION
        },
    },

    "Comparison Algorithms - Hexapod Camera Vertical Latent-10": {
        hexapod_camera_vertical.DICT_HEXAPOD_CAMERA_VERTICAL_AURORA_n_COLORS_UNIFORM[10]: "AURORA Uniform",

        hexapod_camera_vertical.CAMERA_VERTICAL_AURORA_10_COLORS_NOVELTY: "AURORA Novelty",
        hexapod_camera_vertical.CAMERA_VERTICAL_AURORA_10_COLORS_SURPRISE: "AURORA Surprise",
        hexapod_camera_vertical.CAMERA_VERTICAL_AURORA_10_COLORS_NOVELTY_SURPRISE: "AURORA Novelty Surprise",
        hexapod_camera_vertical.CAMERA_VERTICAL_AURORA_10_COLORS_CURIOSITY: "AURORA Curiosity",

        hexapod_camera_vertical.CAMERA_VERTICAL_HAND_CODED_GT: "HC Pos",
        hexapod_camera_vertical.CAMERA_VERTICAL_HAND_CODED_GT_COLORS_NO_SELECTION: "HC Pos No Sel",

        hexapod_camera_vertical.HAND_CODED_GENOTYPE_DESCRIPTORS: "HC Gen BD",
        hexapod_camera_vertical.CAMERA_VERTICAL_HAND_CODED_PIXELS: "HC Pix BD",
    },

    "Hard-coded Pos with adaptive l": {
        hexapod_camera_vertical.CAMERA_VERTICAL_HAND_CODED_GT: "HC Pos - l adapt",
        # hexapod_camera_vertical.CAMERA_VERTICAL_HAND_CODED_GT_COLORS_L_FIXED: "HC Pos - l fixed", # TODO
    },

    "AURORA - Latent Space": {
        **{
            hexapod_camera_vertical.DICT_HEXAPOD_CAMERA_VERTICAL_AURORA_n_COLORS_UNIFORM[latent_dim]: f"AURORA - {latent_dim}"
            for latent_dim in hexapod_camera_vertical.DICT_HEXAPOD_CAMERA_VERTICAL_AURORA_n_COLORS_UNIFORM
        },

        hexapod_camera_vertical.CAMERA_VERTICAL_HAND_CODED_GT: "HC Pos",
    },

    "Non-Constant Fitness": {
        # hexapod_camera_vertical.CAMERA_VERTICAL_AURORA_UNIFORM_10_COLORS_FITNESS: "AURORA 10 - Fitness",
        # hexapod_camera_vertical.CAMERA_VERTICAL_HAND_CODED_GT_FITNESS: "HC Pos - Fitness",

        hexapod_camera_vertical.DICT_HEXAPOD_CAMERA_VERTICAL_AURORA_n_COLORS_UNIFORM[10]: "AURORA 10 - No Fit",
        hexapod_camera_vertical.CAMERA_VERTICAL_HAND_CODED_GT: "HC Pos",
    },

    "TAXONS - Latent Space": {
        hexapod_camera_vertical.DICT_CAMERA_VERTICAL_TAXONS_n_COLORS[latent_dim]: f"TAXONS - {latent_dim}"
        for latent_dim in hexapod_camera_vertical.DICT_CAMERA_VERTICAL_TAXONS_n_COLORS
    },
    "TAXONS - Variants": {
        hexapod_camera_vertical.DICT_CAMERA_VERTICAL_TAXONS_n_COLORS[10]: "TAXONS - 10",
        hexapod_camera_vertical.CAMERA_VERTICAL_TAXO_NOVELTY_10_COLORS: "TAXO-N - 10",
        hexapod_camera_vertical.CAMERA_VERTICAL_TAXO_SURPRISE_10_COLORS: "TAXO-S - 10",
        hexapod_camera_vertical.CAMERA_VERTICAL_TAXONS_HARD_CODED_POS: "TAXONS - HC"
    },
    "Comparison - AURORA - TAXONS": {
        hexapod_camera_vertical.DICT_HEXAPOD_CAMERA_VERTICAL_AURORA_n_COLORS_UNIFORM[10]: "AURORA - Uniform",
        hexapod_camera_vertical.CAMERA_VERTICAL_AURORA_10_COLORS_CURIOSITY: "AURORA - Curiosity",
        hexapod_camera_vertical.CAMERA_VERTICAL_AURORA_10_COLORS_NOVELTY: "AURORA - Novelty",
        hexapod_camera_vertical.CAMERA_VERTICAL_AURORA_10_COLORS_SURPRISE: "AURORA - Surprise",
        hexapod_camera_vertical.CAMERA_VERTICAL_AURORA_10_COLORS_NOVELTY_SURPRISE: "AURORA - Novelty Surprise",
        hexapod_camera_vertical.DICT_CAMERA_VERTICAL_TAXONS_n_COLORS[10]: "TAXONS - 10",
    },
}


"""
dict (title comparison) -> (experiment -> legend_in_the_comparison)
"""
COMPARISONS_WALLS = {}

COMPARISONS_MAZE_STANDARD = {
    "Maze - QD - PSAT": {
        **{
            maze.DICT_MAZE_AURORA_HARD_CODED_POS_L_FIXED_l_HAS_FIT[l]: f"l={l}"
            for l in maze.DICT_MAZE_AURORA_HARD_CODED_POS_L_FIXED_l_HAS_FIT
        },
        # maze.MAZE_AURORA_HARD_CODED_POS_FITNESS: f"PSAT"
    },
    "Coefficient Proportional Control": {
        **{
            maze.DICT_MAZE_AURORA_UNIFORM_n_COLORS_COEFFICIENT_PROPORTIONAL_CONTROL_L[coeff]: f"coeff={coeff}"
            for coeff in maze.DICT_MAZE_AURORA_UNIFORM_n_COLORS_COEFFICIENT_PROPORTIONAL_CONTROL_L
        }
    },
    "Update Container Period": {
        **{
            maze.DICT_MAZE_AURORA_UNIFORM_n_COLORS_UPDATE_CONTAINER_PERIOD[period]: f"period={period}"
            for period in maze.DICT_MAZE_AURORA_UNIFORM_n_COLORS_UPDATE_CONTAINER_PERIOD
        }
    },
    "Maze - AURORA - PSAT": {
        **{
            maze.DICT_MAZE_AURORA_UNIFORM_n_COLORS[latent_dim]: f"PSAT {latent_dim}"
            for latent_dim in maze.DICT_MAZE_AURORA_UNIFORM_n_COLORS
        },
        **{
            maze.DICT_MAZE_AURORA_UNIFORM_n_COLORS_NORMALISATION[latent_dim]: f"no-norm {latent_dim}"
            for latent_dim in maze.DICT_MAZE_AURORA_UNIFORM_n_COLORS_NORMALISATION
        },
        **{
            maze.DICT_MAZE_AURORA_UNIFORM_n_COLORS_VAT[latent_dim]: f"VAT {latent_dim}"
            for latent_dim in maze.DICT_MAZE_AURORA_UNIFORM_n_COLORS_VAT
        },
        **{
            maze.DICT_MAZE_AURORA_UNIFORM_n_COLORS_VAT_NORMALISATION[latent_dim]: f"no-norm VAT {latent_dim}"
            for latent_dim in maze.DICT_MAZE_AURORA_UNIFORM_n_COLORS_VAT_NORMALISATION
        },
    },
    "AURORA - Latent Space": {
        maze.DICT_MAZE_AURORA_UNIFORM_n_COLORS[latent_dim]: f"AURORA - {latent_dim}"
        for latent_dim in maze.DICT_MAZE_AURORA_UNIFORM_n_COLORS
    },
    "AURORA - Variants": {
        maze.DICT_MAZE_AURORA_UNIFORM_n_COLORS[10]: "AURORA - Uniform",

        maze.MAZE_AURORA_CURIOSITY_10_COLORS: "AURORA - Curiosity",
        maze.MAZE_AURORA_NOVELTY_10_COLORS: "AURORA - Novelty",
        maze.MAZE_AURORA_SURPRISE_10_COLORS: "AURORA - Surprise",
        maze.MAZE_AURORA_NOVELTY_SURPRISE_10_COLORS: "AURORA - Novelty Surprise",

        maze.MAZE_AURORA_HARD_CODED_POS: "HC Pos - Uniform Selection",
        maze.MAZE_AURORA_HARD_CODED_POS_NO_SELECTION: "HC Pos - No Selection"
    },
    "AURORA - Hard-coded Pos with adaptive l": {
        maze.MAZE_AURORA_HARD_CODED_POS: "HC Pos - l adapt",
        maze.MAZE_AURORA_HARD_CODED_POS_L_FIXED_5: "HC Pos - l fixed",
    },
    "TAXONS - Latent Space": {
        maze.DICT_MAZE_TAXONS_n_COLORS[latent_dim]: f"TAXONS - {latent_dim}"
        for latent_dim in maze.DICT_MAZE_TAXONS_n_COLORS
    },
    "TAXONS - Variants": {
        maze.DICT_MAZE_TAXONS_n_COLORS[10]: "TAXONS - 10",
        maze.MAZE_TAXO_NOVELTY_10_COLORS: "TAXO-N - 10",
        maze.MAZE_TAXO_SURPRISE_10_COLORS: "TAXO-S - 10",
        maze.MAZE_TAXONS_HARD_CODED_POS: "TAXONS - HC"
    },
    "Comparison - AURORA - TAXONS": {
        maze.DICT_MAZE_AURORA_UNIFORM_n_COLORS[10]: "AURORA - Uniform",
        maze.MAZE_AURORA_CURIOSITY_10_COLORS: "AURORA - Curiosity",
        maze.MAZE_AURORA_NOVELTY_10_COLORS: "AURORA - Novelty",
        maze.MAZE_AURORA_SURPRISE_10_COLORS: "AURORA - Surprise",
        maze.MAZE_AURORA_NOVELTY_SURPRISE_10_COLORS: "AURORA - Novelty Surprise",
        maze.DICT_MAZE_TAXONS_n_COLORS[10]: "TAXONS - 10",
    },
    "Compare Non-constant fitness variants": {
        # maze.MAZE_AURORA_UNIFORM_10_COLORS_FITNESS: "AURORA 10 - Fitness",
        # maze.MAZE_AURORA_HARD_CODED_POS_FITNESS: "HC Pos - Fitness",

        maze.DICT_MAZE_AURORA_UNIFORM_n_COLORS[10]: "AURORA 10 - No Fit",
        maze.MAZE_AURORA_HARD_CODED_POS: "HC Pos - No Fit",
    },

    "Sticky Maze Env": {
        maze.MAZE_AURORA_UNIFORM_10_COLORS_STICKY: "AURORA",
        maze.MAZE_AURORA_HARD_CODED_POS_STICKY: "QD - HC Pos",

        maze.MAZE_TAXONS_10_COLORS_STICKY: "TAXONS",
        maze.MAZE_TAXONS_HARD_CODED_POS_STICKY: "TAXONS - HC",
    },
}

COMPARISONS_AIR_HOCKEY = {
    "Air Hockey - QD - PSAT": {
        **{
            air_hockey.DICT_AIR_HOCKEY_AURORA_HARD_CODED_POS_L_FIXED_l[l]: f"l={l}"
            for l in air_hockey.DICT_AIR_HOCKEY_AURORA_HARD_CODED_POS_L_FIXED_l
        },
        air_hockey.AIR_HOCKEY_HAND_CODED_GT: f"PSAT"
    },

    "Coefficient Proportional Control": {
        **{
            air_hockey.DICT_AIR_HOCKEY_AURORA_UNIFORM_n_COLORS_COEFFICIENT_PROPORTIONAL_CONTROL_L[coeff]: f"coeff={coeff}"
            for coeff in air_hockey.DICT_AIR_HOCKEY_AURORA_UNIFORM_n_COLORS_COEFFICIENT_PROPORTIONAL_CONTROL_L
        }
    },

    "Update Container Period": {
        **{
            air_hockey.DICT_AIR_HOCKEY_AURORA_UNIFORM_n_COLORS_UPDATE_CONTAINER_PERIOD[period]: f"period={period}"
            for period in air_hockey.DICT_AIR_HOCKEY_AURORA_UNIFORM_n_COLORS_UPDATE_CONTAINER_PERIOD
        }
    },

    "Air Hockey - AURORA - PSAT": {
        **{
            air_hockey.DICT_AIR_HOCKEY_AURORA_n_COLORS_UNIFORM[latent_dim]: f"PSAT {latent_dim}"
            for latent_dim in air_hockey.DICT_AIR_HOCKEY_AURORA_n_COLORS_UNIFORM
        },
        **{
            air_hockey.DICT_AIR_HOCKEY_AURORA_UNIFORM_n_COLORS_NORMALISATION[latent_dim]: f"no-norm {latent_dim}"
            for latent_dim in air_hockey.DICT_AIR_HOCKEY_AURORA_UNIFORM_n_COLORS_NORMALISATION
        },
        **{
            air_hockey.DICT_AIR_HOCKEY_AURORA_UNIFORM_n_COLORS_VAT[latent_dim]: f"VAT {latent_dim}"
            for latent_dim in air_hockey.DICT_AIR_HOCKEY_AURORA_UNIFORM_n_COLORS_VAT
        },
        **{
            air_hockey.DICT_AIR_HOCKEY_AURORA_UNIFORM_n_COLORS_VAT_NORMALISATION[latent_dim]: f"no-norm VAT {latent_dim}"
            for latent_dim in air_hockey.DICT_AIR_HOCKEY_AURORA_UNIFORM_n_COLORS_VAT_NORMALISATION
        },
    },

    "Comparison Algorithms - Air Hockey Latent-10": {
        air_hockey.DICT_AIR_HOCKEY_AURORA_n_COLORS_UNIFORM[10]: "AURORA Uniform",

        air_hockey.AIR_HOCKEY_AURORA_10_COLORS_NOVELTY: "AURORA Novelty",
        air_hockey.AIR_HOCKEY_AURORA_10_COLORS_SURPRISE: "AURORA Surprise",
        air_hockey.AIR_HOCKEY_AURORA_10_COLORS_NOVELTY_SURPRISE: "AURORA Novelty Surprise",
        air_hockey.AIR_HOCKEY_AURORA_10_COLORS_CURIOSITY: "AURORA Curiosity",

        air_hockey.AIR_HOCKEY_HAND_CODED_GT: "HC Pos",
        air_hockey.AIR_HOCKEY_HAND_CODED_GT_COLORS_NO_SELECTION: "HC Pos No Sel",

        air_hockey.AIR_HOCKEY_HAND_CODED_FULL_TRAJECTORY: "Full traj",

        # TODO
        # air_hockey.HAND_CODED_GENOTYPE_DESCRIPTORS: "HC Gen BD",
        # air_hockey.AIR_HOCKEY_HAND_CODED_PIXELS: "HC Pix BD",
    },

    "Hard-coded Pos with adaptive l": {
        air_hockey.AIR_HOCKEY_HAND_CODED_GT: "HC Pos - l adapt",
        # air_hockey.AIR_HOCKEY_HAND_CODED_GT_COLORS_L_FIXED: "HC Pos - l fixed", # TODO
    },

    "AURORA - Latent Space": {
        **{
            air_hockey.DICT_AIR_HOCKEY_AURORA_n_COLORS_UNIFORM[latent_dim]: f"AURORA - {latent_dim}"
            for latent_dim in air_hockey.DICT_AIR_HOCKEY_AURORA_n_COLORS_UNIFORM
        },

        air_hockey.AIR_HOCKEY_HAND_CODED_GT: "HC Pos",
    },

    "Non-Constant Fitness": {
        # TODO
        # air_hockey.AIR_HOCKEY_AURORA_UNIFORM_10_COLORS_FITNESS: "AURORA 10 - Fitness",
        # air_hockey.AIR_HOCKEY_HAND_CODED_GT_FITNESS: "HC Pos - Fitness",

        air_hockey.DICT_AIR_HOCKEY_AURORA_n_COLORS_UNIFORM[10]: "AURORA 10",
        air_hockey.AIR_HOCKEY_HAND_CODED_GT: "HC Pos",
    },

    "TAXONS - Latent Space": {
        air_hockey.DICT_AIR_HOCKEY_TAXONS_n_COLORS[latent_dim]: f"TAXONS - {latent_dim}"
        for latent_dim in air_hockey.DICT_AIR_HOCKEY_TAXONS_n_COLORS
    },
    "TAXONS - Variants": {
        air_hockey.DICT_AIR_HOCKEY_TAXONS_n_COLORS[10]: "TAXONS - 10",
        air_hockey.AIR_HOCKEY_TAXO_NOVELTY_10_COLORS: "TAXO-N - 10",
        air_hockey.AIR_HOCKEY_TAXO_SURPRISE_10_COLORS: "TAXO-S - 10",
        air_hockey.AIR_HOCKEY_TAXONS_HARD_CODED_POS: "TAXONS - HC"
    },
    "Comparison - AURORA - TAXONS": {
        air_hockey.DICT_AIR_HOCKEY_AURORA_n_COLORS_UNIFORM[10]: "AURORA - Uniform",
        air_hockey.AIR_HOCKEY_AURORA_10_COLORS_CURIOSITY: "AURORA - Curiosity",
        air_hockey.AIR_HOCKEY_AURORA_10_COLORS_NOVELTY: "AURORA - Novelty",
        air_hockey.AIR_HOCKEY_AURORA_10_COLORS_SURPRISE: "AURORA - Surprise",
        air_hockey.AIR_HOCKEY_AURORA_10_COLORS_NOVELTY_SURPRISE: "AURORA - Novelty Surprise",
        air_hockey.DICT_AIR_HOCKEY_TAXONS_n_COLORS[10]: "TAXONS - 10",
    },
}


# Declaring Mappings to comparisons / metrics / list experiments

WALLS = 'walls'
HEXAPOD_CAMERA_VERTICAL = 'hexapod_camera_vertical'
MAZE = 'maze'
AIR_HOCKEY = 'air_hockey'

# Verify that the two dictionaries do not have any comparison name in common
COMPARISONS_MAZE = COMPARISONS_MAZE_STANDARD

DICT_EXPERIMENT_TO_COMPARISON = {
    WALLS: COMPARISONS_WALLS,
    HEXAPOD_CAMERA_VERTICAL: COMPARISONS_HEXAPOD_CAMERA_VERTICAL,
    MAZE: COMPARISONS_MAZE,
    AIR_HOCKEY: COMPARISONS_AIR_HOCKEY,
}

DICT_EXPERIMENT_TO_METRIC = {
    WALLS: MetricsWalls,
    HEXAPOD_CAMERA_VERTICAL: MetricsHexapodCameraVertical,
    MAZE: MetricsMaze,
    AIR_HOCKEY: MetricsAirHockey,
}

DICT_EXPERIMENT_TO_LIST_EXPERIMENT = {
    WALLS: [],
    HEXAPOD_CAMERA_VERTICAL: hexapod_camera_vertical.LIST_EXPERIMENTS,
    MAZE: maze.LIST_EXPERIMENTS,
    AIR_HOCKEY: air_hockey.LIST_EXPERIMENTS,
}

//
// Created by Luca Grillotti on 04/06/2020.
//

#ifndef AURORA_PROJECT_INCLUDES_HPP
#define AURORA_PROJECT_INCLUDES_HPP

#include "compilation_variables.hpp"

#include "modifier/network_loader_pytorch.hpp"
#include "modifier/dimensionality_reduction.hpp"
#include "modifier/container_update_hand_coded.hpp"

#include "stat/stat_projection.hpp"
#include "stat/stat_current_gen.hpp"
#include "stat/stat_images_observations.hpp"
#include "stat/stat_images_reconstructions_obs.hpp"
#include "stat/stat_model_autoencoder.hpp"
#include "stat/stat_modifier.hpp"
#include "stat/stat_successive_gt.hpp"

#include "interactive_map/visualise_behaviours.hpp"
#include "interactive_map/analysis.hpp"

#include "algorithms/selector/variable_selector.hpp"
#include "algorithms/selector/surprise_value_selector.hpp"

#include "encoder_factory.hpp"

#include "algorithms/algorithms_factory.hpp"
#include "algorithms/aurora/definitions_aurora.hpp"
#include "algorithms/hand_coded/hand_coded_qd.hpp"
#include "algorithms/hand_coded/novelty_search.hpp"
#include "algorithms/quality_diversity_aurora_project.hpp"
#include "algorithms/taxons/definitions_taxons.hpp"
#include "algorithms/taxons/taxons_evolutionary_algorithm.hpp"

#include "environments/environments_factory.hpp"
#include "environments/hexapod/fit_hexapod_utils.hpp"
#include "environments/hexapod/fit_hexapod_abstract.hpp"
#include "environments/hexapod/fit_hexapod.hpp"
#include "environments/hexapod/params_hexapod.hpp"

#include "environments/maze/fit_maze.hpp"
#include "environments/maze/params_maze.hpp"

#include "environments/air_hockey/fit_air_hockey.hpp"
#include "environments/air_hockey/params_air_hockey.hpp"

#include "parameters_factory.hpp"
#include "environment_initialiser.hpp"

#endif //AURORA_PROJECT_INCLUDES_HPP

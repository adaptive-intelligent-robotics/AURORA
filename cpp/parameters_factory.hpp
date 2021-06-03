//
// Created by Luca Grillotti on 08/02/2020.
//

#ifndef AURORA_PARAMETERS_FACTORY_HPP
#define AURORA_PARAMETERS_FACTORY_HPP

#include "environments/environments_factory.hpp"
#include "algorithms/algorithms_factory.hpp"
#include "compilation_variables.hpp"

#include "environments/maze/params_maze.hpp"

namespace aurora {
    struct SpecificParams {

        SFERES_CONST aurora::env::Env env = aurora::get_env();
        SFERES_CONST aurora::algo::Algo algo = aurora::get_algo();
        SFERES_CONST aurora::EncoderType encoder_type = aurora::get_encoder_type();


        SFERES_CONST int behav_dim = LATENT_SPACE_SIZE;

        SFERES_CONST double coefficient_proportional_control_l = aurora::get_coefficient_proportional_control_l();
        SFERES_CONST int update_container_period = aurora::get_update_container_period();


//#ifdef NUMBER_LAYERS_LSTM
//        SFERES_CONST int number_layers = NUMBER_LAYERS_LSTM;
//#endif
//
//#ifdef LATENT_SIZE_LAYER
//        SFERES_CONST int latent_size_per_layer = LATENT_SIZE_LAYER;
//#endif

        SFERES_CONST bool use_colors = aurora::get_use_colors();
        SFERES_CONST bool use_videos = aurora::get_use_videos();
        SFERES_CONST bool use_fixed_l = aurora::get_use_fixed_l();

        SFERES_CONST bool do_consider_bumpers_in_obs_for_maze = aurora::get_do_consider_bumpers_in_obs_for_maze();
        SFERES_CONST int lstm_latent_size_per_layer = aurora::get_lstm_latent_size_per_layer();
        SFERES_CONST int lstm_number_layers = aurora::get_lstm_number_layers();
    };

    template<env::Env, typename SpecificParameters>
    struct DefaultParamsFactory {};

    template<typename SpecificParameters>
    struct DefaultParamsFactory<env::Env::hexa_cam_vertical, SpecificParameters> {
        typedef env::ParamsHexapod <SpecificParameters> default_params_t;
    };

    template<typename SpecificParameters>
    struct DefaultParamsFactory<env::Env::hexa_cam_vert_hc_pix, SpecificParameters> {
        typedef env::ParamsHexapod <SpecificParameters> default_params_t;
    };

    template<typename SpecificParameters>
    struct DefaultParamsFactory<env::Env::hexa_gen_desc, SpecificParameters> {
        typedef env::ParamsHexapod <SpecificParameters> default_params_t;
    };

    template<typename SpecificParameters>
    struct DefaultParamsFactory<env::Env::hard_maze, SpecificParameters> {
        typedef env::ParamsMaze <SpecificParameters> default_params_t;
    };

    template<typename SpecificParameters>
    struct DefaultParamsFactory<env::Env::hard_maze_sticky, SpecificParameters> {
      typedef env::ParamsMaze <SpecificParameters> default_params_t;
    };

    template<typename SpecificParameters>
    struct DefaultParamsFactory<env::Env::hard_maze_gen_desc, SpecificParameters> {
      typedef env::ParamsMaze <SpecificParameters> default_params_t;
    };

  template<typename SpecificParameters>
  struct DefaultParamsFactory<env::Env::air_hockey, SpecificParameters> {
    typedef env::ParamsAirHockey<SpecificParameters> default_params_t;
  };

  template<typename SpecificParameters>
  struct DefaultParamsFactory<env::Env::air_hockey_full_traj, SpecificParameters> {
    typedef env::ParamsAirHockey<SpecificParameters> default_params_t;
  };

  template<algo::Algo, typename DefaultParameters, typename SpecificParameters>
  struct ParamsAlgo : public DefaultParameters {

  };

  template<typename DefaultParameters, typename SpecificParameters>
  struct ParamsAlgo<algo::Algo::taxons,
                    DefaultParameters,
                    SpecificParameters> : public DefaultParameters
  {
//    struct evo_float {
//      SFERES_CONST float mutation_rate = 0.2f; // we mutate all the genes with a probability full_gen_mutation_rate
//      SFERES_CONST float cross_rate = 0.0f;
//      SFERES_CONST sferes::gen::evo_float::mutation_t mutation_type = sferes::gen::evo_float::gaussian;
//      SFERES_CONST sferes::gen::evo_float::cross_over_t cross_over_type = sferes::gen::evo_float::no_cross_over;
//      SFERES_CONST float sigma = 0.05f; // ~sqrt(0.05)
//    };

    struct full_gen_mutation {
      SFERES_CONST float full_gen_mutation_rate = aurora::use_elitism_in_taxons() ? 1.0f : 0.9f; // TODO: same with other ones
    };

//    struct pop {
//      // size of a pop
//      SFERES_CONST size_t
//        size = 100;
//      SFERES_CONST size_t
//        nb_gen = 1001;
//      SFERES_CONST size_t
//        dump_period = 50;
//      SFERES_CONST size_t
//        dump_period_aurora = 50;
//    };
  };

  template<typename DefaultParameters, typename SpecificParameters>
  struct ParamsAlgo<algo::Algo::taxo_n,
                    DefaultParameters,
                    SpecificParameters> : public DefaultParameters
  {
//    struct evo_float {
//      SFERES_CONST float mutation_rate = 0.2f; // we mutate all the genes with a probability full_gen_mutation_rate
//      SFERES_CONST float cross_rate = 0.0f;
//      SFERES_CONST sferes::gen::evo_float::mutation_t mutation_type = sferes::gen::evo_float::gaussian;
//      SFERES_CONST sferes::gen::evo_float::cross_over_t cross_over_type = sferes::gen::evo_float::no_cross_over;
//      SFERES_CONST float sigma = 0.05f; // ~sqrt(0.05)
//    };

    struct full_gen_mutation {
      SFERES_CONST float full_gen_mutation_rate = aurora::use_elitism_in_taxons() ? 1.0f : 0.9f; // TODO: same with other ones
    };

//    struct pop {
//      // size of a pop
//      SFERES_CONST size_t
//        size = 100;
//      SFERES_CONST size_t
//        nb_gen = 1001;
//      SFERES_CONST size_t
//        dump_period = 50;
//      SFERES_CONST size_t
//        dump_period_aurora = 50;
//    };
  };

  template<typename DefaultParameters, typename SpecificParameters>
  struct ParamsAlgo<algo::Algo::taxo_s,
                    DefaultParameters,
                    SpecificParameters> : public DefaultParameters
  {
//    struct evo_float {
//      SFERES_CONST float mutation_rate = 0.2f; // we mutate all the genes with a probability full_gen_mutation_rate
//      SFERES_CONST float cross_rate = 0.0f;
//      SFERES_CONST sferes::gen::evo_float::mutation_t mutation_type = sferes::gen::evo_float::gaussian;
//      SFERES_CONST sferes::gen::evo_float::cross_over_t cross_over_type = sferes::gen::evo_float::no_cross_over;
//      SFERES_CONST float sigma = 0.05f; // ~sqrt(0.05)
//    };

    struct full_gen_mutation {
      SFERES_CONST float full_gen_mutation_rate = aurora::use_elitism_in_taxons() ? 1.0f : 0.9f; // TODO: same with other ones
    };

//    struct pop {
//      // size of a pop
//      SFERES_CONST size_t
//        size = 100;
//      SFERES_CONST size_t
//        nb_gen = 1001;
//      SFERES_CONST size_t
//        dump_period = 50;
//      SFERES_CONST size_t
//        dump_period_aurora = 50;
//    };
  };

  template<typename DefaultParameters, typename SpecificParameters>
  struct ParamsAlgo<algo::Algo::hand_coded_taxons,
                    DefaultParameters,
                    SpecificParameters> : public DefaultParameters
  {
//    struct evo_float {
//      SFERES_CONST float mutation_rate = 0.2f; // we mutate all the genes with a probability full_gen_mutation_rate
//      SFERES_CONST float cross_rate = 0.0f;
//      SFERES_CONST sferes::gen::evo_float::mutation_t mutation_type = sferes::gen::evo_float::gaussian;
//      SFERES_CONST sferes::gen::evo_float::cross_over_t cross_over_type = sferes::gen::evo_float::no_cross_over;
//      SFERES_CONST float sigma = 0.05f; // ~sqrt(0.05)
//    };

    struct full_gen_mutation {
      SFERES_CONST float full_gen_mutation_rate = aurora::use_elitism_in_taxons() ? 1.0f : 0.9f; // TODO: same with other ones
    };

//    struct pop {
//      // size of a pop
//      SFERES_CONST size_t
//        size = 100;
//      SFERES_CONST size_t
//        nb_gen = 1001;
//      SFERES_CONST size_t
//        dump_period = 50;
//      SFERES_CONST size_t
//        dump_period_aurora = 50;
//    };
  };
}

#endif //AURORA_PARAMETERS_FACTORY_HPP

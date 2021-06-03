//
// Created by Luca Grillotti on 29/04/2020.
//

#ifndef AURORA_PARAMS_MAZE_HPP
#define AURORA_PARAMS_MAZE_HPP

#include <modules/nn2/gen_dnn_ff.hpp>
#include <modules/nn2/phen_dnn.hpp>
#include "genotype/gen_mlp.hpp"


namespace aurora {
    namespace env {
        template<typename SpecificParams>
        struct ParamsMaze {
            SFERES_CONST int
              min_update_period = 10;
            SFERES_CONST int
              max_update_period = 15000;
            SFERES_CONST int update_exponential_coefficient = 10;

            SFERES_CONST int update_container_period = SpecificParams::update_container_period;
            SFERES_CONST size_t
            image_width = 50;
            SFERES_CONST size_t
            image_height = 50;
            SFERES_CONST int
                    resolution = 10000; // influences l
            SFERES_CONST int
            update_frequency = -1; // -2 means exponentially decaying update frequency
            SFERES_CONST size_t
            times_downsample = 4; // for taking the image

            SFERES_CONST bool use_colors = SpecificParams::use_colors;
            SFERES_CONST bool use_videos = SpecificParams::use_videos;
            SFERES_CONST aurora::env::Env env = SpecificParams::env;
            SFERES_CONST aurora::algo::Algo algo = SpecificParams::algo;
            SFERES_CONST aurora::EncoderType encoder_type  = SpecificParams::encoder_type;
            SFERES_CONST bool do_consider_bumpers_in_obs_for_maze = SpecificParams::do_consider_bumpers_in_obs_for_maze;

            SFERES_CONST int batch_size = 20000;
            SFERES_CONST int nb_epochs = 100; // TO CHANGE

            static int step_measures;

            struct fit_data {
                static boost::shared_ptr<fastsim::Settings> settings;
                static boost::shared_ptr<fastsim::Map> map;
                static boost::shared_ptr<fastsim::DisplaySurface> display;
                static tbb::mutex sdl_mutex;
            };

            static inline int get_one_obs_size() {
                if (env == aurora::env::Env::hard_maze) { // In this case consider the Ground truth = (x,y,theta)
                    return 3;
                } else if (do_consider_bumpers_in_obs_for_maze) {
                    return fit_data::settings->robot()->get_lasers().size() + 2;
                } else {
                    return fit_data::settings->robot()->get_lasers().size();
                }
            }

            static inline bool does_encode_sequence() {
                return (encoder_type == aurora::EncoderType::lstm_ae) || (encoder_type == aurora::EncoderType::conv_seq_ae);
            }

            struct nov {
                static double l;
                static bool use_fixed_l;
                SFERES_CONST double coefficient_proportional_control_l = SpecificParams::coefficient_proportional_control_l;
                SFERES_CONST double k = 15;
                SFERES_CONST double eps = 0.1;
            };

            struct pop {
//        SFERES_CONST size_t init_size = 8000; // to test later
                // size of a batch
                SFERES_CONST size_t
                size = 128;
                //size = 8;
                SFERES_CONST size_t
                nb_gen = 15001;
                SFERES_CONST size_t
                  dump_period = 500;
                SFERES_CONST size_t
                  dump_period_aurora = 500;
            };

            struct evo_float {
                SFERES_CONST float mutation_rate = 0.05f;
                SFERES_CONST float cross_rate = 0.0f;
                SFERES_CONST float eta_m = 10.0f;
                SFERES_CONST float eta_c = 10.0f;
                SFERES_CONST sferes::gen::evo_float::mutation_t mutation_type = sferes::gen::evo_float::mutation_t::polynomial;
                SFERES_CONST sferes::gen::evo_float::cross_over_t cross_over_type = sferes::gen::evo_float::cross_over_t::sbx;
            };
            struct parameters {
                // maximum value of parameters
                SFERES_CONST float min = -5.0f;
                // minimum value
                SFERES_CONST float max = 5.0f;
            };
            struct dnn {
                SFERES_CONST size_t nb_inputs	= 5;
                SFERES_CONST size_t nb_outputs	= 2;
//                SFERES_CONST size_t min_nb_neurons	= 4;
//                SFERES_CONST size_t max_nb_neurons	= 5;
//                SFERES_CONST size_t min_nb_conns	= 50;
//                SFERES_CONST size_t max_nb_conns	= 101;

                SFERES_CONST float m_rate_add_conn	= 0.f;
                SFERES_CONST float m_rate_del_conn	= 0.f;
                SFERES_CONST float m_rate_change_conn = 1.f;
                SFERES_CONST float m_rate_add_neuron  = 0.0f;
                SFERES_CONST float m_rate_del_neuron  = 0.0f;

                SFERES_CONST int io_param_evolving = true;
                SFERES_CONST sferes::gen::dnn::init_t init = sferes::gen::dnn::init_t::ff;
            };
            struct mlp {
                SFERES_CONST size_t layer_0_size = 5;
                SFERES_CONST size_t layer_1_size = 0;
            };


            struct selector {
              SFERES_CONST float proba_picking_selector_1{0.5f};
            };

            struct qd {
                SFERES_CONST size_t behav_dim = SpecificParams::behav_dim;
            };
            struct lstm {
                SFERES_CONST size_t latent_size_per_layer = SpecificParams::lstm_latent_size_per_layer;
                SFERES_CONST size_t number_layers = SpecificParams::lstm_number_layers;
            };

            struct stat {
                SFERES_CONST size_t save_images_period = 5000;
                SFERES_CONST size_t period_saving_individual_in_population = 50;
            };

            struct taxons {
//                SFERES_CONST int nb_max_policies = 5000;
                SFERES_CONST int Q = 5;
            };

          struct vat {
            SFERES_CONST float resolution_multiplicative_constant = 25.f;
          };
        };

        template<typename SpecificParams> double ParamsMaze<SpecificParams>::nov::l;
        template<typename SpecificParams> bool ParamsMaze<SpecificParams>::nov::use_fixed_l;
        template<typename SpecificParams> boost::shared_ptr<fastsim::Settings> ParamsMaze<SpecificParams>::fit_data::settings;
        template<typename SpecificParams> boost::shared_ptr<fastsim::Map> ParamsMaze<SpecificParams>::fit_data::map;
        template<typename SpecificParams> boost::shared_ptr<fastsim::DisplaySurface> ParamsMaze<SpecificParams>::fit_data::display;
        template<typename SpecificParams> tbb::mutex ParamsMaze<SpecificParams>::fit_data::sdl_mutex;
        template<typename SpecificParams> int ParamsMaze<SpecificParams>::step_measures = 10;
    }
}

#endif //AURORA_PARAMS_MAZE_HPP

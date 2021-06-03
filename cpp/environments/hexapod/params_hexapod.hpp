//
// Created by Luca Grillotti on 04/12/2019.
//

#ifndef SFERES2_PARAMS_HEXAPOD_HPP
#define SFERES2_PARAMS_HEXAPOD_HPP


namespace aurora {
    namespace env {
        template<typename SpecificParams>
        struct ParamsHexapod {
            SFERES_CONST int
                    min_update_period = 10;
            SFERES_CONST int
                    max_update_period = 20000;
            SFERES_CONST int update_exponential_coefficient = 10;
            SFERES_CONST int update_container_period = SpecificParams::update_container_period;

            SFERES_CONST size_t
                    image_width = 32; // TODO : Apply Everywhere
            SFERES_CONST size_t
                    image_height = 32; // TODO : Apply Everywhere
            SFERES_CONST int
                    resolution = 5000; // influences l
            SFERES_CONST size_t
                    update_frequency = -1; // -2 means exponentially decaying update frequency
            SFERES_CONST size_t
                    times_downsample = 2; // for taking the image

            SFERES_CONST bool use_colors = SpecificParams::use_colors;
            SFERES_CONST bool use_videos = SpecificParams::use_videos;
            SFERES_CONST aurora::env::Env env = SpecificParams::env;
            SFERES_CONST aurora::algo::Algo algo = SpecificParams::algo;
            SFERES_CONST aurora::EncoderType encoder_type  = SpecificParams::encoder_type;
            SFERES_CONST size_t latent_size_cnn_ae = 10;
            static int step_measures;

            SFERES_CONST int batch_size = 20000;
            SFERES_CONST int nb_epochs = 10000;

            struct nov {
                static double l;
                static bool use_fixed_l;
                SFERES_CONST double coefficient_proportional_control_l = SpecificParams::coefficient_proportional_control_l;
                SFERES_CONST double k = 15;
                SFERES_CONST double eps = 0.1;
            };
            static inline bool does_encode_sequence() {
                return (encoder_type == aurora::EncoderType::lstm_ae) || (encoder_type == aurora::EncoderType::conv_seq_ae);
            }

            static inline int get_one_obs_size() {
                /*
                 * Get size of one observation in sequence
                 * This method should appear in every Params Class
                 */
                if (use_colors) {
                    return 3 * image_width * image_height;
                } else {
                    return image_width * image_height;
                }
            }

            struct pop {
//        SFERES_CONST size_t init_size = 8000; // to test later
                // size of a batch
                SFERES_CONST size_t
                        size = 128;
                //size = 8;
                SFERES_CONST size_t
                        nb_gen = 30001;
                SFERES_CONST size_t
                  dump_period = 500;
                SFERES_CONST size_t
                  dump_period_aurora = 500;
            };
            struct parameters {
                // used to avoid really degenerated experiments with trajectories not doing anything.
                SFERES_CONST float min = 0.0;
                SFERES_CONST float max = 1.0;
            };

            struct evo_float {
                SFERES_CONST float cross_rate = 0.0f;
                SFERES_CONST float mutation_rate = 0.05f;
                SFERES_CONST float eta_m = 10.0f;
                SFERES_CONST float eta_c = 10.0f;
                SFERES_CONST sferes::gen::evo_float::mutation_t
                        mutation_type = sferes::gen::evo_float::mutation_t::polynomial;
                SFERES_CONST sferes::gen::evo_float::cross_over_t
                        cross_over_type = sferes::gen::evo_float::cross_over_t::sbx;
            };

            struct selector {
                SFERES_CONST float proba_picking_selector_1{0.5f};
            };

            struct qd {
                SFERES_CONST size_t behav_dim = SpecificParams::behav_dim;
            };
            struct stat {
                SFERES_CONST size_t save_images_period = 10000;
                SFERES_CONST size_t period_saving_individual_in_population = 50;
            };

            struct lstm {
                SFERES_CONST size_t latent_size_per_layer = SpecificParams::lstm_latent_size_per_layer;
                SFERES_CONST size_t number_layers = SpecificParams::lstm_number_layers;
            };

            struct taxons {
//                SFERES_CONST int nb_max_policies = 5000;
                SFERES_CONST int Q = 5;
            };

          struct vat {
            SFERES_CONST float resolution_multiplicative_constant = 18.f;
          };

        };
        template<typename SpecificParams> double ParamsHexapod<SpecificParams>::nov::l;
        template<typename SpecificParams> bool ParamsHexapod<SpecificParams>::nov::use_fixed_l;
        template<typename SpecificParams> int ParamsHexapod<SpecificParams>::step_measures = 1;

    }
}


#endif //SFERES2_PARAMS_HEXAPOD_HPP

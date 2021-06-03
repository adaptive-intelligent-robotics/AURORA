//
// Created by Luca Grillotti on 03/04/2020.
//

#ifndef AURORA_COMPILATION_VARIABLES_HPP
#define AURORA_COMPILATION_VARIABLES_HPP

namespace aurora {

    namespace algo {
        enum class Algo {
            aurora_curiosity,
            aurora_novelty,
            aurora_surprise,
            aurora_nov_sur,
            aurora_uniform,

            hand_coded_qd,
            hand_coded_qd_no_sel,
            hand_coded_taxons,

            taxons,
            taxo_n,
            taxo_s,
        };
    }

    namespace env {
        enum class Env {
            hexa_cam_vertical,
            hexa_cam_vert_hc_pix,
            hexa_gen_desc,

            // hard maze
            hard_maze,
            hard_maze_sticky,
            hard_maze_gen_desc,

            // Air Hockey
            air_hockey,
            air_hockey_full_traj
        };
    }

    enum class EncoderType {
        none,
        cnn_ae,
        strg_cnn,
        lstm_ae,
        video_ae,
        conv_seq_ae,
        pca,
        mlp_ae
    };

    constexpr bool strings_equal(char const *a, char const *b) {
        return *a == *b && (*a == '\0' || strings_equal(a + 1, b + 1));
    }

    constexpr env::Env get_env() {
        if (strings_equal(ENVIRONMENT, "hexa_cam_vertical")) {
            return env::Env::hexa_cam_vertical;
        } else if (strings_equal(ENVIRONMENT, "hexa_cam_vert_hc_pix")) {
            return env::Env::hexa_cam_vert_hc_pix;
        } else if (strings_equal(ENVIRONMENT, "hexa_gen_desc")) {
            return env::Env::hexa_gen_desc;
        } else if (strings_equal(ENVIRONMENT, "hard_maze")) {
            return env::Env::hard_maze;
        } else if (strings_equal(ENVIRONMENT, "hard_maze_sticky")) {
          return env::Env::hard_maze_sticky;
        } else if (strings_equal(ENVIRONMENT, "hard_maze_gen_desc")) {
          return env::Env::hard_maze_gen_desc;
        } else if (strings_equal(ENVIRONMENT, "air_hockey")) {
          return env::Env::air_hockey;
        } else if (strings_equal(ENVIRONMENT, "air_hockey_full_traj")) {
          return env::Env::air_hockey_full_traj;
        }
    }

    constexpr bool is_env_hard_maze() {
        return (
                (get_env() == env::Env::hard_maze)
                || (get_env() == env::Env::hard_maze_sticky)
                || (get_env() == env::Env::hard_maze_gen_desc)
        );
    }

    constexpr algo::Algo get_algo() {
        if (strings_equal(ALGORITHM, "aurora_curiosity")) {
            return algo::Algo::aurora_curiosity;
        } else if (strings_equal(ALGORITHM, "aurora_novelty")) {
            return algo::Algo::aurora_novelty;
        } else if (strings_equal(ALGORITHM, "aurora_surprise")) {
            return algo::Algo::aurora_surprise;
        } else if (strings_equal(ALGORITHM, "aurora_nov_sur")) {
            return algo::Algo::aurora_nov_sur;
        } else if (strings_equal(ALGORITHM, "aurora_uniform")) {
                return algo::Algo::aurora_uniform;
        } else if (strings_equal(ALGORITHM, "hand_coded_qd")) {
            return algo::Algo::hand_coded_qd;
        } else if (strings_equal(ALGORITHM, "hand_coded_qd_no_sel")) {
            return algo::Algo::hand_coded_qd_no_sel;
        } else if (strings_equal(ALGORITHM, "hand_coded_taxons")) {
            return algo::Algo::hand_coded_taxons;
        } else if (strings_equal(ALGORITHM, "taxons")) {
            return algo::Algo::taxons;
        } else if (strings_equal(ALGORITHM, "taxo_n")) {
            return algo::Algo::taxo_n;
        } else if (strings_equal(ALGORITHM, "taxo_s")) {
            return algo::Algo::taxo_s;
        }
    }

    constexpr bool is_algo_hand_coded() {
        return (
                (get_algo() == algo::Algo::hand_coded_qd)
                || (get_algo() == algo::Algo::hand_coded_qd_no_sel)
                || (get_algo() == algo::Algo::hand_coded_taxons)
                );
    }

    constexpr bool is_algo_taxons_based() {
      return (
        (get_algo() == algo::Algo::taxons)
        || (get_algo() == algo::Algo::taxo_s)
        || (get_algo() == algo::Algo::taxo_n)
        || (get_algo() == algo::Algo::hand_coded_taxons)
        );
    }

    constexpr EncoderType get_encoder_type() {
        if (strings_equal(ENCODER_TYPE, "cnn_ae")) {
            return aurora::EncoderType::cnn_ae;
        } else if (strings_equal(ENCODER_TYPE, "strg_cnn")) {
          return aurora::EncoderType::strg_cnn;
        } else if (strings_equal(ENCODER_TYPE, "lstm_ae")) {
            return aurora::EncoderType::lstm_ae;
        } else if (strings_equal(ENCODER_TYPE, "video_ae")) {
            return aurora::EncoderType::video_ae;
        } else if (strings_equal(ENCODER_TYPE, "conv_seq_ae")) {
            return aurora::EncoderType::conv_seq_ae;
        } else if (strings_equal(ENCODER_TYPE, "pca")) {
          return aurora::EncoderType::pca;
        } else if (strings_equal(ENCODER_TYPE, "mlp_ae")) {
          return aurora::EncoderType::mlp_ae;
        } else if (strings_equal(ENCODER_TYPE, "none")) {
            return aurora::EncoderType::none;
        }
    }

    inline constexpr bool
    does_encode_sequence()
    {
      return (get_encoder_type() == aurora::EncoderType::lstm_ae) ||
             (get_encoder_type() == aurora::EncoderType::conv_seq_ae) ||
             (get_encoder_type() == aurora::EncoderType::pca) ||
             (get_encoder_type() == aurora::EncoderType::mlp_ae);
    }

    inline constexpr bool
    does_encode_images() {
      return (get_encoder_type() == aurora::EncoderType::cnn_ae)
             or (get_encoder_type() == aurora::EncoderType::strg_cnn);
    }

    constexpr int format_lp_norm_variable(char const *lp_norm) {
        if (strings_equal(lp_norm, "inf")) {
            return Eigen::Infinity;
        }
    }

    constexpr int format_lp_norm_variable(int lp_norm) {
        if (lp_norm >= 1) {
            return lp_norm;
        }
    }

    constexpr int get_lp_norm() {
#ifdef LP_NORM
        return format_lp_norm_variable(LP_NORM);
#else
        return 2; // Euclidian Norm by default
#endif
    }

    constexpr bool get_use_colors() {
#ifdef USE_COLORS
        return true;
#else
        return false;
#endif
    }

    constexpr bool get_use_videos() {
#ifdef USE_VIDEOS
        return true;
#else
        return false;
#endif
    }

    constexpr bool get_use_fixed_l() {
#ifdef PARAMS_FIXED_L
        return true;
#else
        return false;
#endif
    }

    constexpr bool get_do_consider_bumpers_in_obs_for_maze() {
#ifdef DO_CONSIDER_BUMPERS_MAZE
        return true;
#else
        return false;
#endif
    }

    constexpr int get_lstm_latent_size_per_layer() {
#ifdef LSTM_LATENT_SIZE_PER_LAYER
        return LSTM_LATENT_SIZE_PER_LAYER;
#else
        return -1;
#endif
    }

    constexpr int get_lstm_number_layers() {
#ifdef LSTM_NUMBER_LAYERS
        return LSTM_NUMBER_LAYERS;
#else
        return -1;
#endif
    }

    inline
      constexpr bool get_has_fit() {
#ifdef HAS_FIT
      return true;
#else
      return false;
#endif
    }

  inline
  constexpr bool use_volume_adaptive_threshold() {
#ifdef VAT
    return true;
#else
    return false;
#endif
  }

  inline
  constexpr bool no_normalisation_bd_from_latent_space() {
#ifdef NO_NORMALISE
    return true;
#else
    return false;
#endif
  }

  inline
  constexpr bool use_elitism_in_taxons() {
#ifdef TAX_ELI
    return true;
#else
    return false;
#endif
  }

  constexpr double
  get_coefficient_proportional_control_l()
  {
#ifdef ALPHA_L
    return ALPHA_L;
#else
    return 5e-6;
#endif
  }

  constexpr int
  get_update_container_period()
  {
#ifdef T_UPDATE
    return T_UPDATE;
#else
    return 10;
#endif
  }
}
#undef ALGORITHM

#endif //AURORA_COMPILATION_VARIABLES_HPP

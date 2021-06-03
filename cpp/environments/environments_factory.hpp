//
// Created by Luca Grillotti on 07/02/2020.
//

#ifndef AURORA_ENVIRONMENTS_FACTORY_HPP
#define AURORA_ENVIRONMENTS_FACTORY_HPP

#include <modules/nn2/gen_dnn_ff.hpp>
#include <modules/nn2/phen_dnn.hpp>

#include "genotype/gen_mlp.hpp"

#include "compilation_variables.hpp"

#include "environments/hexapod/fit_hexapod.hpp"
#include "environments/maze/fit_maze.hpp"
#include "environments/air_hockey/fit_air_hockey.hpp"

namespace aurora {
    namespace env {
        template<Env, typename Params>
        struct Environment {};

        template<typename Params>
        struct Environment<Env::hexa_cam_vertical, Params> {
            typedef Params param_t;
            typedef sferes::gen::EvoFloat<36, param_t> gen_t;
            typedef aurora::env::FitHexapod<param_t> fit_t;
            typedef sferes::phen::Parameters <gen_t, fit_t, param_t> phen_t;
        };

        template<typename Params>
        struct Environment<Env::hexa_cam_vert_hc_pix, Params> {
            typedef Params param_t;
            typedef sferes::gen::EvoFloat<36, param_t> gen_t;
            typedef aurora::env::FitHexapod<param_t> fit_t;
            typedef sferes::phen::Parameters <gen_t, fit_t, param_t> phen_t;
        };

        template<typename Params>
        struct Environment<Env::hexa_gen_desc, Params> {
            typedef Params param_t;
            typedef sferes::gen::EvoFloat<36, param_t> gen_t;
            typedef aurora::env::FitHexapod<param_t> fit_t;
            typedef sferes::phen::Parameters <gen_t, fit_t, param_t> phen_t;
        };

        template<typename Params>
        struct Environment<Env::hard_maze, Params> {
            typedef Params param_t;

            typedef sferes::phen::Parameters<sferes::gen::EvoFloat<1, param_t>, sferes::fit::FitDummy<>, param_t> weight_t;
            typedef sferes::phen::Parameters<sferes::gen::EvoFloat<1, param_t>, sferes::fit::FitDummy<>, param_t> bias_t;
            typedef ::nn::PfWSum<weight_t> pf_t;
            typedef ::nn::AfTanh<bias_t> af_t;
            typedef ::nn::Neuron<pf_t, af_t> neuron_t;
            typedef ::nn::Connection<weight_t> connection_t;

            typedef sferes::gen::GenMlp<neuron_t, connection_t, param_t> gen_t;
            typedef aurora::env::HardMaze<param_t> fit_t;
            typedef sferes::phen::Dnn<gen_t, fit_t, param_t> phen_t;
        };

        template<typename Params>
      struct Environment<Env::hard_maze_sticky, Params> {
        typedef Params param_t;

        typedef sferes::phen::Parameters<sferes::gen::EvoFloat<1, param_t>, sferes::fit::FitDummy<>, param_t> weight_t;
        typedef sferes::phen::Parameters<sferes::gen::EvoFloat<1, param_t>, sferes::fit::FitDummy<>, param_t> bias_t;
        typedef ::nn::PfWSum<weight_t> pf_t;
        typedef ::nn::AfTanh<bias_t> af_t;
        typedef ::nn::Neuron<pf_t, af_t> neuron_t;
        typedef ::nn::Connection<weight_t> connection_t;

        typedef sferes::gen::GenMlp<neuron_t, connection_t, param_t> gen_t;
        typedef aurora::env::HardMaze<param_t> fit_t;
        typedef sferes::phen::Dnn<gen_t, fit_t, param_t> phen_t;
      };

      template<typename Params>
      struct Environment<Env::hard_maze_gen_desc, Params> {
        typedef Params param_t;

        typedef sferes::phen::Parameters<sferes::gen::EvoFloat<1, param_t>, sferes::fit::FitDummy<>, param_t> weight_t;
        typedef sferes::phen::Parameters<sferes::gen::EvoFloat<1, param_t>, sferes::fit::FitDummy<>, param_t> bias_t;
        typedef ::nn::PfWSum<weight_t> pf_t;
        typedef ::nn::AfTanh<bias_t> af_t;
        typedef ::nn::Neuron<pf_t, af_t> neuron_t;
        typedef ::nn::Connection<weight_t> connection_t;

        typedef sferes::gen::GenMlp<neuron_t, connection_t, param_t> gen_t;
        typedef aurora::env::HardMaze<param_t> fit_t;
        typedef sferes::phen::Dnn<gen_t, fit_t, param_t> phen_t;
      };

        // AIR HOCKEY Environments //

      template<typename Params>
      struct Environment<Env::air_hockey, Params> {
        typedef Params param_t;
        typedef sferes::gen::EvoFloat<8, param_t> gen_t;
        typedef aurora::env::AirHockey<param_t> fit_t;
        typedef sferes::phen::Parameters <gen_t, fit_t, param_t> phen_t;
      };

      template<typename Params>
      struct Environment<Env::air_hockey_full_traj, Params> {
        typedef Params param_t;
        typedef sferes::gen::EvoFloat<8, param_t> gen_t;
        typedef aurora::env::AirHockey<param_t> fit_t;
        typedef sferes::phen::Parameters <gen_t, fit_t, param_t> phen_t;
      };


        template<typename TEnvironment, typename TGen>
        struct EnvironmentChangeGen {
          typedef typename TEnvironment::param_t param_t;

          typedef TGen gen_t;

          typedef typename TEnvironment::fit_t fit_t;

          typedef
            typename sferes::phen::ChangeGenotypeTypeForPhenotype<typename TEnvironment::phen_t,
                                                                  gen_t>::phen_t
              phen_t;
        };

        template<typename TEnvironment>
        struct EnvironmentAllGenMutation {
          typedef EnvironmentChangeGen<TEnvironment,
                               sferes::gen::FullGenotypeMutationWrapper<typename TEnvironment::gen_t,
                                                                        typename TEnvironment::param_t>>
            environment_t;
        };
    }
}

#endif //AURORA_ENVIRONMENTS_FACTORY_HPP

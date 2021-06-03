//
// Created by Luca Grillotti on 13/06/2020.
//

#ifndef AURORA_ENVIRONMENT_INITIALISER_HPP
#define AURORA_ENVIRONMENT_INITIALISER_HPP

namespace aurora {
    template <typename Params>
    struct EnvironmentInitialiser {};

    template <typename SpecificParams>
    struct EnvironmentInitialiser<aurora::env::ParamsHexapod<SpecificParams>> {
        void run(bool use_video=false) {
          const bool use_meshes = use_video;
          aurora::env::load_and_init_robot(use_meshes);
        }
    };

    template <typename SpecificParams>
    struct EnvironmentInitialiser<aurora::env::ParamsMaze<SpecificParams>> {
        void run(bool use_video=false) {
            aurora::env::init_fastsim_settings<aurora::env::ParamsMaze<SpecificParams>>();
        }
    };

  template <typename SpecificParams>
  struct EnvironmentInitialiser<aurora::env::ParamsAirHockey<SpecificParams>> {
    void run(bool use_video=false) {}
  };
}

#endif //AURORA_ENVIRONMENT_INITIALISER_HPP

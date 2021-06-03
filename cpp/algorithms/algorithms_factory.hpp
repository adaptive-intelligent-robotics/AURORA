//
// Created by Luca Grillotti on 07/02/2020.
//

#ifndef AURORA_ALGORITHMS_FACTORY_HPP
#define AURORA_ALGORITHMS_FACTORY_HPP

#include "compilation_variables.hpp"

#include "algorithms/taxons/taxons_evolutionary_algorithm.hpp"
#include "algorithms/aurora/definitions_aurora.hpp"
#include "algorithms/taxons/definitions_taxons.hpp"

#include "algorithms/hand_coded/hand_coded_qd.hpp"
#include "algorithms/hand_coded/novelty_search.hpp"

#include "environments/environments_factory.hpp"

namespace aurora {
    namespace algo {

        template<Algo, typename Environment>
        struct AlgorithmFactory {};

        template<typename Environment>
        struct AlgorithmFactory<Algo::aurora_curiosity, Environment> {
            typedef AuroraCuriosity<Environment, aurora::get_encoder_type(), aurora::get_lp_norm()> algo_t;
        };

        template<typename Environment>
        struct AlgorithmFactory<Algo::aurora_novelty, Environment> {
            typedef AuroraNovelty<Environment, aurora::get_encoder_type(), aurora::get_lp_norm()> algo_t;
        };

        template<typename Environment>
        struct AlgorithmFactory<Algo::aurora_surprise, Environment> {
            typedef AuroraSurprise<Environment, aurora::get_encoder_type(), aurora::get_lp_norm()> algo_t;
        };

        template<typename Environment>
        struct AlgorithmFactory<Algo::aurora_nov_sur, Environment> {
            typedef AuroraNoveltySurprise<Environment, aurora::get_encoder_type(), aurora::get_lp_norm()> algo_t;
        };

        template<typename Environment>
        struct AlgorithmFactory<Algo::aurora_uniform, Environment> {
            typedef AuroraUniform<Environment, aurora::get_encoder_type(), aurora::get_lp_norm()> algo_t;
        };

        template<typename Environment>
        struct AlgorithmFactory<Algo::hand_coded_qd, Environment> {
            typedef HandCodedQD<Environment, aurora::get_lp_norm()> algo_t;
        };

        template<typename Environment>
        struct AlgorithmFactory<Algo::hand_coded_qd_no_sel, Environment> {
            typedef HandCodedQD<Environment,
                                    aurora::get_lp_norm(),
                                    sferes::qd::selector::NoSelection<typename Environment::phen_t,
                                                                      typename Environment::param_t>
                                    > algo_t;
        };

        template<typename Environment>
        struct AlgorithmFactory<Algo::hand_coded_taxons, Environment> {
            typedef NoveltySearchTaxons<typename aurora::env::EnvironmentAllGenMutation<Environment>::environment_t,
                                        aurora::get_lp_norm()> algo_t;
        };

        template<typename Environment>
        struct AlgorithmFactory<Algo::taxons, Environment> {
            typedef Taxons<typename aurora::env::EnvironmentAllGenMutation<Environment>::environment_t, aurora::get_encoder_type(), aurora::get_lp_norm()> algo_t;
        };

        template<typename Environment>
        struct AlgorithmFactory<Algo::taxo_n, Environment> {
            typedef Taxo_n<typename aurora::env::EnvironmentAllGenMutation<Environment>::environment_t, aurora::get_encoder_type(), aurora::get_lp_norm()> algo_t;
        };

        template<typename Environment>
        struct AlgorithmFactory<Algo::taxo_s, Environment> {
            typedef Taxo_s<typename aurora::env::EnvironmentAllGenMutation<Environment>::environment_t, aurora::get_encoder_type(), aurora::get_lp_norm()> algo_t;
        };

    }
}

#endif //AURORA_ALGORITHMS_FACTORY_HPP

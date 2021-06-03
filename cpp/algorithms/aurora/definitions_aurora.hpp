#ifndef AURORA_DEFINITIONS_AURORA_HPP
#define AURORA_DEFINITIONS_AURORA_HPP

#include <sferes/stat/state_qd.hpp>

#include "genotype/all_gen_mutation.hpp"
#include "phenotype/change_gen_in_phen.hpp"

#include "algorithms/container/general_distance_archive.hpp"
#include "algorithms/container/general_sort_based_storage.hpp"
#include "algorithms/initialise_global_parameters.hpp"
#include "algorithms/quality_diversity_aurora_project.hpp"
#include "algorithms/value_sorter/value_sorter.hpp"
#include "algorithms/value_sorter/value_sorter_variable.hpp"

#include "stat/stat_offspring.hpp"

namespace aurora {
  namespace algo {

    template<typename Environment, typename TSelector, EncoderType encoder_type, int PNorm>
    struct AuroraAbstract
    {
      typedef typename Environment::param_t param_t;
      typedef typename Environment::gen_t gen_t;
      typedef typename Environment::fit_t fit_t;
      typedef typename Environment::phen_t phen_t;

      typedef typename aurora::EncoderTypeFactory<param_t, encoder_type>::network_loader_t network_loader_t;

      typedef sferes::modif::DimensionalityReduction<phen_t, param_t, network_loader_t> modifier_t;

      // For the Archive, you can chose one of the following storage:
      // kD_tree storage, recommended for small behavioral descriptors (behav_dim<10)

      // Sort_based storage, recommended for larger behavioral descriptors.
      typedef sferes::qd::container::DistancePNorm<PNorm> distance_t;

      typedef typename std::conditional<
        (param_t::qd::behav_dim < 10) and (PNorm == 2),
        sferes::qd::container::KdtreeStorage<boost::shared_ptr<phen_t>, param_t::qd::behav_dim>,
        sferes::qd::container::GeneralDistanceSortBasedStorage<boost::shared_ptr<phen_t>, distance_t>>::type
        storage_t;

      typedef sferes::qd::container::GeneralDistanceArchive<phen_t, storage_t, param_t, distance_t>
        container_t;

      typedef sferes::eval::Parallel<param_t> eval_t;
      // typedef eval::Eval<Params> eval_t;

      typedef boost::fusion::vector<sferes::stat::CurrentGen<phen_t, param_t>,
                                    sferes::stat::QdContainer<phen_t, param_t>,
                                    sferes::stat::QdProgress<phen_t, param_t>,
                                    sferes::stat::Projection<phen_t, param_t>,
                                    sferes::stat::ImagesObservations<phen_t, param_t>,
                                    sferes::stat::ImagesReconstructionObs<phen_t, param_t>,
                                    //                    sferes::stat::ModelAutoencoder<phen_t, param_t>,
                                    sferes::stat::Modifier<phen_t, param_t>,
                                    sferes::stat::SuccessiveGT<phen_t, param_t>>
        stat_t;

      typedef TSelector selector_t;

      typedef sferes::qd::
        QualityDiversityAuroraProject<phen_t, eval_t, stat_t, modifier_t, selector_t, container_t, param_t>
          ea_t;

      static void
      update_parameters()
      {
        initialise_global_variables<param_t>(); // Initialise l
      }
    };

    template<typename Environment, EncoderType encoder_type, int PNorm>
    using AuroraCuriosity =
      AuroraAbstract<Environment,
                     // Curiosity Selectors (as in Aurora Paper).
                     sferes::qd::selector::ScoreProportionate<typename Environment::phen_t,
                                                              sferes::qd::selector::getCuriosity,
                                                              typename Environment::param_t>,
                     encoder_type,
                     PNorm>;

    template<typename Environment, EncoderType encoder_type, int PNorm>
    using AuroraNoveltySurprise =
      AuroraAbstract<Environment,
                     // Combination of Surprise and Novelty Selectors (as in Taxons Paper). Proba of selection
                     // to give in Params
                     sferes::qd::selector::VariableSelector<
                       typename Environment::phen_t,
                       sferes::qd::selector::ScoreProportionate<typename Environment::phen_t,
                                                                sferes::qd::selector::getNovelty,
                                                                typename Environment::param_t>,
                       sferes::qd::selector::ScoreProportionate<typename Environment::phen_t,
                                                                sferes::qd::selector::getSurprise,
                                                                typename Environment::param_t>,
                       typename Environment::param_t>,
                     encoder_type,
                     PNorm>;

    template<typename Environment, EncoderType encoder_type, int PNorm>
    using AuroraSurprise =
      AuroraAbstract<Environment,
                     // Surprise Only Score Proportionate Selector
                     sferes::qd::selector::ScoreProportionate<typename Environment::phen_t,
                                                              sferes::qd::selector::getSurprise,
                                                              typename Environment::param_t>,
                     encoder_type,
                     PNorm>;

    template<typename Environment, EncoderType encoder_type, int PNorm>
    using AuroraNovelty =
      AuroraAbstract<Environment,
                     // Novelty Only Score Proportionate Selector
                     sferes::qd::selector::ScoreProportionate<typename Environment::phen_t,
                                                              sferes::qd::selector::getNovelty,
                                                              typename Environment::param_t>,
                     encoder_type,
                     PNorm>;

    template<typename Environment, EncoderType encoder_type, int PNorm>
    using AuroraUniform = AuroraAbstract<
      Environment,
      // Uniform Only Score Proportionate Selector
      sferes::qd::selector::Uniform<typename Environment::phen_t, typename Environment::param_t>,
      encoder_type,
      PNorm>;
  } // namespace algo
} // namespace aurora

#endif // AURORA_DEFINITIONS_AURORA_HPP

#ifndef AURORA_DEFINITIONS_TAXONS_HPP
#define AURORA_DEFINITIONS_TAXONS_HPP

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

    template<typename Environment, typename TValueSorter, EncoderType encoder_type, int PNorm>
    struct TaxonsAbstract
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

      // typedef typename std::conditional<
      // (param_t::qd::behav_dim < 10) and (PNorm == 2),
      // sferes::qd::container::KdtreeStorage<boost::shared_ptr<phen_t>, param_t::qd::behav_dim>,
      // sferes::qd::container::GeneralDistanceSortBasedStorage<boost::shared_ptr<phen_t>, distance_t>
      // >::type storage_t;
      typedef sferes::qd::container::GeneralDistanceSortBasedStorage<boost::shared_ptr<phen_t>, distance_t>
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
                                    sferes::stat::SuccessiveGT<phen_t, param_t>,
                                    sferes::stat::QdOffspring<phen_t, param_t>>
        stat_t;

      typedef TValueSorter value_sorter_t;

      typedef TaxonsEvolutionaryAlgorithm<phen_t, eval_t, stat_t, modifier_t, value_sorter_t, container_t, param_t>
        ea_t; // AURORA

      static void
      update_parameters()
      {
        initialise_global_variables<param_t>(); // Initialise l
      }
    };

    template<typename Environment, EncoderType encoder_type, int PNorm>
    using Taxons = TaxonsAbstract<
      Environment,
      value_sorter::ValueVariableSorter<typename Environment::phen_t,
                                        value_sorter::ValueSorter<typename Environment::phen_t,
                                                                  sferes::qd::selector::getNovelty,
                                                                  typename Environment::param_t>,
                                        value_sorter::ValueSorter<typename Environment::phen_t,
                                                                  sferes::qd::selector::getSurprise,
                                                                  typename Environment::param_t>,
                                        typename Environment::param_t>,
      encoder_type,
      PNorm>;

    template<typename Environment, EncoderType encoder_type, int PNorm>
    using Taxo_n = TaxonsAbstract<Environment,
                                  value_sorter::ValueSorter<typename Environment::phen_t,
                                                            sferes::qd::selector::getNovelty,
                                                            typename Environment::param_t>,
                                  encoder_type,
                                  PNorm>;

    template<typename Environment, EncoderType encoder_type, int PNorm>
    using Taxo_s = TaxonsAbstract<Environment,
                                  value_sorter::ValueSorter<typename Environment::phen_t,
                                                            sferes::qd::selector::getSurprise,
                                                            typename Environment::param_t>,
                                  encoder_type,
                                  PNorm>;
  } // namespace algo
} // namespace aurora

#endif // AURORA_DEFINITIONS_TAXONS_HPP

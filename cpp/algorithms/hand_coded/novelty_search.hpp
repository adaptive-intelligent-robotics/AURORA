#ifndef AURORA_NOVELTY_SEARCH_TAXONS_HPP
#define AURORA_NOVELTY_SEARCH_TAXONS_HPP

#include <sferes/modif/novelty.hpp>

#include "algorithms/container/general_distance_archive.hpp"
#include "algorithms/container/general_sort_based_storage.hpp"
#include "algorithms/initialise_global_parameters.hpp"
#include "algorithms/quality_diversity_aurora_project.hpp"
#include "algorithms/value_sorter/value_sorter.hpp"

#include "modifier/dummy_serialisable.hpp"

namespace aurora {
  namespace algo {
    /// Definition of standard novelty search
    /// \tparam Environment Structure containing param_t, gen_t, fit_t, and phen_t
    /// \tparam PNorm which norm to use ? (for example, put 2 if using Euclidian Norm) -- 2 is used by default
    template<typename Environment, int PNorm>
    struct NoveltySearchTaxons
    {
      typedef typename Environment::param_t param_t;
      typedef typename Environment::gen_t gen_t;
      typedef typename Environment::fit_t fit_t;
      typedef typename Environment::phen_t phen_t;

      typedef sferes::modif::DummySerialisable<> modifier_t;

      // For the Archive, you can chose one of the following storage:
      // kD_tree storage, recommended for small behavioral descriptors (behav_dim<10)

      typedef sferes::qd::container::DistancePNorm<PNorm> distance_t;

      // typedef typename std::conditional<
      // (param_t::qd::behav_dim < 10) and (PNorm == 2),
      // sferes::qd::container::KdtreeStorage<boost::shared_ptr <phen_t>, param_t::qd::behav_dim >,
      // sferes::qd::container::GeneralDistanceSortBasedStorage< boost::shared_ptr<phen_t>, distance_t>
      // >::type storage_t;
      typedef sferes::qd::container::GeneralDistanceSortBasedStorage<boost::shared_ptr<phen_t>, distance_t>
        storage_t;

      //            typedef sferes::qd::container::KdtreeStorage<boost::shared_ptr <phen_t>,
      //            param_t::qd::behav_dim > storage_t;
      // Sort_based storage, recommended for larger behavioral descriptors.
      // typedef qd::container::SortBasedStorage< boost::shared_ptr<phen_t> > storage_t;

      typedef sferes::qd::container::GeneralDistanceArchive<phen_t, storage_t, param_t, distance_t>
        container_t;

      typedef sferes::eval::Parallel<param_t> eval_t;
      // typedef eval::Eval<Params> eval_t;

      typedef boost::fusion::vector<sferes::stat::CurrentGen<phen_t, param_t>,
                                    sferes::stat::QdContainer<phen_t, param_t>,
                                    sferes::stat::QdProgress<phen_t, param_t>,
                                    sferes::stat::Projection<phen_t, param_t>,
                                    sferes::stat::Modifier<phen_t, param_t>,
                                    sferes::stat::SuccessiveGT<phen_t, param_t>,
                                    sferes::stat::ImagesObservations<phen_t, param_t>,
                                    sferes::stat::QdOffspring<phen_t, param_t>>
        stat_t;

      typedef value_sorter::ValueSorter<phen_t, sferes::qd::selector::getNovelty, param_t> value_sorter_t;
      typedef TaxonsEvolutionaryAlgorithm<phen_t, eval_t, stat_t, modifier_t, value_sorter_t, container_t, param_t>
        ea_t; // True Novelty Search

      static void
      update_parameters()
      {
        initialise_global_variables<param_t>(); // Initialise l
      }
    };
  } // namespace algo
} // namespace aurora

#endif // AURORA_NOVELTY_SEARCH_TAXONS_HPP

//
// Created by Luca Grillotti on 11/08/2020.
//

#ifndef AURORA_TAXONS_EVOLUTIONARY_ALGORITHM_HPP
#define AURORA_TAXONS_EVOLUTIONARY_ALGORITHM_HPP

#include <sferes/qd/quality_diversity.hpp>

#include <sferes/eval/parallel.hpp>
#include <sferes/qd/selector/noselection.hpp>
#include "dbg_tools/dbg.hpp"
#include "algorithms/quality_diversity_aurora_project.hpp"
#include "algorithms/container/general_sort_based_storage.hpp"
#include "algorithms/container/general_distance_archive.hpp"


namespace aurora {
    namespace algo {

        template<typename Phen, typename Eval, typename Stat, typename FitModifier, typename ValueSorter, typename Container, typename Params, typename Exact = stc::Itself, int PNorm = 2>
        class TaxonsEvolutionaryAlgorithm
        : public sferes::qd::QualityDiversityAuroraProject<
                        Phen,
                        Eval,
                        Stat,
                        FitModifier,
                        sferes::qd::selector::NoSelection<Phen, Params>,
                        Container,
                        Params,
                        typename stc::FindExact<TaxonsEvolutionaryAlgorithm <Phen, Eval, Stat, FitModifier, ValueSorter, Container, Params, Exact>, Exact>::ret
                > {
            /**
             * TAXONS Algorithm as described in original Paper
             * https://arxiv.org/abs/1909.05508
             *
             * Parameters:
             * - Params::pop::size
             * - Params::taxons::Q
             */

        public:
            typedef boost::shared_ptr <Phen> indiv_t;
            typedef std::vector<indiv_t> pop_t;
            typedef Phen phen_t;
            typedef Params param_t;

            typedef sferes::qd::container::DistancePNorm<PNorm> distance_t;

            // typedef typename std::conditional<
                    // (param_t::qd::behav_dim < 10) and (PNorm == 2),
                    // sferes::qd::container::KdtreeStorage<boost::shared_ptr <phen_t>, param_t::qd::behav_dim >,
                    // sferes::qd::container::GeneralDistanceSortBasedStorage< boost::shared_ptr<phen_t>, distance_t>
            // >::type storage_t;
            typedef typename Container::storage_t storage_t;

            void random_pop() {
                // we always add all Q best individuals in TAXONS
                Params::nov::l = -1.;
                Params::nov::use_fixed_l = true;

                sferes::parallel::init();

                this->_pop.clear();

                this->_offspring.resize(Params::pop::size);
                for (indiv_t &indiv : this->_offspring) {
                    indiv = indiv_t(new Phen());
                    indiv->random();
                }
                this->_eval_pop(this->_offspring, 0, this->_offspring.size());
                this->apply_modifier();

                _add(this->_offspring);

                this->_parents = this->_offspring;
                this->_offspring.resize(Params::pop::size);

                for (indiv_t &indiv : this->_offspring) {
                    indiv = indiv_t(new Phen());
                    indiv->random();
                }

                this->_eval_pop(this->_offspring, 0, this->_offspring.size());
                this->apply_modifier();
                _add(this->_offspring);

                this->_container.get_full_content(this->_pop);
            }

            // Main Iteration of the QD algorithm
            void epoch() {
                // we always add all Q best individuals in TAXONS
                Params::nov::l = -1.;
                Params::nov::use_fixed_l = true;

                this->_parents.resize(Params::pop::size);

                // parents are the same as offspring
                this->_parents = this->_offspring;

                // CLEAR _offspring ONLY after selection, as it can be
                // used by the selector (via this->_offspring)
                this->_offspring.clear();
                this->_offspring.resize(Params::pop::size);

                // Generation of the offspring
                std::vector<size_t> a;
                sferes::misc::rand_ind(a, this->_parents.size());
                assert(this->_parents.size() == Params::pop::size);
                for (size_t i = 0; i < Params::pop::size; i += 2) {
                    boost::shared_ptr <Phen> i1, i2;
                    this->_parents[a[i]]->cross(this->_parents[a[i + 1]], i1, i2);
                    i1->mutate();
                    i2->mutate();
                    i1->develop();
                    i2->develop();
                    this->_offspring[a[i]] = i1;
                    this->_offspring[a[i + 1]] = i2;
                }

                // Evaluation of the offspring
                this->_eval_pop(this->_offspring, 0, this->_offspring.size());
                this->apply_modifier();

                // Addition of the offspring to the container
                assert(Params::taxons::Q * 2 <= this->_offspring.size());

                // Create Novelty Reference Set = {archive + offspring + [ parents if elitism ]}
                sferes::qd::container::GeneralDistanceArchive<Phen, storage_t, Params, distance_t> _novelty_reference_set;
                bool is_added;
                int count_not_added = 0;
                pop_t container_content;
                this->_container.get_full_content(container_content);
                for (indiv_t& indiv: container_content) {
                    is_added = _novelty_reference_set.direct_add(indiv, true);
                    if (not is_added) {
                      count_not_added++;
                    }
                }
                for (indiv_t& indiv: this->_offspring) {
                    is_added = _novelty_reference_set.direct_add(indiv, true);
                    if (not is_added) {
                      count_not_added++;
                    }
                }

                // When using elitism, also consider the parents to compute novelty
                if (aurora::use_elitism_in_taxons()) {
                  for (indiv_t& indiv: this->_parents) {
                    is_added = _novelty_reference_set.direct_add(indiv, true);
                    if (not is_added) {
                      count_not_added++;
                    }
                  }
                }
                // airl::dbg::out(airl::dbg::debug) << count_not_added << " / " << this->_offspring.size() + this->_parents.size() + container_content.size() << '\n';

                std::ostringstream test;
                airl::dbg::out(airl::dbg::debug) << "BEFORE compute novelty - offspring" << '\n';
                test.str("");
                test.clear();
//                for (const indiv_t& indiv: this->_offspring) {
//                    test << indiv->fit().novelty() << ' ';
//                }
                airl::dbg::out(airl::dbg::debug) << test.str() << '\n';

                pop_t population_considered_for_next_generation;
                pop_t population_considered_for_adding_to_container = this->offspring();

                if (not aurora::use_elitism_in_taxons()) {
                  population_considered_for_next_generation = this->_offspring;
                } else {
                  population_considered_for_next_generation = this->_offspring;
                  population_considered_for_next_generation.insert(population_considered_for_next_generation.end(),
                                                              this->_parents.begin(),
                                                              this->_parents.end());
                }

                _novelty_reference_set.compute_novelty_for_specific_population(population_considered_for_next_generation);

                airl::dbg::out(airl::dbg::debug) << "BEFORE sorting - AFTER COMPUTE NOVELTY - _offspring size ->" << this->_offspring.size() << '\n';
                test.str("");
                test.clear();

//                for (const indiv_t& indiv: this->_offspring) {
//                    test << indiv->fit().novelty() << ' ';
//                }
                airl::dbg::out(airl::dbg::debug) << test.str() << '\n';

                airl::dbg::out(airl::dbg::debug) << "BEFORE sorting - AFTER COMPUTE NOVELTY - full pop" << '\n';
                test.str("");
                test.clear();
//                for (const indiv_t& indiv: population_considered_for_next_generation) {
//                    test << indiv->fit().novelty() << ' ';
//                }
                airl::dbg::out(airl::dbg::debug) << test.str() << '\n';

                // airl::dbg::out(airl::dbg::debug) << "AFTER sorting" << '\n';
                // test.str("");
                // test.clear();
                // for (const indiv_t& indiv: population_considered_for_next_generation) {
                    // test << indiv->fit().novelty() << ' ';
                // }
                // airl::dbg::out(airl::dbg::debug) << test.str() << '\n';

                // this->_add(population_considered_for_next_generation); // and add the Q last to the container

                this->m_value_sorter(population_considered_for_next_generation); // sorting the individuals
                this->m_value_sorter(population_considered_for_adding_to_container); // sorting the individuals
                airl::dbg::out(airl::dbg::debug) << "BEFORE UPDATE novelty pop" << '\n';
                test.str("");
                test.clear();
//                for (const indiv_t& indiv: population_considered_for_next_generation) {
//                    test << indiv->fit().novelty() << ' ';
//                }
                airl::dbg::out(airl::dbg::debug) << test.str() << '\n';
                this->_add(population_considered_for_adding_to_container);
                this->_update_offspring(population_considered_for_next_generation); // and replace Q worst with Q best

                airl::dbg::out(airl::dbg::debug) << "AFTER UPDATE offspring" << '\n';
                test.str("");
                test.clear();
//                for (const indiv_t& indiv: population_considered_for_next_generation) {
//                    test << indiv->fit().novelty() << ' ';
//                }
                airl::dbg::out(airl::dbg::debug) << test.str() << '\n';

                assert(this->_offspring.size() == this->_parents.size());

                this->_pop.clear();

                // Copy of the containt of the container into the _pop object.
                this->_container.get_full_content(this->_pop);

                // TODO: uncomment this is the container size is used as a stopping criterion
//                if (this->_pop.size() > Params::taxons::nb_max_policies) {
//                    this->_stop = true;
//                }
            }

        protected:
            ValueSorter m_value_sorter;

            void _add(pop_t & full_pop) {
                // Once the pop has been sorted from lowest to highest
                for (size_t i = 0; i < Params::taxons::Q; ++i) {
                    this->_container.direct_add(full_pop[full_pop.size() - 1 - i]);
                }

                // No need to update the container in TAXONS as it is not used !
//                pop_t empty;
//                this->_container.update(full_pop, empty);
            }

            void _update_offspring(pop_t &population_considered_for_next_generation) {
                // Once the pop has been sorted from lowest to highest

                // Take best indivs in population to fill offspring.
                airl::dbg::out(airl::dbg::debug) << "Size Non-Updated Offspring -> " << this->_offspring.size() << '\n';
                airl::dbg::out(airl::dbg::debug) << "Size Population Considered for Comparison -> " << population_considered_for_next_generation.size() << '\n';

                this->_offspring = std::vector<indiv_t>(population_considered_for_next_generation.end() - this->_offspring.size(),
                                                        population_considered_for_next_generation.end());

                airl::dbg::out(airl::dbg::debug) << "Size Updated Offspring -> " << this->_offspring.size() << '\n';

                // Only replace Q worst with Q best when parents are not considered in the process
                if (not aurora::use_elitism_in_taxons()) {
                  if (Params::taxons::Q * 2 <= this->_offspring.size()) {
                    for (size_t i = 0; i < Params::taxons::Q; ++i) {
                      this->_offspring[i] = boost::make_shared<Phen>(*this->_offspring[this->_offspring.size() - 1 - i]);
                    }
                  }
                }
            }
        };
    }
}

#endif // AURORA_TAXONS_EVOLUTIONARY_ALGORITHM_HPP

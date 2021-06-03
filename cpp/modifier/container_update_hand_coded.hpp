//
// Created by Luca Grillotti on 26/03/2020.
//

#ifndef AURORA_CONTAINER_UPDATE_HAND_CODED_HPP
#define AURORA_CONTAINER_UPDATE_HAND_CODED_HPP

namespace sferes {
    namespace modif {
        template<typename Phen, typename Params>
        class ContainerUpdateHandCoded {
        public:
            typedef Phen phen_t;
            typedef boost::shared_ptr<Phen> indiv_t;
            typedef typename std::vector<indiv_t> pop_t;
            typedef std::vector<std::pair<std::vector<double>, float>> stat_t;

            ContainerUpdateHandCoded() : last_update(0), update_id(0) {}

            using Mat = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

            // defining new matrix for better precision when calculating the new minimum distance l
            using Mat_dist = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

            void copy(const ContainerUpdateHandCoded<Phen, Params>& other_modifier) {
              last_update = other_modifier.last_update;
              update_id = other_modifier.update_id;
            }

          ContainerUpdateHandCoded<Phen, Params>& operator=(const ContainerUpdateHandCoded<Phen, Params>& other_modifier)
          {
            // check for self-assignment
            if (&other_modifier == this) {
              return *this;
            }
            last_update = other_modifier.last_update;
            update_id = other_modifier.update_id;
            return *this;
          }

            template<typename EA>
            void apply(EA &ea) {
                if (!Params::nov::use_fixed_l) {
                    if (Params::update_frequency == -1) { // spacing between updates = max {k * min_update_period, max_update_period } with k progressively increasing
                        const int expected_update = last_update + std::min(Params::min_update_period * (update_id + 1), Params::max_update_period);
                        if (Params::min_update_period > 0 &&
                            (ea.gen() == 1 ||
                             ea.gen() == expected_update)) {
                          update_id++;
                          last_update = ea.gen();
                          update_container(ea);  // clear the archive and re-fill it using the new network
                        }
                    } else if (Params::update_frequency == -2) { // exponential spacing
                        const int expected_update = last_update + Params::update_exponential_coefficient * std::pow(2, update_id - 1);
                        if (ea.gen() == 1 ||
                            ea.gen() == expected_update) {
                          update_id++;
                          last_update = ea.gen();
                          update_container(ea);  // clear the archive and re-fill it using the new network
                        }
                    } else if (ea.gen() > 0) {
                        if ((ea.gen() % Params::update_frequency == 0) || ea.gen() == 1) {
                            update_container(ea);  // clear the archive and re-fill it using the new network
                        }
                    }
                }

                if ((ea.gen() > 0)
                    and (ea.gen() % Params::update_container_period == 0)) {
                    update_container(ea);
                }

                if (!ea.offspring().size()) return;

                assign_descriptor_to_population(ea, ea.offspring());
                // collect_advers(ea);
                // std::cout << "mod dim apply done" << std::endl;
            }

            template<typename EA>
            void assign_descriptor_to_population(EA &ea, pop_t &pop) const {
                // std::cout << "assign_descriptor_to_population" << std::endl;
                pop_t filtered_pop;
                for (auto ind:pop) {
                    if (!ind->fit().dead()) {
                        filtered_pop.push_back(ind);
                    } else {
                        std::vector<double> dd(Params::qd::behav_dim, -1.); // CHANGED from float to double
                        ind->fit().set_desc(dd);
                    }
                }
                if (!Params::nov::use_fixed_l) {

                    pop_t tmp_pop;
                    ea.container().get_full_content(tmp_pop);

                    if ((ea.gen() > 1) && (!tmp_pop.empty()) && (Params::nov::l > 0.)) {
                        this->update_l(tmp_pop);
                        std::cout << "NS - l = " << Params::nov::l << "; size_pop = " << tmp_pop.size() << std::endl;
                    } else if (!tmp_pop.empty()) {
                        this->initialise_l(tmp_pop);
                        std::cout << "NS - l = " << Params::nov::l << "; size_pop = " << tmp_pop.size() << std::endl;
                    }
                }
            }

            void update_l(const pop_t &pop) const {
                constexpr float alpha = 5e-6f;
                Params::nov::l *= (1 - alpha * (static_cast<float>(Params::resolution) - static_cast<float>(pop.size())));
            }

            void initialise_l(const pop_t &pop) const {
                Mat matrix_behavioural_descriptors;
                get_matrix_behavioural_descriptors(pop, matrix_behavioural_descriptors);
                Mat_dist abs_matrix{matrix_behavioural_descriptors.cast<double>()};

                abs_matrix = abs_matrix.rowwise() - abs_matrix.colwise().mean();

                Eigen::SelfAdjointEigenSolver<Mat_dist> eigensolver(abs_matrix.transpose() * abs_matrix);
                if (eigensolver.info() != 0) {
                    abort();
                }
                abs_matrix = (eigensolver.eigenvectors().transpose() * abs_matrix.transpose()).transpose();
                Eigen::PermutationMatrix<Eigen::Dynamic,Eigen::Dynamic> perm(abs_matrix.cols());
                perm.setIdentity();
                std::random_shuffle(perm.indices().data(), perm.indices().data()+perm.indices().size());
                double volume = ((abs_matrix.colwise().maxCoeff() - abs_matrix.colwise().minCoeff()) * perm).prod();


                Params::nov::l = static_cast<float>(0.5 * std::pow(volume / Params::resolution, 1. / matrix_behavioural_descriptors.cols()));
            }

            template<typename EA>
            void update_container(EA &ea) {
                pop_t tmp_pop;
                // Copy of the containt of the container into the _pop object.
                ea.container().get_full_content(tmp_pop);
                ea.container().erase_content();
                std::cout << "size pop: " << tmp_pop.size() << std::endl;

                this->assign_descriptor_to_population(ea, tmp_pop);

                // update l to maintain a number of indiv lower than 10k
                std::cout << "NEW L= " << Params::nov::l << std::endl;

                // Addition of the offspring to the container
                std::vector<bool> added;
                Params::nov::l = Params::nov::l * (1.f - Params::nov::eps);
                ea.add(tmp_pop, added);
                Params::nov::l = Params::nov::l / (1.f - Params::nov::eps);
                ea.pop().clear();
                // Copy of the content of the container into the _pop object.
                ea.container().get_full_content(ea.pop());

                std::cout << "Gen " << ea.gen()
                          << " - size population with l updated : " << ea.pop().size()
                          << std::endl;
            }

            void get_matrix_behavioural_descriptors(const pop_t &pop, Mat &matrix_behavioural_descriptors) const {
                matrix_behavioural_descriptors = Mat(pop.size(), Params::qd::behav_dim);

                for (size_t i = 0; i < pop.size(); i++) {
                    auto desc = pop[i]->fit().desc();
                    for (size_t id = 0; id < Params::qd::behav_dim; id++) {
                        matrix_behavioural_descriptors(i, id) = desc[id];
                    }
                }
            }

          // Serialization
          template<class Archive>
          void serialize(Archive & ar, const unsigned int version)
          {
            ar& BOOST_SERIALIZATION_NVP(this->last_update);
            ar& BOOST_SERIALIZATION_NVP(this->update_id);
          }

        protected:
            int last_update;
            int update_id;
        };
    }
}

#endif //AURORA_CONTAINER_UPDATE_HAND_CODED_HPP

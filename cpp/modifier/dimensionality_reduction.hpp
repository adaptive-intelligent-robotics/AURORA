//
// Created by Luca Grillotti on 24/10/2019.
//

#ifndef SFERES2_DIMENSIONALITY_REDUCTION_HPP
#define SFERES2_DIMENSIONALITY_REDUCTION_HPP

#include <memory>
#include <sferes/stc.hpp>
#include "preprocessor.hpp"

namespace sferes {
    namespace modif {
        template<typename Phen, typename Params, typename NetworkLoader>
        class DimensionalityReduction {
        public:
            typedef Phen phen_t;
            typedef boost::shared_ptr<Phen> indiv_t;
            typedef typename std::vector<indiv_t> pop_t;
            typedef std::vector<std::pair<std::vector<double>, float>> stat_t;

            DimensionalityReduction() : last_update(0), update_id(0), _prep(RescaleFeature<Params>(0.001f)) {
                _prep.init();
                network = std::make_unique<NetworkLoader>();
            }

            using Mat = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

            // defining new matrix for better precision when calculating the new minimum distance l
            using Mat_dist = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

            void copy(const DimensionalityReduction<Phen, Params, NetworkLoader>& other_modifier) {
              // this->network = std::move(other_modifier.network);
              this->network = std::make_unique<NetworkLoader>(*other_modifier.network.get());
              this->last_update = other_modifier.last_update;
              this->update_id = other_modifier.update_id;
              this->_prep = other_modifier._prep;
            }

          DimensionalityReduction<Phen, Params, NetworkLoader>& operator=(const DimensionalityReduction<Phen, Params, NetworkLoader>& other_modifier)
          {
            // check for self-assignment
            if (&other_modifier == this) {
              return *this;
            }
              this->network = std::make_unique<NetworkLoader>(*other_modifier.network.get());
            this->last_update = other_modifier.last_update;
            this->update_id = other_modifier.update_id;
            this->_prep = other_modifier._prep;
            return *this;
          }

            template<typename EA>
            void apply(EA &ea) {
                /*
                 * Basis function used for applying the modifyer to the population
                 * first update descriptors (only every k steps)
                 * assign those values to the population (every step)
                 * */

                if (Params::update_frequency == -1) {  // spacing between updates = max {k * min_update_period, max_update_period } with k progressively increasing
                  const int expected_update = last_update + std::min(Params::min_update_period * (update_id + 1), Params::max_update_period);
                  if (Params::min_update_period > 0 &&
                      (ea.gen() == 1 ||
                       ea.gen() == expected_update)) {
                      update_id++;
                      last_update = ea.gen();
                      update_descriptors(ea);
                  }
                } else if (Params::update_frequency == -2) { // exponential spacing
                  const int expected_update = last_update + Params::update_exponential_coefficient * std::pow(2, update_id - 1);
                  if (ea.gen() == 1 ||
                      ea.gen() == expected_update) {
                      update_id++;
                      last_update = ea.gen();
                      update_descriptors(ea);
                  }
                } else if (ea.gen() > 0) {
                    if ((ea.gen() % Params::update_frequency == 0) || ea.gen() == 1) {
                        update_descriptors(ea);
                    }
                }

                constexpr bool do_perform_update_l_every_iteration = (not aurora::use_volume_adaptive_threshold());

                if (not aurora::is_algo_taxons_based()) {
                  if (do_perform_update_l_every_iteration
                      and (ea.gen() > 0)
                      and (ea.gen() % Params::update_container_period == 0)) {
                    update_container(ea);
                  }
                } else {
                  std::cout << "TAXONS - no update container - gen: " << ea.gen() << " - size pop: " << ea.pop().size() << std::endl;
                }

                if (!ea.offspring().size()) return;

                pop_t entire_pop_container;
                ea.container().get_full_content(entire_pop_container);
                assign_descriptor_to_population(ea, ea.offspring(), entire_pop_container, do_perform_update_l_every_iteration);
            }

            template<typename EA>
            void update_descriptors(EA &ea) {
                Mat data;
                collect_dataset(data, ea, true); // gather the data from the indiv in the archive into a dataset
                train_network(data);
                if (not aurora::is_algo_taxons_based()) {
                  update_container(ea); // clear the archive and re-fill it using the new network
                } else {
                  pop_t tmp_pop;
                  // Copy of the content of the container into the tmp_pop object.
                  ea.container().get_full_content(tmp_pop);
                  std::cout << "TAXONS - update descriptors - gen: " << ea.gen() << " - size pop: " << tmp_pop.size() << std::endl;
                  constexpr bool do_perform_update_l_when_updating_container = false;
                  this->assign_descriptor_to_population(ea, tmp_pop, tmp_pop, do_perform_update_l_when_updating_container);
                }
                ea.pop_advers.clear(); // clearing adversarial examples
            }

            template<typename EA>
            void collect_dataset(Mat &data,
                    EA &ea,
                    const std::vector<typename EA::indiv_t>& content,
                    bool training = false) const {

                size_t pop_size = content.size();
                Mat pop_data, advers_data;
                advers_data.resize(0, 0);

                get_data(content, pop_data);


                if (training && ea.pop_advers.size()) {
                    get_data(ea.pop_advers, advers_data);
                    int rows = pop_data.rows() + advers_data.rows();
                    int cols = pop_data.cols();
                    data.resize(rows, cols);
                    data << pop_data,
                            advers_data;
                } else {
                    int rows = pop_data.rows();
                    int cols = pop_data.cols();
                    data.resize(rows, cols);
                    data << pop_data;
                }

                if (training) {
                    std::cout << "training set is composed of " << data.rows() << " samples  ("
                              << ea.gen() << " archive size : " << pop_size << ")" << std::endl;
                }
            }

            template<typename EA>
            void collect_dataset(Mat &data, EA &ea, bool training = false) const {
                std::vector<typename EA::indiv_t> content;
                if (ea.gen() > 0) {
                    ea.container().get_full_content(content);
                } else {
                    content = ea.offspring();
                }
                collect_dataset(data, ea, content, training);
            }

            void train_network(const Mat &data) {
                // we change the data normalisation each time we train/refine network, could cause small changes in loss between two trainings.
                _prep.init(data);
                Mat scaled_data;
                _prep.apply(data, scaled_data);

                // TODO: Put running mean BACK
//                Mat transformed_data = scaled_data;
//                if ((Params::encoder_type == aurora::EncoderType::conv_seq_ae)
//                    or (Params::encoder_type == aurora::EncoderType::lstm_ae)) {
//                    constexpr int c_len_running_mean = 20;
//                    for (int index_col=c_len_running_mean-1; index_col < transformed_data.cols(); ++index_col) {
//                        transformed_data.col(index_col) = scaled_data.middleCols<c_len_running_mean>(index_col - c_len_running_mean + 1).rowwise().mean();
//                    }
//                    scaled_data = transformed_data;
//                } else {
//                    scaled_data = transformed_data;
//                }

                // Training resets optimiser at the beginning 
                float final_entropy = network->training(scaled_data);
            }

            template<typename EA>
            void assign_descriptor_to_population(EA &ea,
                                            const pop_t &pop,
                                            const RescaleFeature<Params> &prep,
                                            const pop_t& entire_pop_container,
                                            const bool do_perform_update_l) const {
                pop_t filtered_pop;
                for (const indiv_t& ind:pop) {
                    if (!ind->fit().dead()) {
                        filtered_pop.push_back(ind);
                    } else {
                        std::vector<double> dd(Params::qd::behav_dim, -1.); // CHANGED from float to double
                        ind->fit().set_desc(dd);
                    }
                }

                Mat data;
                get_data(filtered_pop, data);
                Mat res; //will be resized
                get_descriptor_autoencoder(data, res, prep, ea, entire_pop_container);

                for (size_t i = 0; i < filtered_pop.size(); i++) {
                    std::vector<double> dd;
                    for (size_t index_latent_space = 0;
                         index_latent_space < Params::qd::behav_dim;
                         ++index_latent_space) {

                        dd.push_back((double) res(i, index_latent_space));

                    }

                    //std::cout << std::endl;
                    //for (size_t i = 0; i< data.rows(); ++i) {
                    //    std::cout << data.row(i) << " - ";
                    //}
                    //std::cout << std::endl;

                    filtered_pop[i]->fit().set_desc(dd);
                    filtered_pop[i]->fit().entropy() = (float) res(i, Params::qd::behav_dim);
                }

                if ((not Params::nov::use_fixed_l)
                    and do_perform_update_l) {
                    // Updating value for l

                    // If we do not use volume adaptive threshold (from original AURORA paper),
                    // Then we use population size adaptive threshold (PSAT)
                    if (not aurora::use_volume_adaptive_threshold()) {
                      if ((ea.gen() > 1) && (!entire_pop_container.empty()) && (Params::nov::l > 0.)) {
                        this->update_l(entire_pop_container);
                      } else if (!entire_pop_container.empty()) {
                        this->initialise_l(entire_pop_container); // TODO: How to initialise l ?
                      }
                    } else { // Using Volume Adaptive Threshold (VAT) -> original method from AURORA paper
                      if (not entire_pop_container.empty()) {
                        Params::nov::l = this->get_new_l(entire_pop_container);
                      }
                    }

                    std::cout << "updated l = " << Params::nov::l << "; size_pop = " << entire_pop_container.size() << std::endl;

                }

            }

            template<typename EA>
            void assign_descriptor_to_population(EA &ea, const pop_t &pop, const pop_t& entire_pop_container, const bool do_perform_update_l) const {
                assign_descriptor_to_population(ea, pop, _prep, entire_pop_container, do_perform_update_l);
            }

            void get_data(const pop_t &pop, Mat &data) const {
//                // std::cout << "get_data" << std::endl;
//                if(pop[0]->fit().dead())
//
//                    std::cout << '\n'; // if no flush, then EIGEN (auto row=data.row(i)) gives an error

                data = Mat(pop.size(), pop[0]->fit().get_flat_obs_size());

                for (size_t i = 0; i < pop.size(); i++) {
                    // std::cout << data.rows() << std::endl;
                    auto row = data.row(i);
                    // std::cout << "here" << std::endl;
                    pop[i]->fit().get_flat_observations(row);
                }
                // std::cout << "get_data done" << std::endl;
            }

            void get_data(const std::vector<std::vector<float>> &v_successions_measures, Mat &data) const {
                data = Mat(v_successions_measures.size(), v_successions_measures[0].size());

                for (size_t index_sequence = 0;
                    index_sequence < v_successions_measures.size();
                    index_sequence++) {
                    const std::vector<float>& succession_measures_in_sequence = v_successions_measures[index_sequence];
                    for (size_t index_measure = 0; index_measure < succession_measures_in_sequence.size(); index_measure++) {
                        data(index_sequence, index_measure) = succession_measures_in_sequence[index_measure];
                    }
                }
            }

            template<typename EA>
            void get_descriptor_autoencoder(const Mat &data, Mat &res,
                                       const RescaleFeature<Params> &prep,
                                       EA& ea,
                                       const pop_t& entire_pop_container) const {
                Mat scaled_data;
                prep.apply(data, scaled_data);
                Mat descriptor, entropy, loss, reconst;
                network->eval(scaled_data, descriptor, entropy, reconst);
                res = Mat(descriptor.rows(), descriptor.cols() + entropy.cols());

                //std::cout << descriptor.rows() << "x" << descriptor.cols() << " " << entropy.rows() << "x" << entropy.cols() << std::endl;

                // (that happens during a call to assign_descriptor_to_population(ea, ea.offspring()))
                // Ideally, it should all the behavioural descriptors from the entire population! (that happens during a call to update_container...)

                Mat scaled_desc;
                RescaleFeature<Params> prep_desc(0.f);

                // init normalisation parameters based on full pop
                if (entire_pop_container.size() > 0) {
                    Mat mat_entire_pop;
                    get_data(entire_pop_container, mat_entire_pop);
                    Mat scaled_mat_entire_pop;
                    prep.apply(mat_entire_pop, scaled_mat_entire_pop);
                    Mat descriptor1, entropy1, loss1, reconst1;
                    network->eval(scaled_mat_entire_pop, descriptor1, entropy1, reconst1);

                    if (aurora::no_normalisation_bd_from_latent_space()) { // No normalisation before setting bd
                      res << descriptor, entropy;
                    } else { // Normalise descriptors before setting them as BDs
                      prep_desc.init(descriptor1);
                      prep_desc.apply(descriptor, scaled_desc);
                      res << scaled_desc, entropy;
                    }
                
                } else {
                    res << descriptor, entropy;
                }
                    // res << descriptor, entropy;
            }


            stat_t get_stat(const pop_t &pop) {
                stat_t result;
                for (auto ind:pop)
                    result.push_back({ind->fit().desc(), ind->fit().value()});
                return result;
            }

            template<typename EA>
            void update_container(EA &ea, bool do_perform_update_l_when_updating_container=true) {
                pop_t tmp_pop;
                // Copy of the containt of the container into the _pop object.
                ea.container().get_full_content(tmp_pop);
                ea.container().erase_content();
                std::cout << "size pop: " << tmp_pop.size() << std::endl;

                this->assign_descriptor_to_population(ea, tmp_pop, tmp_pop, do_perform_update_l_when_updating_container);

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
                // dump_data(ea,stat1,stat2,added);

                std::cout << "Gen " << ea.gen() << " - size population with l updated : " << ea.pop().size() << std::endl;
            }

            void update_l(const pop_t &pop) const {
                Params::nov::l *= (1 - Params::nov::coefficient_proportional_control_l * (static_cast<float>(Params::resolution) - static_cast<float>(pop.size())));
            }

            void
            distance(const Mat& X, Mat& dist) const
            {
              // std::cout<<"Neg distance"<<std::endl;
              // Compute norms
              Mat XX = X.array().square().rowwise().sum();
              Mat XY = (2 * X) * X.transpose();

              // Compute final expression
              dist = XX * Eigen::MatrixXf::Ones(1, XX.rows());
              dist = dist + Eigen::MatrixXf::Ones(XX.rows(), 1) * (XX.transpose());
              dist = dist - XY;
              // std::cout<<"END Neg distance"<<std::endl;
            }

            void
            distance(const Mat& X, Mat_dist& dist) const
            {
              // Compute norms
              Mat_dist X_double = X.cast<double>();
              Mat_dist XX = X_double.array().square().rowwise().sum();
              Mat_dist XY = (2 * X_double) * X_double.transpose();

              // Compute final expression
              dist = XX * Eigen::MatrixXd::Ones(1, XX.rows());
              dist = dist + Eigen::MatrixXd::Ones(XX.rows(), 1) * (XX.transpose());
              dist = dist - XY;
            }

            float
            get_new_l(const pop_t& pop) const
            {
              Mat data;
              this->get_matrix_behavioural_descriptors(pop, data);
              Mat dist;
              this->distance(data, dist);
              float maxdist = sqrt(dist.maxCoeff());
              float K = Params::vat::resolution_multiplicative_constant * Params::resolution; // value to have a specific "resolution" (Params::resolution)
              return static_cast<float>(maxdist / std::pow(K, 1. / Params::qd::behav_dim));
            }

            /**
             * Initialise value of distance threshold by estimating the volume of the convex hull around
             * the individuals in the population pop
             *
             * It uses a PCA to find the most relevant axes,
             * By projecting the BDs on these axes, we hope to get an accurate estimation of the volume.
             */
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
                double volume = (abs_matrix.colwise().maxCoeff() - abs_matrix.colwise().minCoeff()).prod();

                Params::nov::l = static_cast<float>(0.5 * std::pow(volume / Params::resolution, 1. / matrix_behavioural_descriptors.cols()));
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

            void get_reconstruction(const Mat &data, Mat &res) const {
                Mat scaled_data, scaled_res;
                _prep.apply(data, scaled_data);
                network->get_reconstruction(scaled_data, scaled_res);
                const bool do_clip_between_zero_one = aurora::does_encode_images();
                _prep.deapply(scaled_res, res, do_clip_between_zero_one);
            }

            NetworkLoader *get_network_loader() const {
                return &*network;
            }

            RescaleFeature<Params>& prep() {
                return _prep;
            }

            void
            save_model(const std::string& name_file) const
            {
              this->get_network_loader()->save_model(name_file);
            }

            // Serialization
            template<class Archive>
            void
            serialize(Archive& ar, const unsigned int version)
            {
              ar& BOOST_SERIALIZATION_NVP((*(network.get())));
              ar& BOOST_SERIALIZATION_NVP(this->last_update);
              ar& BOOST_SERIALIZATION_NVP(this->update_id);
              ar& BOOST_SERIALIZATION_NVP(this->_prep);
            }

          protected:
            std::unique_ptr<NetworkLoader> network;
            int last_update;
            int update_id;
            RescaleFeature<Params> _prep;
        };
    }
}

#endif //SFERES2_DIMENSIONALITY_REDUCTION_HPP

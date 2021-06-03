//
// Created by Luca Grillotti on 16/04/2020.
//

#ifndef AURORA_ANALYSIS_HPP
#define AURORA_ANALYSIS_HPP

#include <Eigen/Core>

#include "torch/script.h"

#include "modifier/autoencoder/autoencoder.hpp"
#include "modifier/network_loader_pytorch.hpp"
#include "modifier/dimensionality_reduction.hpp"
#include "modifier/container_update_hand_coded.hpp"
#include "stat/stat_custom_metric.hpp"

#include <boost/array.hpp>
#include <boost/foreach.hpp>
#include <boost/fusion/algorithm/iteration/for_each.hpp>
#include <boost/fusion/include/for_each.hpp>
#include <boost/multi_array.hpp>
#include <boost/timer/timer.hpp>

#include "interactive_map.hpp"

namespace aurora {
    namespace analysis {
        template<typename ea_t, typename fit_t, typename Phen, typename Params>
        void project_to_latent_space(ea_t &ea,
                                     const std::string &path_ns_archive,
                                     const std::string &path_aurora_archive,
                                     const std::string &path_network,
                                     const std::string &path_save,
                                     const size_t behav_dim_ns) {
            typedef Phen phen_t;
            typedef boost::shared_ptr<phen_t> indiv_t;
            typedef typename std::vector<indiv_t> pop_t;

            using Mat = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

            auto &fit_modifier = ea.template fit_modifier<0>();

            torch::load(fit_modifier.get_network_loader()->auto_encoder().ptr(), path_network);

            Interactive_map<Params> imap;

            archive_content_t archive_ns_individuals;
            archive_content_t archive_aurora_individuals;

            imap.load_archive(path_ns_archive, archive_ns_individuals, behav_dim_ns);
            const controllers_archive_t &controllers_to_show_ns = std::get<3>(archive_ns_individuals);

            imap.load_archive(path_aurora_archive, archive_aurora_individuals, Params::qd::behav_dim);
            const controllers_archive_t &controllers_to_show_aurora = std::get<3>(archive_aurora_individuals);

            sferes::parallel::init();

            std::vector<indiv_t> offspring_ns;
            std::vector<indiv_t> offspring_aurora;

            offspring_ns.resize(controllers_to_show_ns.size());
            offspring_aurora.resize(controllers_to_show_aurora.size());
            for (indiv_t &indiv : offspring_ns) {
                indiv = indiv_t(new Phen());
            }
            for (indiv_t &indiv : offspring_aurora) {
                indiv = indiv_t(new Phen());
            }
            for (size_t i = 0; i < offspring_ns.size(); ++i) {
                for (size_t index_ctrl = 0; index_ctrl < 36; ++index_ctrl) {
                    offspring_ns[i]->gen().data(index_ctrl, controllers_to_show_ns[i][index_ctrl]);
                }
            }
            for (size_t i = 0; i < offspring_aurora.size(); ++i) {
                for (size_t index_ctrl = 0; index_ctrl < 36; ++index_ctrl) {
                    offspring_aurora[i]->gen().data(index_ctrl, controllers_to_show_aurora[i][index_ctrl]);
                }
            }
            for (indiv_t &indiv : offspring_ns) {
                indiv->develop();
            }
            for (indiv_t &indiv : offspring_aurora) {
                indiv->develop();
            }
            fit_t fit = fit_t();
            std::cout << "Evaluate individuals from NS archive" << std::endl;
            ea.eval().eval(offspring_ns, 0, offspring_ns.size(), fit);
            fit = fit_t();
            std::cout << "Evaluate individuals from Aurora archive" << std::endl;
            ea.eval().eval(offspring_aurora, 0, offspring_aurora.size(), fit);
            std::cout << "Prep" << std::endl;

            Mat data_aurora;
            RescaleFeature<Params> prep(0.001f);
            std::cout << "Collect Dataset" << std::endl;
            ea.pop().clear();
            ea.pop() = offspring_aurora;
            std::cout << "Collect Dataset" << std::endl;
            fit_modifier.collect_dataset(data_aurora, ea, ea.pop());
            std::cout << "Prep Init" << std::endl;
            prep.init(data_aurora);
            std::cout << "Assign Descriptor to pop" << std::endl;
            fit_modifier.assign_descriptor_to_population(ea, offspring_ns, prep);
            std::cout << "STAT" << std::endl;

            sferes::stat::Projection<phen_t, Params> stat_proj;
            ea.pop().clear();
            ea.pop() = offspring_ns;
            ea.set_gen(1);
            ea.set_res_dir("/");
            stat_proj.refresh_instantly(ea, "projected_ns_");
        }

        template<typename T>
        T get_squared_dist(std::vector<T> v_1, std::vector<T> v_2) {
            assert(v_1.size() == v_2.size());
            T d = 0.0;
            T x;
            for (size_t i = 0; i < v_1.size(); ++i) {
                x = v_1[i] - v_2[i];
                d += x * x;
            }
            return d;
        }

        template<typename T>
        T get_squared_dist_modulo(std::vector<T> v_1, std::vector<T> v_2, T period) {
            assert(v_1.size() == v_2.size());
            T d = 0.0;
            T x;
            for (size_t i = 0; i < v_1.size(); ++i) {
                x = (v_1[i] - v_2[i]) % period;
                d += std::min(x * x, (period - x) * (period - x));
            }
            return d;
        }

        template<typename T>
        T get_squared_dist_gt_walls(std::vector<T> v_1, std::vector<T> v_2) {
            assert(v_1.size() == v_2.size());
            assert(v_1.size() == 6);

            std::vector<T> v_1_pos(v_1.begin(), v_1.begin() + 3);
            std::vector<T> v_2_pos(v_2.begin(), v_2.begin() + 3);

            std::vector<T> v_1_rot(v_1.begin() + 3, v_1.end());
            std::vector<T> v_2_rot(v_2.begin() + 3, v_2.end());

            return get_squared_dist(v_1_pos, v_2_pos) + get_squared_dist_modulo(v_1_rot, v_2_rot, static_cast<T>(2 * M_PI));
        }

        template<typename ea_t, typename fit_t, typename Phen, typename Params>
        void save_images_nearest_neighbours(ea_t &ea,
                                            const std::string &path_archive,
                                            const std::string &path_proj,
                                            const std::string &prefix_save,
                                            const size_t behav_dim,
                                            const size_t index_chosen_indiv,
                                            const size_t number_nearest_neighbours) {
            typedef Phen phen_t;
            typedef boost::shared_ptr<phen_t> indiv_t;
            typedef typename std::vector<indiv_t> pop_t;

            Interactive_map<Params> imap;

            archive_content_t archive_individuals;
            stat_projection_t stat_projection_individuals;

            imap.load_archive(path_archive, archive_individuals, behav_dim);
            imap.load_stat_projection(path_proj, stat_projection_individuals, behav_dim);

            const controllers_archive_t &controllers_archive = std::get<3>(archive_individuals);
            const gt_archive_t &gt_archive = std::get<1>(stat_projection_individuals);
            std::vector<size_t> indexes = std::get<0>(archive_individuals);

            sferes::parallel::init();

            std::vector<indiv_t> offspring;

            // creating offspring
            offspring.reserve(controllers_archive.size());
            for (size_t i = 0; i < controllers_archive.size(); ++i) {
                offspring.push_back(indiv_t(new Phen()));
            }

            for (size_t i = 0; i < offspring.size(); ++i) {
                for (size_t index_ctrl = 0; index_ctrl < 36; ++index_ctrl) {
                    offspring[i]->gen().data(index_ctrl, controllers_archive[i][index_ctrl]);
                }
            }

            for (indiv_t &indiv : offspring) {
                indiv->develop();
            }
            // Calculating distances
            std::vector<float> distances;
            distances.reserve(controllers_archive.size());
            for (size_t index_indiv = 0;
                 index_indiv < controllers_archive.size();
                 ++index_indiv) {

                distances.push_back(
                        get_squared_dist(gt_archive[index_chosen_indiv], gt_archive[index_indiv])
                );

            }

            // Get indexes n+1 th first neighbours (including itself)
            std::nth_element(indexes.begin(),
                             indexes.begin() + number_nearest_neighbours,
                             indexes.end(),
                             [distances](const size_t &index_1, const size_t &index_2) {
                                 return (distances[index_1] < distances[index_2]);
                             });

            std::vector<size_t> chosen_indexes(indexes.begin(), indexes.begin() + number_nearest_neighbours + 1);
            std::vector<indiv_t> chosen_individuals;
            chosen_individuals.reserve(number_nearest_neighbours + 1);


            for (size_t index : chosen_indexes) {
                chosen_individuals.push_back(offspring[index]);
            }

            fit_t fit = fit_t();
            std::cout << "Evaluate individuals from archive" << std::endl;
            std::cout << chosen_individuals.size() << std::endl;
            ea.eval().eval(chosen_individuals, 0, chosen_individuals.size(), fit);
            fit = fit_t();

            ea.pop().clear();
            ea.pop() = offspring;

            std::cout << "STAT" << std::endl;
            sferes::stat::ImagesObservations<phen_t, Params> stat_images_observation;
            ea.set_gen(1);
            ea.set_res_dir("/");
            stat_images_observation.save_images_specific_observations(prefix_save, ea, chosen_indexes);
            for (size_t index : chosen_indexes) {
                std::cout << "Index: " << index
                          << " - Distance: " << distances[index]
                          << " - Size Pixels: " << ea.pop()[index_chosen_indiv]->fit().observations().size()
                          << " - Distance Pixels: " << get_squared_dist(ea.pop()[index_chosen_indiv]->fit().observations(), ea.pop()[index]->fit().observations()) << std::endl;
            }
        }

        template<typename ea_t, typename fit_t, typename Phen, typename Params>
        void save_all_pixel_distances(ea_t &ea,
                                      const std::string &path_archive,
                                      const std::string &path_proj,
                                      const std::string &prefix_save,
                                      const size_t behav_dim,
                                      const size_t number_nearest_neighbours) {
            typedef Phen phen_t;
            typedef boost::shared_ptr<phen_t> indiv_t;
            typedef typename std::vector<indiv_t> pop_t;

            Interactive_map<Params> imap;

            archive_content_t archive_individuals;
            stat_projection_t stat_projection_individuals;

            imap.load_archive(path_archive, archive_individuals, behav_dim);
            imap.load_stat_projection(path_proj, stat_projection_individuals, behav_dim);

            const controllers_archive_t &controllers_archive = std::get<3>(archive_individuals);
            const gt_archive_t &gt_archive = std::get<1>(stat_projection_individuals);
            std::vector<size_t> indexes = std::get<0>(archive_individuals);

            sferes::parallel::init();

            std::vector<indiv_t> offspring;

            // creating offspring
            offspring.reserve(controllers_archive.size());
            for (size_t i = 0; i < controllers_archive.size(); ++i) {
                offspring.push_back(indiv_t(new Phen()));
            }

            for (size_t i = 0; i < offspring.size(); ++i) {
                for (size_t index_ctrl = 0; index_ctrl < 36; ++index_ctrl) {
                    offspring[i]->gen().data(index_ctrl, controllers_archive[i][index_ctrl]);
                }
            }

            for (indiv_t &indiv : offspring) {
                indiv->develop();
            }

            // Creating EA pop and evaluate it
            ea.pop().clear();
            ea.pop() = offspring;

            fit_t fit = fit_t();
            std::cout << "Evaluate individuals from archive" << std::endl;
            ea.eval().eval(ea.pop(), 0, ea.pop().size(), fit);

            // Calculating distances
            std::vector<float> distances;
            distances.reserve(controllers_archive.size());

            Eigen::ArrayXXf array_distances_f = Eigen::ArrayXXf::Zero(controllers_archive.size(), controllers_archive.size());

            for (size_t index_indiv_1 = 0;
                 index_indiv_1 < controllers_archive.size();
                 ++index_indiv_1) {
                if (index_indiv_1 % 100 == 0) {
                    std::cout << "Calculating GT distances - " << index_indiv_1 << " / " << controllers_archive.size() << std::endl;
                }
                for (size_t index_indiv_2 = 1;
                     index_indiv_2 < index_indiv_1;
                     ++index_indiv_2) {
                    float distance_two_indiv = get_squared_dist(gt_archive[index_indiv_1], gt_archive[index_indiv_2]);
                    array_distances_f(index_indiv_1, index_indiv_2) = distance_two_indiv;
                    array_distances_f(index_indiv_2, index_indiv_1) = distance_two_indiv;
                }
            }

            // creating array indexes
            Eigen::Array<int, 1, Eigen::Dynamic, Eigen::RowMajor> v_indexes =
                    Eigen::Array<int, 1, Eigen::Dynamic, Eigen::RowMajor>::LinSpaced(array_distances_f.cols(), 0, array_distances_f.cols());
            Eigen::Array<int, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> array_indexes(array_distances_f.rows(), array_distances_f.cols());
            for (Eigen::Index index_row = 0; index_row < array_indexes.rows(); ++index_row) {
                array_indexes.row(index_row) = v_indexes;
            }

            // Reshaping array indexes: at every row, keep at first the indexes of the closest elements
            for (Eigen::Index index_row = 0; index_row < array_indexes.rows(); ++index_row) {
                if (index_row % 10 == 0) {
                    std::cout << "Sorting distances - " << index_row << " / " << controllers_archive.size() << std::endl;
                }

                std::nth_element(array_indexes.row(index_row).data(),
                                 array_indexes.row(index_row).data() + number_nearest_neighbours,
                                 array_indexes.row(index_row).data() + array_indexes.row(index_row).size(),
                                 [&array_distances_f, index_row](const size_t &index_1, const size_t &index_2) {
                                     return (array_distances_f(index_row, index_1) < array_distances_f(index_row, index_2));
                                 });
            }

            // Creating array for pixel-distances
            Eigen::Array<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> array_pixel_distances_f(array_indexes.rows(), number_nearest_neighbours);
            for (Eigen::Index index_row = 0; index_row < array_pixel_distances_f.rows(); ++index_row) {

                if (index_row % 10 == 0) {
                    std::cout << "Calculating Pixel Distances - " << index_row << " / " << controllers_archive.size() << std::endl;
                }

                for (Eigen::Index index_col = 0; index_col < array_pixel_distances_f.cols(); ++index_col) {
                    array_pixel_distances_f(index_row, index_col) =
                            get_squared_dist(ea.pop()[index_row]->fit().observations(),
                                     ea.pop()[array_indexes(index_row, index_col)]->fit().observations());
                }
            }

            // Taking the mean pixel-wise distance of the closest elements
            Eigen::ArrayXf mean_distances_f = array_pixel_distances_f.rowwise().mean();
            std::vector<float> vector_mean_distances(mean_distances_f.data(), mean_distances_f.data() + mean_distances_f.size());

            std::cout << "STAT" << std::endl;
            sferes::stat::CustomMetric<phen_t, Params> stat_custom_metric;
            ea.set_gen(1);
            ea.set_res_dir("/");
            stat_custom_metric.write_container(prefix_save, ea, vector_mean_distances);

        }
    }
}

#endif //AURORA_ANALYSIS_HPP

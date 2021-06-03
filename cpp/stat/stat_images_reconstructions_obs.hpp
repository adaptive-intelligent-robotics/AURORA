//
// Created by Luca Grillotti on 30/10/2019.
//

#ifndef SFERES2_STAT_IMAGES_RECONSTRUCTION_OBS_HPP
#define SFERES2_STAT_IMAGES_RECONSTRUCTION_OBS_HPP

#include <sferes/stat/stat.hpp>

#include "robot_dart/gui/helper.hpp"
#include "stat/utils.hpp"

namespace sferes {
    namespace stat {

        SFERES_STAT(ImagesReconstructionObs, Stat)
        {
        public:

            typedef Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> matrix_t;

            template<typename EA>
            void _write_container_color(const std::string &prefix, const EA &ea) const {
                std::cout << "writing..." << prefix << std::endl;
                std::string prefix_image_indiv;

                size_t index_indiv{0};

                matrix_t observations_population;
                matrix_t reconstruction_obs_population;

                boost::fusion::at_c<0>(ea.fit_modifier()).get_data(ea.pop(), observations_population);
                boost::fusion::at_c<0>(ea.fit_modifier()).get_reconstruction(observations_population, reconstruction_obs_population);

                size_t size_observation = Params::image_width * Params::image_height;


                matrix_t reconstruction_red{reconstruction_obs_population.leftCols(size_observation)};
                matrix_t reconstruction_green{
                    reconstruction_obs_population.block(0, size_observation,
                        reconstruction_obs_population.rows(), size_observation)};
                matrix_t reconstruction_blue{reconstruction_obs_population.rightCols(size_observation)};


                for (auto it = ea.pop().begin(); it != ea.pop().end(); ++it) {
                    if (index_indiv % Params::stat::period_saving_individual_in_population == 0) {
                        prefix_image_indiv = prefix
                                             + "_indiv_"
                                             + add_leading_zeros(index_indiv);

                        robot_dart::gui::ImageSerialisable rgb_image;
                        rgb_image.width = Params::image_width;
                        rgb_image.height = Params::image_height;

                        size_t offset = size_observation * index_indiv;

                        for (size_t index_start_row_obs = offset;
                             index_start_row_obs < offset + size_observation;
                             index_start_row_obs += Params::image_width) {

                            std::vector<float> row_observation_float_red(reconstruction_red.data() + index_start_row_obs,
                                                                         reconstruction_red.data() + index_start_row_obs + Params::image_width);
                            std::vector<float> row_observation_float_green(reconstruction_green.data() + index_start_row_obs,
                                                                         reconstruction_green.data() + index_start_row_obs + Params::image_width);
                            std::vector<float> row_observation_float_blue(reconstruction_blue.data() + index_start_row_obs,
                                                                         reconstruction_blue.data() + index_start_row_obs + Params::image_width);

                            std::vector<uint8_t> row_observation_red;
                            std::vector<uint8_t> row_observation_green;
                            std::vector<uint8_t> row_observation_blue;

                            for (auto item_row: row_observation_float_red) {
                                row_observation_red.push_back(floor(255. * item_row));
                            }
                            for (auto item_row: row_observation_float_green) {
                                row_observation_green.push_back(floor(255. * item_row));
                            }
                            for (auto item_row: row_observation_float_blue) {
                                row_observation_blue.push_back(floor(255. * item_row));
                            }

                            rgb_image.data.reserve(3 * row_observation_blue.size());

                            for (size_t index = 0;
                                 index < row_observation_float_red.size();
                                 ++index) {
                                rgb_image.data.push_back(row_observation_red[index]); 
                                rgb_image.data.push_back(row_observation_green[index]); 
                                rgb_image.data.push_back(row_observation_blue[index]); 
                            }

                        }
                        robot_dart::gui::save_png_image(
                                prefix_image_indiv + "_rgb.png",
                                rgb_image);
                    }
                    ++index_indiv;
                }
            }

            template<typename EA>
            void _write_container(const std::string &prefix, const EA &ea) const {
                std::cout << "writing..." << prefix << std::endl;
                std::string prefix_image_indiv;

                size_t index_indiv{0};
                
                matrix_t observations_population;
                matrix_t reconstruction_obs_population;
                boost::fusion::at_c<0>(ea.fit_modifier()).get_data(ea.pop(), observations_population);
                boost::fusion::at_c<0>(ea.fit_modifier()).get_reconstruction(observations_population, reconstruction_obs_population);

                for (auto it = ea.pop().begin(); it != ea.pop().end(); ++it) {
                    if (index_indiv % Params::stat::period_saving_individual_in_population == 0) {
                        prefix_image_indiv = prefix 
                          + "_indiv_" 
                          + add_leading_zeros(index_indiv);

                        robot_dart::gui::GrayscaleImage grayscale_image;
                        grayscale_image.width = Params::image_width;
                        grayscale_image.height = Params::image_height;

                        size_t size_observation = Params::image_width * Params::image_height;
                        size_t offset = size_observation * index_indiv;

                        for (size_t index_start_row_obs = offset;
                            index_start_row_obs < offset + size_observation;
                            index_start_row_obs += Params::image_width) {
                            std::vector<float> row_observation_float(reconstruction_obs_population.data() + index_start_row_obs,
                                    reconstruction_obs_population.data() + index_start_row_obs + Params::image_width);
                            grayscale_image.data.reserve(row_observation_float.size());
                            for (auto item_row: row_observation_float) {
                                grayscale_image.data.push_back(floor(255. * item_row));
                            }
                        }
                        robot_dart::gui::save_png_image(
                            prefix_image_indiv + "_grayscale.png",
                            grayscale_image);
                    }
                    ++index_indiv;
                }
            }

            template<typename EA>
            void _write_container_measures(const std::string &filename, const EA &ea) const {
                std::cout << "writing..." << filename << std::endl;

                size_t index_indiv{0};

                matrix_t observations_population;
                matrix_t reconstruction_obs_population;
                boost::fusion::at_c<0>(ea.fit_modifier()).get_data(ea.pop(), observations_population);
                boost::fusion::at_c<0>(ea.fit_modifier()).get_reconstruction(observations_population, reconstruction_obs_population);

                std::ofstream ofs(filename.c_str());

                size_t offset = 0;
                ofs.precision(8);

                for (auto it = ea.pop().begin(); it != ea.pop().end(); ++it) {

                    if (index_indiv % Params::stat::period_saving_individual_in_population == 0) {
                        ofs << index_indiv << "    ";

                        for (long index_reconstruction = 0;
                             index_reconstruction < reconstruction_obs_population.row(index_indiv).size();
                             ++index_reconstruction) {

                            ofs << reconstruction_obs_population.row(index_indiv)(index_reconstruction) << " ";
                        }

                        ofs << std::endl;
                    }
                    ++index_indiv;
                }
            }

            template<typename EA>
            void _write_container_measures(
                    const std::string &filename,
                    const EA &ea,
                    const matrix_t reconstruction_obs_population) const {
                /*
                 * Variant of the original function that does not look at the population,
                 * Insead, it directly writes the successive measures into a file
                 */

                std::cout << "writing..." << filename << std::endl;

                std::ofstream ofs(filename.c_str());

                size_t offset = 0;
                ofs.precision(8);

                for (long index_indiv = 0; index_indiv < reconstruction_obs_population.rows(); ++index_indiv) {

                    if (index_indiv % Params::stat::period_saving_individual_in_population == 0) {
                        ofs << index_indiv << "    ";

                        for (long index_reconstruction = 0;
                             index_reconstruction < reconstruction_obs_population.row(index_indiv).size();
                             ++index_reconstruction) {

                            ofs << reconstruction_obs_population.row(index_indiv)(index_reconstruction) << " ";
                        }

                        ofs << std::endl;
                    }
                }
            }


            template<typename EA>
            void refresh(EA &ea) {
                if ((ea.gen() % Params::stat::save_images_period == 0) || (ea.gen() == 1) ) {
                    if (aurora::does_encode_images()) {
                        std::string prefix = ea.res_dir() + "/"
                                             + "reconstruction_obs_gen_"
                                             + add_leading_zeros(ea.gen());
                        if (Params::use_colors) {
                            _write_container_color(prefix, ea);
                        } else {
                            _write_container(prefix, ea);
                        }
                    } else if (aurora::does_encode_sequence()) {
                        std::string filename = ea.res_dir() + "/"
                                           + "reconstruction_obs_gen_"
                                           + add_leading_zeros(ea.gen())
                                           + std::string(".dat");
                        _write_container_measures(filename, ea);
                    }
                }

            }
        };

    }
}


#endif

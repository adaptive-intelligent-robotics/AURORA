//
// Created by Luca Grillotti on 30/10/2019.
//

#ifndef SFERES2_STAT_IMAGES_OBSERVATIONS_HPP
#define SFERES2_STAT_IMAGES_OBSERVATIONS_HPP

#include <sferes/stat/stat.hpp>

#include <robox2d/gui/helper.hpp>

#include "stat/utils.hpp"
#include "environments/image_utils.hpp"

namespace sferes {
    namespace stat {

    SFERES_STAT(ImagesObservations, Stat){

      public:
            void save_image(const std::string& prefix_image_indiv, const robot_dart::gui::ImageSerialisable& rgb_image) const
            {
              if (Params::use_colors) {
                robot_dart::gui::save_png_image(prefix_image_indiv + "_color.png", rgb_image);
              } else {
                robot_dart::gui::save_png_image(prefix_image_indiv + "_grayscale.png",
                                                robot_dart::gui::convert_rgb_to_grayscale(rgb_image));
              }
            }

            void save_image(const std::string& prefix_image_indiv, const robox2d::gui::ImageSerialisable& rgb_image) const
            {
              if (Params::use_colors) {
                robox2d::gui::save_png_image(prefix_image_indiv + "_color.png", rgb_image);
              } else {
                robox2d::gui::save_png_image(prefix_image_indiv + "_grayscale.png",
                                             robox2d::gui::convert_rgb_to_grayscale(rgb_image));
              }
            }

            void save_specific_index(const boost::shared_ptr<Phen>& indiv, const std::string &prefix_image_indiv) const {
                this->save_image(prefix_image_indiv, indiv->fit().get_rgb_image());
            }

            template<typename EA>
            void save_images_specific_observations(const std::string &prefix, const EA &ea,
                                                   const std::vector<size_t>& indexes_to_save) const {
                std::cout << "writing..." << prefix << std::endl;
                std::string prefix_image_indiv;

                for (size_t index_to_save : indexes_to_save) {
                    prefix_image_indiv = prefix
                                         + "_indiv_"
                                         + add_leading_zeros(index_to_save);

                    save_specific_index(ea.pop()[index_to_save], prefix_image_indiv);
                }
            }

            template<typename EA>
            void _save_images_observations(const std::string &prefix, const EA &ea) const {
                std::cout << "writing..." << prefix << std::endl;
                std::string prefix_image_indiv;

                size_t index_indiv{0};
                for (auto it = ea.pop().begin(); it != ea.pop().end(); ++it) {
                    if (index_indiv % Params::stat::period_saving_individual_in_population == 0) {
                        prefix_image_indiv = prefix
                                             + "_indiv_"
                                             + add_leading_zeros(index_indiv);

                        save_specific_index(*it, prefix_image_indiv);
                    }
                    ++index_indiv;
                }
            }

            template<typename EA>
            void _save_images_observations_measures(const std::string &filename, const EA &ea) const {
                std::cout << "writing..." << filename << std::endl;
                std::string prefix_image_indiv;

                std::ofstream ofs(filename.c_str());

                ofs.precision(8);
                size_t index_indiv{0};

                for (auto it = ea.pop().begin(); it != ea.pop().end(); ++it) {
                    if (index_indiv % Params::stat::period_saving_individual_in_population == 0) {
                        ofs << index_indiv << "    ";

                        for (long index_reconstruction = 0;
                             index_reconstruction < (*it)->fit().observations().size();
                             ++index_reconstruction) {
                            ofs << (*it)->fit().observations()[index_reconstruction] << " ";
                        }

                        ofs << std::endl;
                    }
                    ++index_indiv;
                }
            }

            template<typename EA>
            void refresh(EA &ea) {
                if ((ea.gen() % Params::stat::save_images_period == 0) || (ea.gen() == 1)) {
                    if (aurora::does_encode_images()) {
                        std::string prefix = ea.res_dir() + "/"
                                             + "observation_gen_"
                                             + add_leading_zeros(ea.gen());
                        _save_images_observations(prefix, ea);
                    } else if (aurora::does_encode_sequence() or aurora::is_algo_hand_coded()) {
                        std::string filename = ea.res_dir() + "/"
                                                 + "observation_gen_"
                                                 + add_leading_zeros(ea.gen())
                                                 + std::string(".dat");
                        _save_images_observations_measures(filename, ea);
                    }
                }
            }
        };

    }
}


#endif

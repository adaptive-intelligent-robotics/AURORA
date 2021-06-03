//
// Created by Luca Grillotti on 04/06/2020.
//

#ifndef AURORA_STAT_SUCCESSIVE_GT_HPP
#define AURORA_STAT_SUCCESSIVE_GT_HPP

#include <sferes/stat/stat.hpp>

#include "stat/utils.hpp"

namespace sferes {
    namespace stat {

        SFERES_STAT(SuccessiveGT, Stat)
        {
        public:
            template<typename EA>
            void _save_successive_gt(const std::string &filename, const EA &ea) const {
                std::cout << "writing..." << filename << std::endl;
                std::string prefix_image_indiv;

                std::ofstream ofs(filename.c_str());

                ofs.precision(5);
                size_t index_indiv{0};

                for (auto it = ea.pop().begin(); it != ea.pop().end(); ++it) {

                    if (index_indiv % Params::stat::period_saving_individual_in_population == 0) {
                        ofs << index_indiv << "    ";

                        const size_t c_size_gt = (*it)->fit().gt().size();

                        ofs << c_size_gt << "    ";

                        for (long index_reconstruction = 0;
                             index_reconstruction < (*it)->fit().successive_gt().size();
                             ++index_reconstruction) {
                            ofs << (*it)->fit().successive_gt()[index_reconstruction] << " ";
                        }

                        ofs << std::endl;
                    }
                    ++index_indiv;
                }
            }

            template<typename EA>
            void refresh(EA &ea) {
                if ((ea.gen() % Params::stat::save_images_period == 0) or (ea.gen() == 1)) {
                    /*
                     * Saving Successive GT only makes sense with LSTMs
                     * TODO: if aurora::EncoderType == video_ae ?
                     */
                    if (Params::encoder_type == aurora::EncoderType::lstm_ae) {
                        std::string filename = ea.res_dir() + "/"
                                               + "successive_gt_"
                                               + add_leading_zeros(ea.gen())
                                               + std::string(".dat");
                        _save_successive_gt(filename, ea);
                    }
                }
            }
        };

    }
}

#endif //AURORA_STAT_SUCCESSIVE_GT_HPP

//
// Created by Luca Grillotti on 13/01/2020.
//

#ifndef AURORA_STAT_MODEL_AUTOENCODER_HPP
#define AURORA_STAT_MODEL_AUTOENCODER_HPP

#include <sferes/stat/stat.hpp>

namespace sferes {
    namespace stat {

        SFERES_STAT(ModelAutoencoder, Stat)
        {
        public:
            template<typename EA>
            void refresh(EA &ea) {
                if (ea.gen() % Params::pop::dump_period == 0) {
                    std::string name_file = ea.res_dir() + "/"
                                            + "model_gen_"
                                            + boost::lexical_cast<std::string>(ea.gen())
                                            + ".pt";
                    std::cout << "writing... " << name_file << std::endl;
                    boost::fusion::at_c<0>(ea.fit_modifier()).save_model(name_file);
                }
            }

            template<typename EA>
            void load(EA &ea) {
              std::string path_saved_module = ea.res_dir() + "/"
                                      + "model_autoencoder_gen_"
                                      + boost::lexical_cast<std::string>(ea.gen())
                                      + ".pt";
              torch::load(m_module_ptr, path_saved_module);
            }

            const std::shared_ptr<torch::nn::Module>& module_ptr() {
              return m_module_ptr;
            }

        protected:
            std::shared_ptr<torch::nn::Module> m_module_ptr{nullptr};
        };

    }
}

#endif //AURORA_STAT_MODEL_AUTOENCODER_HPP

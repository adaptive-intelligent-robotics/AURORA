//
// Created by Luca Grillotti on 08/05/2020.
//

#ifndef AURORA_STAT_CUSTOM_METRIC_HPP
#define AURORA_STAT_CUSTOM_METRIC_HPP

#include <boost/serialization/shared_ptr.hpp>
#include <boost/serialization/nvp.hpp>
#include <boost/fusion/sequence.hpp>
#include <boost/lexical_cast.hpp>
#include <sferes/stat/stat.hpp>
#include <sferes/fit/fitness.hpp>

namespace sferes {
    namespace stat {
        SFERES_STAT(CustomMetric, Stat) {
        public:
            template<typename EA, typename T>
            void write_container(const std::string &prefix, const EA &ea, const std::vector<T> metric) const {
                assert(metric.size() == ea.pop().size());
                std::cout << "writing..." << prefix << ea.gen() << std::endl;
                std::string fname = ea.res_dir() + "/"
                                    + prefix
                                    + boost::lexical_cast<std::string>(ea.gen())
                                    + std::string(".dat");

                std::ofstream ofs(fname.c_str());

                ofs.precision(17);
                for (size_t index_indiv = 0; index_indiv < ea.pop().size(); ++index_indiv) {
                    ofs << index_indiv << "    " << metric[index_indiv] << std::endl;
                }
            }

            template<typename EA>
            void refresh(EA &ea) {

            }
        };
    }
}

#endif //AURORA_STAT_CUSTOM_METRIC_HPP

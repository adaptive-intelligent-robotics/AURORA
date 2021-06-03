//
// Created by Luca Grillotti on 30/10/2019.
//

#ifndef SFERES2_STAT_PROJECTION_HPP
#define SFERES2_STAT_PROJECTION_HPP

#include <sferes/stat/stat.hpp>

namespace sferes {
    namespace stat {

        SFERES_STAT(Projection, Stat)
        {
        public:
            template<typename EA>
            void _write_container(const std::string &prefix, const EA &ea) const {
                std::cout << "writing..." << prefix << ea.gen() << std::endl;
                std::string fname = ea.res_dir() + "/"
                                    + prefix
                                    + boost::lexical_cast<std::string>(ea.gen())
                                    + std::string(".dat");

                std::ofstream ofs(fname.c_str());

                size_t offset = 0;
                ofs.precision(17);
                for (auto it = ea.pop().begin(); it != ea.pop().end(); ++it) {
                    ofs << offset << "    " << (*it)->fit().entropy() << "    ";

                    for (size_t dim = 0; dim < (*it)->fit().desc().size(); ++dim)
                        ofs << (*it)->fit().desc()[dim] << " ";
                    //ofs << " " << array(idx)->fit().value() << std::endl;
                    ofs << "     ";
                    for (size_t dim = 0; dim < (*it)->fit().gt().size(); ++dim)
                        ofs << (*it)->fit().gt()[dim] << " ";

                    // Save fitness value as well
                    ofs << "     ";
                    ofs << (*it)->fit().value() << " ";

                    // Save implicit fitness value as well
                    ofs << "     ";
                    ofs << (*it)->fit().implicit_fitness_value() << " ";

                    // Save novelty value as well
                    ofs << "     ";
                    ofs << (*it)->fit().novelty() << " ";

                    ofs << std::endl;
                    ++offset;
                }
            }

            template<typename EA>
            void refresh_instantly(EA &ea) {
                _write_container(std::string("proj_"), ea);
            }

            template<typename EA>
            void refresh_instantly(EA &ea, const std::string& prefix) {
                _write_container(prefix, ea);
            }

            template<typename EA>
            void refresh(EA &ea) {
                if (ea.gen() % Params::pop::dump_period_aurora == 0) {
                    _write_container(std::string("proj_"), ea);
                }
            }
        };

    }
}


#endif
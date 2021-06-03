//
// Created by Luca Grillotti on 06/01/2021.
//

#ifndef AURORA_STAT_DELETE_GEN_FILES_HPP
#define AURORA_STAT_DELETE_GEN_FILES_HPP

#include <cstdio>

#include <boost/lexical_cast.hpp>

#include <sferes/stat/stat.hpp>

namespace sferes {
  namespace stat {
    SFERES_STAT(DeleteGenFiles, Stat
    ) {
    public:
      template<typename E>
      void refresh(const E &ea) {
        if (ea.gen() % Params::pop::dump_period == 0) {
          int previous_gen_dump = static_cast<int>(ea.gen()) - 2 * static_cast<int>(Params::pop::dump_period);
          if (previous_gen_dump >= 0) {
            std::string fname = ea.res_dir() + "/gen_" + boost::lexical_cast<std::string>(previous_gen_dump);
            std::remove(fname.c_str());
            std::cout << "Deleting file: " << fname << std::endl;
          }
        }
      }
    };
  }
}

#endif // AURORA_STAT_DELETE_GEN_FILES_HPP

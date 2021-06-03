#ifndef STAT_CURRENT_GEN_HPP_
#define STAT_CURRENT_GEN_HPP_

#include <sferes/stat/stat.hpp>

namespace sferes {
    namespace stat {
        SFERES_STAT(CurrentGen, Stat
        ) {
        public:
        template<typename E>
        void refresh(const E &ea) {
            std::string fname = ea.res_dir() + "/current_gen";
            std::ofstream ofs(fname.c_str());
            ofs << "Current generation: " << ea.gen() << std::endl;
            ofs << "Current generation size: " << ea.pop().size() << std::endl;
            ofs.close();
        }
    };
}
}

#endif

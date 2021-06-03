//
// Created by Luca Grillotti on 13/04/2020.
//

#ifndef AURORA_STAT_MODIFIER_HPP
#define AURORA_STAT_MODIFIER_HPP

#include <boost/serialization/shared_ptr.hpp>
#include <boost/serialization/nvp.hpp>
#include <boost/fusion/sequence.hpp>
#include <sferes/stat/stat.hpp>
#include <sferes/fit/fitness.hpp>

namespace sferes {
    namespace stat {
        SFERES_STAT(Modifier, Stat) {
        public:
        template<typename E>
        void refresh(const E& ea) {
            assert(!ea.pop().empty());

            if (ea.dump_enabled()) {
                this->_create_log_file(ea, "stat_modifier.dat");
                (*this->_log_file) << ea.gen()
                                    << " " << Params::nov::l
                                    << " " << ea.pop().size() // Content of the archive is in the population;
                                    << std::endl;
            }
        }
    };
}
}

#endif //AURORA_STAT_MODIFIER_HPP

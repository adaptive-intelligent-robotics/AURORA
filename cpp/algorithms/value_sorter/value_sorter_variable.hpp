//
// Created by Luca Grillotti on 11/05/2020.
//

#ifndef AURORA_VALUE_SORTER_VARIABLE_HPP
#define AURORA_VALUE_SORTER_VARIABLE_HPP

#include <boost/shared_ptr.hpp>
#include <sferes/misc/rand.hpp>

#include "dbg_tools/dbg.hpp"

namespace aurora {
    namespace algo {
        namespace value_sorter {

            template <typename TPhen, typename TValueSorter1, typename TValueSorter2, typename TParams>
            struct ValueVariableSorter {
            public:
                typedef boost::shared_ptr<TPhen> indiv_t;

                void operator()(std::vector<indiv_t>& pop) {
                    if (sferes::misc::rand(1.f) < TParams::selector::proba_picking_selector_1) {
                        airl::dbg::out(airl::dbg::debug) << "Choosing Value Sorter 1\n";
                        value_sorter_1(pop);
                    } else {
                        airl::dbg::out(airl::dbg::debug) << "Choosing Value Sorter 2\n";
                        value_sorter_2(pop);
                    }
                }

            private:
                TValueSorter1 value_sorter_1;
                TValueSorter2 value_sorter_2;
            };

        }
    }
}


#endif //AURORA_VALUE_SORTER_VARIABLE_HPP

//
// Created by Luca Grillotti on 10/05/2020.
//

#ifndef AURORA_VALUE_SORTER_HPP
#define AURORA_VALUE_SORTER_HPP

#include <boost/shared_ptr.hpp>
#include <sferes/misc/rand.hpp>

namespace aurora {
    namespace algo {
        namespace value_sorter {

            template <typename TPhen, typename TValueSelector, typename TParams>
            struct ValueSorter {
            public:
                typedef boost::shared_ptr<TPhen> indiv_t;

                void operator()(std::vector<indiv_t>& pop) {
                    std::sort(pop.begin(),
                              pop.end(),
                              [](const indiv_t& indiv_1, const indiv_t& indiv_2) {
                                  return (TValueSelector::getValue(indiv_1) < TValueSelector::getValue(indiv_2));
                              });
                }
            };

        }
    }
}



#endif //AURORA_VALUE_SORTER_HPP

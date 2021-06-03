//
// Created by Luca Grillotti on 20/11/2019.
//

#ifndef SFERES_VARIABLE_SELECTOR_HPP
#define SFERES_VARIABLE_SELECTOR_HPP

#include <cstdlib>
#include <iostream>
#include <ctime>

namespace sferes {
    namespace qd {
        namespace selector {
            template<typename Phen, typename Selector1, typename Selector2, typename Params>

            class VariableSelector {

            public:
                typedef boost::shared_ptr <Phen> indiv_t;

                template<typename EA>
                void operator()(std::vector <indiv_t> &pop, const EA &ea) const {
                    if ((float) rand() / (RAND_MAX) < Params::selector::proba_picking_selector_1) {
                        selector_1(pop, ea);
                    } else {
                        selector_2(pop, ea);
                    }
                }

            private:
                Selector1 selector_1;
                Selector2 selector_2;
            };
        }
    }
}

#endif //SFERES_VARIABLE_SELECTOR_HPP

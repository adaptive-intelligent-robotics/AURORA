//
// Created by Luca Grillotti on 20/11/2019.
//

#ifndef SFERES_SURPRISE_VALUE_SELECTOR_HPP
#define SFERES_SURPRISE_VALUE_SELECTOR_HPP

namespace sferes {
    namespace qd {
        namespace selector {
            struct getSurprise {
                template <typename Phen>
                static inline double getValue(const Phen& p) {
                    return p->fit().entropy();
                }
            };
        }
    }
}

#endif //SFERES_SURPRISE_VALUE_SELECTOR_HPP

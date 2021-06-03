//
// Created by Luca Grillotti on 06/05/2020.
//

#ifndef AURORA_INITIALISE_GLOBAL_PARAMETERS_HPP
#define AURORA_INITIALISE_GLOBAL_PARAMETERS_HPP

namespace aurora {
    namespace algo {
        template <typename params_t>
        void initialise_global_variables() {
#ifdef PARAMS_FIXED_L
                params_t::nov::l = PARAMS_FIXED_L;
#else
                params_t::nov::l = 0;
#endif
        }
    }
}

#endif //AURORA_INITIALISE_GLOBAL_PARAMETERS_HPP

//
// Created by Luca Grillotti on 29/09/2020.
//

#ifndef AURORA_ALL_GEN_MUTATION_HPP
#define AURORA_ALL_GEN_MUTATION_HPP

#include <sferes/misc.hpp>
#include <sferes/stc.hpp>

namespace sferes {
  namespace gen {

    /**
     * Required Parameters (from TParams):
     * - TParams::full_gen_mutation::full_gen_mutation_rate -> proba to start mutating a genotype
     */
    template<typename TGen, typename TParams>
    class FullGenotypeMutationWrapper : public TGen
    {
    public:
      FullGenotypeMutationWrapper()
        : TGen(){};

      void
      mutate()
      {
        if (misc::rand<float>() < TParams::full_gen_mutation::full_gen_mutation_rate) {
          TGen::mutate();
        }
      }
    };

  } // namespace gen
} // namespace sferes

#endif // AURORA_ALL_GEN_MUTATION_HPP

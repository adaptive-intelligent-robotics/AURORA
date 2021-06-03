//
// Created by Luca Grillotti on 29/09/2020.
//

#ifndef AURORA_CHANGE_GEN_IN_PHEN_HPP
#define AURORA_CHANGE_GEN_IN_PHEN_HPP

namespace sferes {
  namespace phen {
    template<typename TPhen, typename TGenNew>
    struct ChangeGenotypeTypeForPhenotype
    {};

    template<template<typename, typename, typename, typename> class TPhen,
             typename TGenOld,
             typename TFit,
             typename TParams,
             typename TExact,
             typename TGenNew>
    struct ChangeGenotypeTypeForPhenotype<TPhen<TGenOld, TFit, TParams, TExact>, TGenNew>
    {
      typedef TPhen<TGenNew, TFit, TParams, TExact> phen_t;
    };

  } // namespace phen
} // namespace sferes

#endif // AURORA_CHANGE_GEN_IN_PHEN_HPP

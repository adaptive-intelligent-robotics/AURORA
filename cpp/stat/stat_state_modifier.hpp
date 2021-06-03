//
// Created by Luca Grillotti on 23/12/2020.
//

#ifndef AURORA_STAT_STATE_MODIFIER_HPP
#define AURORA_STAT_STATE_MODIFIER_HPP

#include <boost/serialization/nvp.hpp>
#include <boost/serialization/shared_ptr.hpp>
#include <sferes/fit/fitness.hpp>
#include <sferes/stat/stat.hpp>

namespace sferes {
  namespace stat {
    template<typename TArchive>
    struct FusionVectorSerialiser {
      explicit FusionVectorSerialiser(TArchive& _ar) : ar(_ar) {}

      template<typename TModifierItem>
      void operator()(TModifierItem& modifier_item) const {
        ar& BOOST_SERIALIZATION_NVP(modifier_item);
      }

      TArchive& ar;
    };

    // a statistics class that saves the modifier
    // this is useful for restarting sferes when it is killed
    template<typename TModifier, typename TPhen, typename TParams, typename Exact = stc::Itself>
    class StateModifier
      : public Stat<TPhen, TParams, typename stc::FindExact<StateModifier<TPhen, TParams, Exact>, Exact>::ret>
    {
    public:
      template<typename EA>
      void
      refresh(const EA& ea)
      {
        m_modifier = ea.fit_modifier();
        m_l_distance_threshold = TParams::nov::l;
      }

      const TModifier&
      modifier() const
      {
        return m_modifier;
      }

      double
      l_distance_threshold() const
      {
        return m_l_distance_threshold;
      }

      template<class TArchive>
      void
      serialize(TArchive& ar, const unsigned int version)
      {
        boost::fusion::for_each(m_modifier, FusionVectorSerialiser<TArchive>(ar));
        ar& BOOST_SERIALIZATION_NVP(m_l_distance_threshold);
      }

    protected:
      TModifier m_modifier;
      double m_l_distance_threshold;
    };
  } // namespace stat
} // namespace sferes

#endif // AURORA_STAT_STATE_MODIFIER_HPP

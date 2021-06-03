//
// Created by Luca Grillotti on 23/12/2020.
//

#ifndef AURORA_DUMMY_SERIALISABLE_HPP
#define AURORA_DUMMY_SERIALISABLE_HPP

namespace sferes {
  namespace modif {
    SFERES_CLASS(DummySerialisable){
      public:
        template<typename Ea>
        void apply(Ea & ea){}

        template<class Archive>
        void serialize(Archive & ar, const unsigned int version){}

        void copy(const DummySerialisable<>& other_modifier) {}
    };
  }
} // namespace sferes

#endif // AURORA_DUMMY_SERIALISABLE_HPP

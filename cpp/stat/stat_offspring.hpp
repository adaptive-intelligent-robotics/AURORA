//
// Created by Luca Grillotti on 30/09/2020.
//

#ifndef AURORA_STAT_OFFSPRING_HPP
#define AURORA_STAT_OFFSPRING_HPP

#include <iostream>
#include <numeric>
#include <string>

#include <boost/lexical_cast.hpp>

#include <sferes/stat/stat.hpp>

namespace sferes {
  namespace stat {
    SFERES_STAT(QdOffspring, Stat)
    {
    public:
      typedef std::vector<boost::shared_ptr<Phen>> pop_t;

      template<typename EA>
      void refresh(const EA& ea)
      {
        m_offspring.clear();
        for (auto it = ea.offspring().begin(); it != ea.offspring().end(); ++it) {
          m_offspring.push_back(*it);
        }

        if (ea.gen() % Params::pop::dump_period_aurora == 0) {
          _write_container(std::string("offspring_"), ea);
        }
      }

      template<typename EA>
      void _write_container(const std::string& prefix, const EA& ea) const
      {
        std::cout << "writing..." << prefix << ea.gen() << std::endl;
        std::string fname =
          ea.res_dir() + "/" + prefix + boost::lexical_cast<std::string>(ea.gen()) + std::string(".dat");

        std::ofstream ofs(fname.c_str());

        size_t offset = 0;
        ofs.precision(17);
        for (auto it = ea.offspring().begin(); it != ea.offspring().end(); ++it) {
          ofs << offset << "    " << (*it)->fit().entropy() << "    ";

          for (size_t dim = 0; dim < (*it)->fit().desc().size(); ++dim)
            ofs << (*it)->fit().desc()[dim] << " ";

          ofs << "    ";
          for (size_t dim = 0; dim < (*it)->fit().gt().size(); ++dim)
            ofs << (*it)->fit().gt()[dim] << " ";

          ofs << std::endl;
          ++offset;
        }
      }

      template<class Archive>
      void serialize(Archive & ar, const unsigned int version)
      {
        ar& BOOST_SERIALIZATION_NVP(m_offspring);
      }

      const pop_t& offspring() const { return m_offspring; }

    protected:
      pop_t m_offspring;
    }; // QdOffspring
  }    // namespace stat
} // namespace sferes

#endif // AURORA_STAT_OFFSPRING_HPP

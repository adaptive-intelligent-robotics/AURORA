//
// Created by Luca Grillotti on 09/01/2020.
//

#ifndef AURORA_QUALITY_DIVERSITY_AURORA_PROJECT_HPP
#define AURORA_QUALITY_DIVERSITY_AURORA_PROJECT_HPP

#include <sferes/ea/ea.hpp>
#include <sferes/qd/quality_diversity.hpp>

#include <sferes/eval/parallel.hpp>
#include <sferes/qd/selector/noselection.hpp>
#include <sferes/stat/state.hpp>
#include <sferes/stat/state_qd.hpp>

#include "stat/stat_state_modifier.hpp"
#include "stat/stat_delete_gen_files.hpp"

namespace sferes {
  namespace qd {

    // Structure for loading modifier components (including AE model parameters) when resuming QD algo
    template<typename T, typename A>
    struct ResumeModifier
    {
      template<typename EA>
      void
      resume(EA& ea)
      {
        typedef sferes::stat::
          StateModifier<typename EA::modifier_t, typename EA::phen_t, typename EA::params_t>
            stat_state_modifier_t;
        const stat_state_modifier_t& stat = *boost::fusion::find<stat_state_modifier_t>(ea.stat());
        ea.set_modifier(stat.modifier());
        ea.set_l_distance_threshold(stat.l_distance_threshold());
      }
    };

    // Do nothing if there is no  archive stat
    template<typename T>
    struct ResumeModifier<T, typename boost::fusion::result_of::end<T>::type>
    {
      template<typename EA>
      void
      resume(EA& ea)
      {}
    };

    // Default Stat
    template<typename TPhen, typename TModifier, typename TParams>
    struct DefaultStatWithStateModifier {
      typedef typename boost::mpl::if_<boost::fusion::traits::is_sequence<TModifier>,
                                       TModifier,
                                       boost::fusion::vector<TModifier>>::type
        sequence_modifier_t;

      typedef typename boost::fusion::vector<sferes::stat::StateModifier<sequence_modifier_t, TPhen, TParams>,
                                             sferes::stat::StateQD<TPhen, TParams>,
                                             sferes::stat::DeleteGenFiles<TPhen, TParams>
                                             >
        default_state_t;
    };

    // Inherits most functionalities from QualityDiversity class in Sferes2
    // + quick hack to have "write" access to the container, this need to be added to the main API later.
    // + adds custom resuming functionalities (for saving also the modifier components, including the AE model)
    template<typename Phen,
             typename Eval,
             typename Stat,
             typename FitModifier,
             typename Select,
             typename Container,
             typename Params,
             typename Exact = stc::Itself,
             typename DefaultStat = typename DefaultStatWithStateModifier<Phen, FitModifier, Params>::default_state_t>
    class QualityDiversityAuroraProject
      : public sferes::qd::QualityDiversity<
          Phen,
          Eval,
          Stat,
          FitModifier,
          Select,
          Container,
          Params,
          typename stc::FindExact<
            QualityDiversityAuroraProject<Phen, Eval, Stat, FitModifier, Select, Container, Params, Exact, DefaultStat>,
            Exact>::ret,
          DefaultStat
          >
    {

    public:
      typedef Phen phen_t;
      typedef boost::shared_ptr<Phen> indiv_t;
      typedef typename std::vector<indiv_t> pop_t;
      typedef typename pop_t::iterator it_t;

      typedef typename
      boost::mpl::if_<boost::fusion::traits::is_sequence<FitModifier>,
        FitModifier,
        boost::fusion::vector<FitModifier> >::type modifier_t;

      // Copy of the state structure which add State automatically

#ifdef SFERES_NO_STATE
      typedef Stat stat_t;
#else
      typedef typename boost::fusion::joint_view<stat_qd_with_default_t<Stat, DefaultStat>,
                                                 boost::fusion::vector<stat::State<Phen, Params>>
                                                 > joint_qd_t;
      typedef typename boost::fusion::result_of::as_vector<joint_qd_t>::type  stat_t;
#endif

      void
      set_modifier(const modifier_t& saved_modifier)
      {
        boost::fusion::at_c<0>(this->_fit_modifier).copy(boost::fusion::at_c<0>(saved_modifier));
      }

      void
      set_l_distance_threshold(double l_distance_threshold)
      {
        Params::nov::l = l_distance_threshold;
      }

      // Override the resume function
      void
      resume(const std::string& fname)
      {
        dbg::trace trace("ea", DBG_HERE);

        // Create directory, load file
        this->_make_res_dir();
        this->_set_status("resumed");
        if ((boost::fusion::find<sferes::stat::State<Phen, Params>>(this->_stat) ==
             boost::fusion::end(this->_stat)) or
            (boost::fusion::find<sferes::stat::StateModifier<modifier_t, Phen, Params>>(this->_stat) ==
             boost::fusion::end(this->_stat)) or
            (boost::fusion::find<sferes::stat::StateQD<Phen, Params>>(this->_stat) ==
             boost::fusion::end(this->_stat))) {
          std::cout << "WARNING: no State or Archive found in stat_t, cannot resume" << std::endl;
          return;
        }
        this->_load(fname);

        // Use ea Resume structure
        typedef typename boost::fusion::result_of::find<stat_t, sferes::stat::State<Phen, Params>>::type
          has_state_t;

        sferes::ea::Resume<stat_t, has_state_t> r;
        r.resume(*this);

        // Use new Resume structure
        typedef typename boost::fusion::result_of::find<stat_t, sferes::stat::StateModifier<modifier_t, Phen, Params>>::type
          has_archive_t;

        sferes::qd::ResumeModifier<stat_t, has_archive_t> a;
        a.resume(*this);

        // Use StateQD to save Offspring and Parents
        typedef typename boost::fusion::result_of::find<stat_t, sferes::stat::StateQD<Phen, Params>>::type
          has_state_qd_t;

        sferes::qd::ResumeQD<stat_t, has_state_qd_t> resume_state_qd;
        resume_state_qd.resume(*this);

        // Perform few test and resume algorithm
        assert(!this->_pop.empty()); // test pop size
        std::cout << "Resuming at gen " << this->_gen;
        std::cout << std::endl;
        for (; this->_gen < Params::pop::nb_gen && !this->_stop; ++this->_gen)
          this->_iter();
        if (!this->_stop)
          this->_set_status("finished");
      }

      void
      recover_population(const std::string& fname)
      {
        dbg::trace trace("ea", DBG_HERE);

        // Create directory, load file
        this->_make_res_dir();
        this->_set_status("resumed");
        if ((boost::fusion::find<sferes::stat::State<Phen, Params>>(this->_stat) ==
             boost::fusion::end(this->_stat)) or
            (boost::fusion::find<sferes::stat::StateModifier<modifier_t, Phen, Params>>(this->_stat) ==
             boost::fusion::end(this->_stat)) or
            (boost::fusion::find<sferes::stat::StateQD<Phen, Params>>(this->_stat) ==
             boost::fusion::end(this->_stat))) {
          std::cout << "WARNING: no State or Archive found in stat_t, cannot resume" << std::endl;
          return;
        }
        this->_load(fname);

        // Use ea Resume structure
        typedef typename boost::fusion::result_of::find<stat_t, sferes::stat::State<Phen, Params>>::type
          has_state_t;

        sferes::ea::Resume<stat_t, has_state_t> r;
        r.resume(*this);

// //         Use new Resume structure
//        typedef typename boost::fusion::result_of::find<stat_t, sferes::stat::StateModifier<modifier_t, Phen, Params>>::type
//          has_archive_t;
//
//        sferes::qd::ResumeModifier<stat_t, has_archive_t> a;
//        a.resume(*this);

        // Use StateQD to save Offspring and Parents
        typedef typename boost::fusion::result_of::find<stat_t, sferes::stat::StateQD<Phen, Params>>::type
          has_state_qd_t;

        sferes::qd::ResumeQD<stat_t, has_state_qd_t> resume_state_qd;
        resume_state_qd.resume(*this);

        // Perform few test and resume algorithm
        assert(!this->_pop.empty()); // test pop size
        std::cout << "Resuming at gen " << this->_gen;

      }

      void
      update_container_with_modifier() {
        using Mat = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;


        Params::nov::l = -1;

        // same instructions as in update_descriptors but without updating l
        Mat data;
        boost::fusion::at_c<0>(this->_fit_modifier).collect_dataset(data, *this, true); // gather the data from the indiv in the archive into a dataset
        boost::fusion::at_c<0>(this->_fit_modifier).train_network(data);
        boost::fusion::at_c<0>(this->_fit_modifier).update_container(*this, false);  // clear the archive and re-fill it using the new network
        this->pop_advers.clear(); // clearing adversarial examples


//        pop_t entire_pop_container;
//        this->container().get_full_content(entire_pop_container);
//        boost::fusion::at_c<0>(this->_fit_modifier).assign_descriptor_to_population(*this, this->offspring(), entire_pop_container, true);
      }

      // Override the resume function
      pop_t
      get_pop_from_gen_file(const std::string& fname)
      {
        dbg::trace trace("ea", DBG_HERE);
        std::cout << "loading " << fname << std::endl;
        std::ifstream ifs(fname.c_str());
        if (ifs.fail()) {
          std::cerr << "Cannot open :" << fname
                    << "(does file exist ?)" << std::endl;
          exit(1);
        }
#ifdef SFERES_XML_WRITE
        typedef boost::archive::xml_iarchive ia_t;
#else
        typedef boost::archive::binary_iarchive ia_t;
#endif
        ia_t ia(ifs);
        sferes::stat::State<Phen, Params> stat_state;
        boost::fusion::for_each(this->_stat, sferes::ea::ReadStat_f<ia_t>(ia));
        stat_state = *boost::fusion::find<sferes::stat::State<Phen, Params>>(this->_stat);
        return stat_state.pop();
      }

      pop_t pop_advers;
      // pop_t& get_pop_advers() { return this->pop_advers; }

      Container&
      container()
      {
        return this->_container;
      }

      void
      add(pop_t& pop_off, std::vector<bool>& added, pop_t& pop_parents)
      {
        this->_add(pop_off, added, pop_parents);
      }

      // Same function, but without the need of parent.
      void
      add(pop_t& pop_off, std::vector<bool>& added)
      {
        std::cout << "adding with l: " << Params::nov::l << std::endl;
        this->_add(pop_off, added);
      }
    };
  } // namespace algo
} // namespace aurora

#endif // AURORA_QUALITY_DIVERSITY_AURORA_PROJECT_HPP

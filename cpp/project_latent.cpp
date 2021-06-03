//| This file is a part of the sferes2 framework.
//| Copyright 2016, ISIR / Universite Pierre et Marie Curie (UPMC)
//| Main contributor(s): Jean-Baptiste Mouret, mouret@isir.fr
//|
//| This software is a computer program whose purpose is to facilitate
//| experiments in evolutionary computation and evolutionary robotics.
//|
//| This software is governed by the CeCILL license under French law
//| and abiding by the rules of distribution of free software.  You
//| can use, modify and/ or redistribute the software under the terms
//| of the CeCILL license as circulated by CEA, CNRS and INRIA at the
//| following URL "http://www.cecill.info".
//|
//| As a counterpart to the access to the source code and rights to
//| copy, modify and redistribute granted by the license, users are
//| provided only with a limited warranty and the software's author,
//| the holder of the economic rights, and the successive licensors
//| have only limited liability.
//|
//| In this respect, the user's attention is drawn to the risks
//| associated with loading, using, modifying and/or developing or
//| reproducing the software by the user in light of its specific
//| status of free software, that may mean that it is complicated to
//| manipulate, and that also therefore means that it is reserved for
//| developers and experienced professionals having in-depth computer
//| knowledge. Users are therefore encouraged to load and test the
//| software's suitability as regards their requirements in conditions
//| enabling the security of their systems and/or data to be ensured
//| and, more generally, to use and operate it in the same conditions
//| as regards security.
//|
//| The fact that you are presently reading this means that you have
//| had knowledge of the CeCILL license and that you accept its terms.

#include <iostream>
#include <algorithm>
#include <unistd.h>

#include <boost/program_options.hpp>

#include <torch/torch.h>
#include <torch/script.h> // One-stop header.

#include <sferes/eval/parallel.hpp>
#include <sferes/gen/evo_float.hpp>
#include <sferes/modif/dummy.hpp>
#include <sferes/phen/parameters.hpp>
#include <sferes/run.hpp>
#include <sferes/stat/best_fit.hpp>
#include <sferes/stat/qd_container.hpp>
#include <sferes/stat/qd_selection.hpp>
#include <sferes/stat/qd_progress.hpp>

#include <sferes/fit/fit_qd.hpp>
#include <sferes/qd/container/archive.hpp>
#include <sferes/qd/container/kdtree_storage.hpp>
#include <sferes/qd/quality_diversity.hpp>
#include <sferes/qd/selector/value_selector.hpp>
#include <sferes/qd/selector/score_proportionate.hpp>

#include <robot_dart/gui/magnum/base_application.hpp>

// replacing physics with fit_hexa
// #include "minimal_physics.hpp"

#include "project_includes.hpp"


namespace aurora {

    struct Arguments {
        size_t number_threads;
        std::string path_ns;
        std::string path_aurora;
        std::string path_network;
        std::string path_save;
        size_t behav_dim_ns;
    };

    void get_arguments(const boost::program_options::options_description &desc, Arguments &arg, int argc, char **argv) {
        // For the moment, only returning number of threads
        boost::program_options::variables_map vm;
        boost::program_options::parsed_options parsed = boost::program_options::command_line_parser(argc, argv).options(
                desc).allow_unregistered().run();

        boost::program_options::store(parsed, vm);
        boost::program_options::notify(vm);
        arg.number_threads = vm["number-threads"].as<size_t>();
        arg.behav_dim_ns = vm["behav-dim-ns"].as<size_t>();
    }
}


int main(int argc, char **argv) {
    boost::program_options::options_description desc;
    aurora::Arguments arg{};

    desc.add_options()
            ("number-threads", boost::program_options::value<size_t>(), "Set Number of Threads")
            ("path-ns", boost::program_options::value<std::string>(&arg.path_ns)->default_value(""), "Set path of proj file (representing the NS archive)")
            ("path-aurora", boost::program_options::value<std::string>(&arg.path_aurora)->default_value(""), "Set path of proj file (representing the AURORA archive)")
            ("path-network", boost::program_options::value<std::string>(&arg.path_network)->default_value(""), "Set path of Network")
            ("path-save", boost::program_options::value<std::string>(&arg.path_save)->default_value(""), "Set path of saved file")
            ("behav-dim-ns", boost::program_options::value<size_t>(), "Number of dimensions for the NS behavioural descriptor");

    aurora::get_arguments(desc, arg, argc, argv);

    srand(time(0));

    robot_dart::gui::magnum::GlobalData::instance()->set_max_contexts(arg.number_threads);
    tbb::task_scheduler_init init(arg.number_threads);

    constexpr aurora::env::Env environment = aurora::get_env();
    constexpr aurora::algo::Algo algorithm = aurora::algo::Algo::aurora_curiosity;
    constexpr int latent_space_size = LATENT_SPACE_SIZE;
    constexpr bool use_colors = aurora::get_use_colors();
    constexpr bool use_videos = aurora::get_use_videos();

    typedef aurora::SpecificParams specific_params_t;
    typedef aurora::DefaultParamsFactory<environment,
      specific_params_t>::default_params_t default_params_t;
    typedef aurora::ParamsAlgo<algorithm, default_params_t, specific_params_t> params_t;

    if (environment == aurora::env::Env::hard_maze) {
        aurora::env::init_fastsim_settings<params_t>();
    } else { // if using the hexapod environment
        aurora::env::load_and_init_robot();
    }

    typedef aurora::env::Environment<environment, params_t> env_t;
    typedef aurora::algo::AlgorithmFactory<algorithm, env_t>::algo_t algo_t;

    algo_t::update_parameters();
    typedef algo_t::ea_t ea_t;

    ea_t ea;

    aurora::analysis::project_to_latent_space<ea_t, algo_t::fit_t, algo_t::phen_t, algo_t::param_t>
            (ea, arg.path_ns, arg.path_aurora, arg.path_network, arg.path_save, arg.behav_dim_ns);

    return 0;
}

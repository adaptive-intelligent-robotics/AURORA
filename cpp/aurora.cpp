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


#include <sys/stat.h>
#include <cstdint>
#include <iostream>
#include <cmath>
#include <algorithm>
#include <unistd.h>

#include <Eigen/Core>
#include "compilation_variables.hpp"

#include <boost/process.hpp>
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
#include <robox2d/gui/magnum/base_application.hpp>

// replacing physics with fit_hexa
// #include "minimal_physics.hpp"

#include "project_includes.hpp"

namespace aurora {

    struct Arguments {
        size_t number_threads;
        int step_measures;
    };

    void get_arguments(const boost::program_options::options_description &desc, Arguments &arg, int argc, char **argv) {
        // For the moment, only returning number of threads
        boost::program_options::variables_map vm;
        boost::program_options::parsed_options parsed = boost::program_options::command_line_parser(argc, argv).options(
                desc).allow_unregistered().run();

        boost::program_options::store(parsed, vm);
        boost::program_options::notify(vm);
        arg.number_threads = vm["number-threads"].as<size_t>();
    }
}

int main(int argc, char **argv) {

    dbg::enable(dbg::tracing, "ea", true);
    dbg::attach_ostream(dbg::tracing, std::cout);
    airl::dbg::attach_ostream(airl::dbg::tracing, std::cout);
    airl::dbg::attach_ostream(airl::dbg::debug, std::cout);
    // airl::dbg::enable_all(airl::dbg::debug, true);
    airl::dbg::enable_all(airl::dbg::info, true);
    airl::dbg::enable_all(airl::dbg::debug, true);
    airl::dbg::enable_all(airl::dbg::tracing, true);
    airl::dbg::enable_level_prefix(true);
    airl::dbg::enable_time_prefix(true);

    std::cout << aurora::get_lp_norm() << std::endl;

    boost::program_options::options_description desc;
    aurora::Arguments arg{};

    constexpr int step_measures_default_value = 10;
    desc.add_options()
            ("number-threads", boost::program_options::value<size_t>(), "Set Number of Threads")
            ("step-measures", boost::program_options::value<int>(&arg.step_measures)->default_value(step_measures_default_value), "Step between two stored measures in Hard Maze environment");

    aurora::get_arguments(desc, arg, argc, argv);

    srand(time(0));

    robot_dart::gui::magnum::GlobalData::instance()->set_max_contexts(arg.number_threads);
    robox2d::gui::magnum::GlobalData::instance()->set_max_contexts(arg.number_threads);

    tbb::task_scheduler_init init(arg.number_threads);
    torch::set_num_threads(arg.number_threads);

    constexpr aurora::env::Env environment = aurora::get_env();
    constexpr aurora::algo::Algo algorithm = aurora::get_algo();

    typedef aurora::SpecificParams specific_params_t;
    typedef aurora::DefaultParamsFactory<environment,
      specific_params_t>::default_params_t default_params_t;
    typedef aurora::ParamsAlgo<algorithm, default_params_t, specific_params_t> params_t;

    params_t::step_measures = arg.step_measures;
    params_t::nov::use_fixed_l = specific_params_t::use_fixed_l;

    aurora::EnvironmentInitialiser<default_params_t> environment_initialiser;
    environment_initialiser.run();

    static_assert(
            (params_t::encoder_type != aurora::EncoderType::lstm_ae)
            || (params_t::use_videos),
            "Use of LSTM AE => need for use_videos"
    );


    typedef aurora::env::Environment<environment, params_t> env_t;
    typedef aurora::algo::AlgorithmFactory<algorithm, env_t>::algo_t algo_t;

    aurora::algo::initialise_global_variables<algo_t::param_t>();
    algo_t::update_parameters();

    typedef algo_t::ea_t ea_t;

    ea_t ea;

    sferes::run_ea(argc, argv, ea, desc);

    return 0;
}

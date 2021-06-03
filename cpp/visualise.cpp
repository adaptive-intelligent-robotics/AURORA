//
// Created by Luca Grillotti on 03/04/2020.
//

#include <iostream>
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

// replacing physics with fit_hexa
// #include "minimal_physics.hpp"

#include "project_includes.hpp"

namespace visu {

    struct Arguments {
        std::string path_archive;
        size_t index;
    };

    void get_arguments(const boost::program_options::options_description &desc, Arguments &arg, int argc, char **argv) {
        // For the moment, only returning number of threads
        boost::program_options::variables_map vm;
        boost::program_options::parsed_options parsed = boost::program_options::command_line_parser(argc, argv).options(
                desc).allow_unregistered().run();

        boost::program_options::store(parsed, vm);
        boost::program_options::notify(vm);
        arg.path_archive = vm["path"].as<std::string>();
        arg.index = vm["index"].as<size_t>();
    }
}

int main(int argc, char **argv) {
    boost::program_options::options_description desc;
    desc.add_options()("path", boost::program_options::value<std::string>(), "Set path of proj file (representing the archive)");
    desc.add_options()("index", boost::program_options::value<size_t>(), "Index of behaviour to show in that file");

    visu::Arguments arg{};
    visu::get_arguments(desc, arg, argc, argv);

    srand(time(0));

    tbb::task_scheduler_init init(1);

    constexpr aurora::env::Env environment = aurora::get_env();
    constexpr aurora::algo::Algo algorithm = aurora::get_algo();

    typedef aurora::SpecificParams specific_params_t;
    typedef aurora::DefaultParamsFactory<environment,
      specific_params_t>::default_params_t default_params_t;
    typedef aurora::ParamsAlgo<algorithm, default_params_t, specific_params_t> params_t;

    params_t::nov::use_fixed_l = specific_params_t::use_fixed_l;

    aurora::EnvironmentInitialiser<default_params_t> environment_initialiser;
    environment_initialiser.run();

    typedef aurora::env::Environment<environment, params_t> env_t;
    typedef aurora::algo::AlgorithmFactory<algorithm, env_t>::algo_t algo_t;

    algo_t::update_parameters();
    typedef algo_t::ea_t ea_t;
    ea_t ea;

    aurora::visualise_one_behaviour<ea_t, env_t::fit_t, env_t::param_t>(ea, arg.path_archive, arg.index);

    return 0;
}


//
// Created by Luca Grillotti on 05/05/2020.
//

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

namespace visu {
    struct ArgumentsImagesAdjacentBehaviours {
        std::string path_archive;
        std::string path_proj;
        std::string prefix_save;
        size_t index;
        size_t nb_nearest_neighbours;
        size_t behav_dim_ns;
        size_t number_threads;
        bool do_calculate_pixel_distances;
    };

    void get_arguments(const boost::program_options::options_description &desc,
            ArgumentsImagesAdjacentBehaviours &arg,
            int argc,
            char **argv) {
        // For the moment, only returning number of threads
        boost::program_options::variables_map vm;
        boost::program_options::parsed_options parsed = boost::program_options::command_line_parser(argc, argv).options(
                desc).allow_unregistered().run();

        boost::program_options::store(parsed, vm);
        boost::program_options::notify(vm);
        arg.path_archive = vm["path-archive"].as<std::string>();
        arg.path_proj = vm["path-proj"].as<std::string>();
        arg.index = vm["index"].as<size_t>();
        arg.nb_nearest_neighbours = vm["nb-nn"].as<size_t>();
        arg.prefix_save = vm["prefix-save"].as<std::string>();
        arg.behav_dim_ns = vm["behav-dim-ns"].as<size_t>();
        arg.number_threads = vm["number-threads"].as<size_t>();
        arg.do_calculate_pixel_distances = vm["calc-pix-dist"].as<bool>();
    }
}

int main(int argc, char **argv) {
    visu::ArgumentsImagesAdjacentBehaviours arg{};

    boost::program_options::options_description desc;
    desc.add_options()
            ("path-proj", boost::program_options::value<std::string>(), "Set path of proj file (representing the archive)")
            ("path-archive", boost::program_options::value<std::string>()->default_value(""), "Set path of proj file (representing the archive)")
            ("index", boost::program_options::value<size_t>(), "Index of behaviour to show in that file")
            ("nb-nn", boost::program_options::value<size_t>(), "Number of Nearest Neighbours to take into account")
            ("prefix-save", boost::program_options::value<std::string>()->default_value(""), "Set prefix for path of saved file")
            ("behav-dim-ns", boost::program_options::value<size_t>(), "Number of dimensions for the NS behavioural descriptor")
            ("calc-pix-dist", boost::program_options::bool_switch()->default_value(false), "Do calculate the mean pixel distance to the k-NN")
            ("number-threads", boost::program_options::value<size_t>(), "Set Number of Threads");


    visu::get_arguments(desc, arg, argc, argv);

    srand(time(0));

    robot_dart::gui::magnum::GlobalData::instance()->set_max_contexts(arg.number_threads);
    tbb::task_scheduler_init init(arg.number_threads);


    constexpr aurora::env::Env environment = aurora::get_env();
    constexpr aurora::algo::Algo algorithm = aurora::get_algo();
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

    if (!arg.do_calculate_pixel_distances) {
        aurora::analysis::save_images_nearest_neighbours<ea_t, algo_t::fit_t, algo_t::phen_t, algo_t::param_t>
                (ea, arg.path_archive, arg.path_proj, arg.prefix_save, arg.behav_dim_ns, arg.index, arg.nb_nearest_neighbours);
    } else {
        aurora::analysis::save_all_pixel_distances<ea_t, algo_t::fit_t, algo_t::phen_t, algo_t::param_t>
                (ea, arg.path_archive, arg.path_proj, arg.prefix_save, arg.behav_dim_ns, arg.nb_nearest_neighbours);
    }


    return 0;
}


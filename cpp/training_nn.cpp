//
// Created by Luca Grillotti on 13/06/2020.
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

#include <torch/torch.h>

namespace aurora {

    struct Arguments {
        std::string path_sequences_measures;
        std::string path_save;
    };

    void get_arguments(const boost::program_options::options_description &desc, Arguments &arg, int argc, char **argv) {
        // For the moment, only returning number of threads
        boost::program_options::variables_map vm;
        boost::program_options::parsed_options parsed = boost::program_options::command_line_parser(argc, argv).options(
                desc).allow_unregistered().run();

        boost::program_options::store(parsed, vm);
        boost::program_options::notify(vm);
    }
}


int main(int argc, char **argv) {
    boost::program_options::options_description desc;
    aurora::Arguments arg{};

    desc.add_options()
            ("path-save", boost::program_options::value<std::string>(&arg.path_save)->default_value(""), "Set path of saved file")
            ("path-sequences-measures", boost::program_options::value<std::string>(&arg.path_sequences_measures)->default_value(""), "Set path to saved sequences of measures");

    aurora::get_arguments(desc, arg, argc, argv);

    srand(time(0));

    tbb::task_scheduler_init init(1);
//    torch::set_num_threads(1);

    constexpr aurora::env::Env environment = aurora::get_env();
    constexpr aurora::algo::Algo algorithm = aurora::get_algo();
    constexpr int latent_space_size = LATENT_SPACE_SIZE;
    constexpr bool use_colors = aurora::get_use_colors();
    constexpr bool use_videos = aurora::get_use_videos();

    typedef aurora::SpecificParams specific_params_t;
    typedef aurora::DefaultParamsFactory<environment,
      specific_params_t>::default_params_t default_params_t;
    typedef aurora::ParamsAlgo<algorithm, default_params_t, specific_params_t> params_t;

    aurora::EnvironmentInitialiser<default_params_t> environment_initialiser;
    environment_initialiser.run();

    typedef aurora::env::Environment<environment, params_t> env_t;
    typedef aurora::algo::AlgorithmFactory<algorithm, env_t>::algo_t algo_t;

    algo_t::update_parameters();
    typedef algo_t::ea_t ea_t;

    ea_t ea;


    typedef env_t::fit_t fit_t;
    typedef env_t::phen_t phen_t;
    typedef boost::shared_ptr<phen_t> indiv_t;
    typedef typename std::vector<indiv_t> pop_t;

    using Mat = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

    auto &fit_modifier = ea.template fit_modifier<0>();

    const std::string path_sequences_measures = arg.path_sequences_measures;

    Interactive_map<params_t> imap;

    archive_content_t archive_ns_individuals;
    archive_content_t archive_aurora_individuals;

    std::cout << "Read file and collect sequences measures" << std::endl;
    std::vector<std::vector<float>> v_successive_measures;
    imap.load_stat_sequence_observations(path_sequences_measures, v_successive_measures);

    std::cout << "Collect Dataset" << std::endl;
    Mat m_sequences_measures_complete, m_sequences_measures;
    fit_modifier.get_data(v_successive_measures, m_sequences_measures_complete);

    constexpr int STEP_COL = 1; // TODO: to fix (do not forget that there are several measures at each timestep)
    typedef Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> Mat;
    m_sequences_measures = Eigen::Map<Mat, 0, Eigen::Stride<Eigen::Dynamic, STEP_COL>>(m_sequences_measures_complete.data(), m_sequences_measures_complete.rows(), (m_sequences_measures_complete.cols() + STEP_COL - 1) / STEP_COL, Eigen::Stride<Eigen::Dynamic, STEP_COL>(m_sequences_measures_complete.outerStride(), STEP_COL));

    RescaleFeature<params_t> prep(0.001f);
    prep.init(m_sequences_measures);
    Mat scaled_data_;
    prep.apply(m_sequences_measures, scaled_data_);
    Mat transformed_data = scaled_data_;

    // TODO: To change! (mean on rows instead of cols)
    constexpr int c_len_running_mean = 20;
    for (int index_col=c_len_running_mean-1; index_col < transformed_data.rows(); ++index_col) {
        transformed_data.col(index_col) = scaled_data_.middleCols<c_len_running_mean>(index_col - c_len_running_mean + 1).rowwise().mean();
    }
    Mat scaled_data = transformed_data;
//    Mat scaled_data =  transformed_data.bottomRows(transformed_data.rows() - c_len_running_mean);
//    Mat scaled_data = scaled_data_;

    float final_entropy = fit_modifier.get_network_loader()->training(scaled_data);

    std::cout << "STAT" << std::endl;
    sferes::stat::ImagesReconstructionObs<phen_t, params_t> stat_reconstructions;

    std::cout << "STAT - compute reconstructions by hanb" << std::endl;
    Mat scaled_res, reconstruction_obs_population;
    fit_modifier.get_network_loader()->get_reconstruction(scaled_data, scaled_res);
    constexpr bool do_clip_between_zero_one = aurora::does_encode_images();
    prep.deapply(scaled_res, reconstruction_obs_population, do_clip_between_zero_one);


    stat_reconstructions._write_container_measures(arg.path_save, ea, reconstruction_obs_population);
    stat_reconstructions._write_container_measures(arg.path_save + "_scaled_obs", ea, scaled_data);
    stat_reconstructions._write_container_measures(arg.path_save + "_scaled_res", ea, scaled_res);

    return 0;
}


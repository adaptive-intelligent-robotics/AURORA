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
    std::string path_file_to_load;
    std::vector<size_t> indexes;
    std::string name_video;
    bool use_archive_dat;
  };

  void get_arguments(const boost::program_options::options_description &desc, Arguments &arg, int argc, char **argv) {
    // For the moment, only returning number of threads
    boost::program_options::variables_map vm;
    boost::program_options::parsed_options parsed = boost::program_options::command_line_parser(argc, argv).options(
      desc).allow_unregistered().run();

    boost::program_options::store(parsed, vm);
    boost::program_options::notify(vm);
    arg.path_file_to_load = vm["path"].as<std::string>();
    arg.indexes = vm["index"].as<std::vector<size_t>>();
    arg.name_video = vm["video"].as<std::string>();
    arg.use_archive_dat = vm.count("archive-dat");
  }
}

namespace aurora {
  template<typename Fit>
  struct VideoSaver
  {
    template<typename EA>
    void
    save_video(const std::string& path_file_to_load,
               int index,
               const std::string& name_video,
               EA& ea,
               bool use_archive_dat_file=false)
    {}
  };



  template<typename Params, typename Exact>
  struct VideoSaver<aurora::env::FitHexapod<Params, Exact>>
  {
    template <typename EA>
    void visualise_behaviour_dat_file(const std::string &path_archive_individuals,
                             size_t index_chosen_behaviour,
                             const std::string& name_video) {
      Interactive_map<Params> data_loader;
      archive_content_t archive_all_individuals;
      data_loader.load_archive(path_archive_individuals, archive_all_individuals);
      const controllers_archive_t &controllers_to_show = std::get<3>(archive_all_individuals);

      constexpr bool save_single_video = false;

      std::vector<double> controller = controllers_to_show[index_chosen_behaviour];

      Eigen::VectorXd ctrl(controller.size());
      for (size_t index_controller = 0; index_controller < controller.size(); ++index_controller) {
        ctrl(index_controller) = std::round(controller[index_controller] * 1000.0) / 1000.0;
      }

      std::cout << "showing index -> " << index_chosen_behaviour << std::endl;

      const std::string name_video_chosen_behaviour = name_video + '-' + std::to_string(index_chosen_behaviour) + ".mp4";

      typename EA::fit_t fit;
      std::vector<double> ctrl_v(ctrl.data(), ctrl.data() + ctrl.size());
      fit.set_ctrl(ctrl_v);
      fit.simulate_with_video(name_video_chosen_behaviour);
    }

    template<typename EA>
    void
    save_video(const std::string& path_file_to_load,
               int index,
               const std::string& name_video,
               EA& ea,
               bool use_archive_dat_file=false)
    {
      if (not use_archive_dat_file) {
        typename EA::pop_t pop = ea.get_pop_from_gen_file(path_file_to_load);
        pop[index]->fit().simulate_with_video(name_video);
      } else {
        this->visualise_behaviour_dat_file<EA>(path_file_to_load, index, name_video);
      }
    }
  };

  template<typename Params, typename Exact>
  struct VideoSaver<aurora::env::AirHockey<Params, Exact>>
  {
    template<typename EA>
    void
    save_video(const std::string& path_file_to_load,
               int index,
               const std::string& name_video,
               EA& ea,
               bool use_archive_dat_file=false)
    {
      typename EA::pop_t pop = ea.get_pop_from_gen_file(path_file_to_load);
//      pop[index]->fit().simulate_with_video(name_video);
      pop[index]->develop();

      Eigen::VectorXd angle, angle_post;
      std::tie(angle, angle_post) = pop[index]->fit().initialise_ctrl(*pop[index]);
      pop[index]->fit().simulate_with_video(angle, angle_post, name_video);
    }
  };

}

int main(int argc, char **argv) {
  boost::program_options::options_description desc;
  desc.add_options()("path", boost::program_options::value<std::string>(), "Set path of proj file (serialised gen file)");
  desc.add_options()("index", boost::program_options::value<std::vector<size_t>>()->multitoken(), "List of Indexes of behaviour to save in that file");
  desc.add_options()("video", boost::program_options::value<std::string>(), "Name video to save without the extension");
  desc.add_options()("archive-dat", boost::program_options::bool_switch()->default_value(false), "Using archive_XXX.dat file ? (don't use this option if you resume from a gen_XXX file)");

  visu::Arguments arg{};
  visu::get_arguments(desc, arg, argc, argv);

  srand(time(0));

  tbb::task_scheduler_init init(1);
  robox2d::gui::magnum::GlobalData::instance()->set_max_contexts(2);



  constexpr aurora::env::Env environment = aurora::get_env();
  constexpr aurora::algo::Algo algorithm = aurora::get_algo();

  typedef aurora::SpecificParams specific_params_t;
  typedef aurora::DefaultParamsFactory<environment,
    specific_params_t>::default_params_t default_params_t;
  typedef aurora::ParamsAlgo<algorithm, default_params_t, specific_params_t> params_t;

  params_t::step_measures = 10;
  params_t::nov::use_fixed_l = specific_params_t::use_fixed_l;

  aurora::EnvironmentInitialiser<default_params_t> environment_initialiser;

  constexpr bool use_meshes = false;
  environment_initialiser.run(use_meshes);

  typedef aurora::env::Environment<environment, params_t> env_t;
  typedef aurora::algo::AlgorithmFactory<algorithm, env_t>::algo_t algo_t;

  aurora::algo::initialise_global_variables<algo_t::param_t>();

  algo_t::update_parameters();
  typedef algo_t::ea_t ea_t;
  ea_t ea;

  typedef boost::shared_ptr<typename env_t::phen_t> indiv_t;
  typedef std::vector<indiv_t> pop_t;

  ea.set_fit_proto(env_t::phen_t::fit_t());

  aurora::VideoSaver<env_t::phen_t::fit_t> video_saver;

  for (size_t index: arg.indexes) {
    const std::string name_video = arg.name_video + '-' + std::to_string(index) + ".mp4";
    video_saver.save_video(arg.path_file_to_load, index, name_video, ea, arg.use_archive_dat);
  }

  return 0;
}


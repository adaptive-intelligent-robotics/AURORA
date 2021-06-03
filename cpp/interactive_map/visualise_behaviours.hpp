//
// Created by Luca Grillotti on 08/02/2020.
//

#ifndef AURORA_TEST_HPP
#define AURORA_TEST_HPP

#include <Eigen/Core>

#include "interactive_map/interactive_map.hpp"

namespace aurora {
    typedef Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> Mat;

    template<typename fit_t, typename param_t>
    void visualise_behaviour(const std::string &path_archive_individuals,
                             const std::string &path_stat_projection) {
        Interactive_map<param_t> imap;

        archive_content_t archive_all_individuals;
        archive_content_t archive_chosen_individuals;

        imap.load_archive(path_archive_individuals, archive_all_individuals);

        imap.visualise_map(path_archive_individuals, path_stat_projection, archive_chosen_individuals,
                           archive_all_individuals);

        const controllers_archive_t &controllers_to_show = std::get<3>(archive_chosen_individuals);

        for (const auto &controller : controllers_to_show) {
            std::stringstream ss;
            std::vector<double> ctrl;
            for (double d: controller) {
                ss << d << " ";
                ctrl.push_back(std::round(d * 100.0) / 100.0);
            }
            std::cout << ctrl.size() << std::endl;
            std::cout << std::endl;
            std::cout << "Controller: " << ss.str() << std::endl;
            fit_t fit;

            fit.simulate(ctrl);
        }
    }

    template<typename ea_t, typename fit_t, typename param_t>
    void visualise_one_behaviour(
            ea_t& ea,
            const std::string &path_archive_individuals,
            size_t index_chosen_behaviour) {
        Interactive_map<param_t> imap;

        archive_content_t archive_all_individuals;
//        ea.load(path_archive_individuals);
//        auto ind = ea.pop()[index_chosen_behaviour];
//        std::vector<float> ctrl = ind->fit().data();

        imap.load_archive(path_archive_individuals, archive_all_individuals);
        const controllers_archive_t &controllers_to_show = std::get<3>(archive_all_individuals);
        std::vector<double> controller = controllers_to_show[index_chosen_behaviour];

        std::stringstream ss;
        std::vector<double> ctrl;
        for (double d: controller) {
            ss << d << " ";
            ctrl.push_back(std::round(d * 1000.0) / 1000.0);
        }
        std::cout << ctrl.size() << std::endl;
        std::cout << std::endl;
        std::cout << "Controller: " << ss.str() << std::endl;
        for (double & i : ctrl) {
            std::cout << i << ' ';
        }
        std::cout << '\n';

        fit_t fit;
        for (size_t i = 0; i < 10; ++i) {
          std::cout << i << std::endl;
            fit.simulate_visualisations(ctrl);
        }
    }

    template<typename ea_t, typename fit_t, typename param_t>
    void
    save_video_one_behaviour(ea_t& ea,
                             const std::string& path_archive_individuals,
                             size_t index_chosen_behaviour,
                             const std::string& name_video)
    {
      Interactive_map<param_t> imap;

      archive_content_t archive_all_individuals;
      //        ea.load(path_archive_individuals);
      //        auto ind = ea.pop()[index_chosen_behaviour];
      //        std::vector<float> ctrl = ind->fit().data();

      imap.load_archive(path_archive_individuals, archive_all_individuals);
      const controllers_archive_t& controllers_to_show = std::get<3>(archive_all_individuals);
      std::vector<double> controller = controllers_to_show[index_chosen_behaviour];

      std::stringstream ss;
      std::vector<double> ctrl;
      for (double d : controller) {
        ss << d << " ";
        ctrl.push_back(std::round(d * 1000.0) / 1000.0);
      }
      std::cout << ctrl.size() << std::endl;
      std::cout << std::endl;
      std::cout << "Controller: " << ss.str() << std::endl;
      for (double& i : ctrl) {
        std::cout << i << ' ';
      }
      std::cout << '\n';

      fit_t fit;
      fit.set_ctrl(ctrl);
      fit.simulate_with_video(name_video);
    }
}

#endif //AURORA_TEST_HPP

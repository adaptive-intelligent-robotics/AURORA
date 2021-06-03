//
// Created by Luca Grillotti on 12/05/2020.
//

#ifndef AURORA_FIT_HEXAPOD_UTILS_HPP
#define AURORA_FIT_HEXAPOD_UTILS_HPP

#include <iostream>
#include <Eigen/Core>
#include <robot_dart/robot.hpp>

namespace aurora {
    namespace env {
        // amounts to use of each RGB channel when converting to grayscale
        constexpr float R = 0.5;
        constexpr float G = 0.85;
        constexpr float B = 0.15;


        namespace global {
            std::shared_ptr<robot_dart::Robot> global_robot;
        }

        typedef Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> Mat;

        void load_and_init_robot(bool use_meshes) {
            std::cout << "INIT Robot" << std::endl;

            if (not use_meshes) {
              global::global_robot = std::make_shared<robot_dart::Robot>("exp/aurora/resources/hexapod_v2.urdf");
            } else {
              std::vector<std::pair<std::string, std::string>> packages = {{"hexapod_v2_description", "exp/aurora/resources/robots/hexapod_v2_description"}};
              global::global_robot = std::make_shared<robot_dart::Robot>("exp/aurora/resources/robots/hexapod_v2_with_meshes.urdf", packages);
            }
            global::global_robot->set_position_enforced(true);

            global::global_robot->set_actuator_types("servo");
//            global::global_robot->skeleton()->enableSelfCollisionCheck();
            std::cout << "End init Robot" << std::endl;
        }
    }
}

#endif //AURORA_FIT_HEXAPOD_UTILS_HPP

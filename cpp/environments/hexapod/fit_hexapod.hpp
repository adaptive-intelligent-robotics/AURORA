#ifndef ___FIT_HEXA_HPP__
#define ___FIT_HEXA_HPP__

#include <iostream>
#include <robot_dart/robot_dart_simu.hpp>

#include <robot_dart/gui/helper.hpp>
#include <robot_dart/gui/magnum/sensor/camera.hpp>
#include <robot_dart/gui/magnum/graphics.hpp>
#include <robot_dart/gui/magnum/glfw_application.hpp>

#include <robot_dart/gui/magnum/windowless_graphics.hpp>
#include <robot_dart/gui/magnum/windowless_gl_application.hpp>


#include <dart/collision/bullet/BulletCollisionDetector.hpp>
#include <dart/constraint/ConstraintSolver.hpp>

#include <ctime>
#include <cmath>
#include <cstdint>

#include "environments/hexapod/fit_hexapod_utils.hpp"
#include "environments/hexapod/fit_hexapod_abstract.hpp"
#include "environments/hexapod/control/hexa_control.hpp"


namespace aurora {
    namespace env {

        SFERES_FITNESS (FitHexapod, FitHexapodExperiments)
        {
        public:
            FitHexapod() : SFERES_PARENT(FitHexapod, FitHexapodExperiments)() {}

            typedef Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> Mat;

            std::shared_ptr<robot_dart::Robot> generate_and_configure_robot() {
                std::shared_ptr<robot_dart::Robot> g_robot = global::global_robot->clone();

                g_robot->skeleton()->setPosition(5, 0.15);

                Eigen::Map<Eigen::VectorXd> ctrl_vector(this->_ctrl.data(), this->_ctrl.size());
                constexpr double ctrl_dt = 0.015;
                g_robot->add_controller(std::make_shared<robot_dart::control::HexaControl>(ctrl_dt, ctrl_vector));
                Eigen::VectorXd h_params(1);
                h_params << ctrl_dt;
                std::static_pointer_cast<robot_dart::control::HexaControl>(g_robot->controllers()[0])->set_h_params(h_params);
                return g_robot;
            }

            void create_environment(const std::shared_ptr<robot_dart::RobotDARTSimu> &simu,
                                    const std::shared_ptr<robot_dart::Robot> &robot,
                                    bool use_video_settings = false) {
              // std::cout << "Start - env" << std::endl;
                robot_dart::gui::magnum::GraphicsConfiguration graphics_configuration = robot_dart::gui::magnum::WindowlessGraphics::default_configuration();

                if (not use_video_settings) {
                  graphics_configuration.width = Params::image_width * Params::times_downsample;
                  graphics_configuration.height = Params::image_height * Params::times_downsample;
                  graphics_configuration.shadowed = false;
                  graphics_configuration.transparent_shadows = false;
                  graphics_configuration.draw_debug = false;
                } else {
                  graphics_configuration.width = 1280;
                  graphics_configuration.height = 720;
                  graphics_configuration.shadowed = true;
                  graphics_configuration.transparent_shadows = true;
                  graphics_configuration.draw_debug = false;
                  graphics_configuration.bg_color = Eigen::Vector4d{0.52941, 0.8078433, 0.9215686, 1.};
                }


                auto graphics = std::make_shared<robot_dart::gui::magnum::WindowlessGraphics>(
                        graphics_configuration);

                simu->set_graphics(graphics);
                graphics->set_enable(false);

                if (not use_video_settings) {
                    graphics->look_at({0.0, 0.05, 2.0}, {0.0, 0.0, 0.25});
                } else {
                    graphics->look_at({0.0, 2., 0.55}, {0.0, 0.0, 0.2});
                }

                simu->world()->getConstraintSolver()->setCollisionDetector(
                        dart::collision::BulletCollisionDetector::create());
                // std::cout << "Coucou 4" << std::endl;
                if (not use_video_settings) {
                    simu->add_floor();
                } else {
                    simu->add_checkerboard_floor();
                }
                // std::cout << "Coucou 5" << std::endl;
                //            simu->add_checkerboard_floor(10.);

                simu->add_robot(robot);
                // std::cout << "End - env" << std::endl;
            }

            void execute_simu(const std::shared_ptr<robot_dart::RobotDARTSimu> &simu) {
                float simulation_length = 3.f;
                // std::cout << "Start" << std::endl;
                simu->run(simulation_length);
                // std::cout << "End" << std::endl;
                // take picture here and copy to private member variable _image

                // cam1->set_recording(true); // to be deleted
                // cam1->set_filename("exp/camera_experiment/saved_images/test_" + std::to_string(rand() % 10) + ".png");

                // cam1->set_enable(true);


                auto graphics = std::static_pointer_cast<robot_dart::gui::magnum::WindowlessGraphics>(simu->graphics());
                // graphics->set_recording(true);
                graphics->magnum_app()->render();
                this->m_rgb_image = simu->graphics()->image();
                // std::cout << "Image Saved" << std::endl;
            }

            void execute_simu_with_video(const std::shared_ptr<robot_dart::RobotDARTSimu>& simu,
                                         const std::string& video_name) {
              float simulation_length = 3.f;
              // std::cout << "Start" << std::endl;
              auto graphics = std::static_pointer_cast<robot_dart::gui::magnum::WindowlessGraphics>(simu->graphics());
              graphics->set_enable(true);
              graphics->record_video(video_name, 30);

              simu->run(simulation_length);

              // Looking at position close to robot
              Eigen::VectorXd init_trans = simu->robots().back()->skeleton()->getPositions();

              // graphics->look_at({init_trans[3], init_trans[4] + 0.8, 0.55},
                                // {init_trans[3], init_trans[4], 0.2});

              graphics->magnum_app()->render();
              this->m_rgb_image = simu->graphics()->image();
              const std::string image_name = video_name + ".png";
              robot_dart::gui::save_png_image(image_name, this->m_rgb_image);


              std::cout << "recorded video: " << video_name << std::endl;
              std::cout << "saved image: " << image_name << std::endl;
            }
        };

    }
}
#endif

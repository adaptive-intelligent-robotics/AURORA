//
// Created by Luca Grillotti on 12/05/2020.
//

#ifndef AURORA_FIT_HEXAPOD_ABSTRACT_HPP
#define AURORA_FIT_HEXAPOD_ABSTRACT_HPP

#include <iostream>
#include <Eigen/Core>

#include <robot_dart/robot_dart_simu.hpp>

#include <robot_dart/gui/helper.hpp>
#include <robot_dart/gui/magnum/sensor/camera.hpp>
#include <robot_dart/gui/magnum/graphics.hpp>
#include <robot_dart/gui/magnum/glfw_application.hpp>
#include <robot_dart/gui/magnum/windowless_graphics.hpp>
#include <robot_dart/gui/magnum/windowless_gl_application.hpp>


#include <dart/collision/bullet/BulletCollisionDetector.hpp>
#include <dart/constraint/ConstraintSolver.hpp>

#include <sferes/fit/fit_qd.hpp>

#include "environments/hexapod/control/hexa_control.hpp"
#include "fit_hexapod_utils.hpp"
#include "environments/image_utils.hpp"


namespace aurora {
    namespace env {

      double angle_dist(double a, double b)
      {
        double theta = b - a;
        while (theta < -M_PI)
          theta += 2 * M_PI;
        while (theta > M_PI)
          theta -= 2 * M_PI;
        return theta;
      }


      FIT_QD(FitHexapodExperiments)
        {
        public:
            FitHexapodExperiments() : _entropy(-1), _implicit_fitness_value(-1) {}

            typedef Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> Mat;



            template<typename Indiv>
            void eval(Indiv &ind) {
                initialise_ctrl(ind);
                _implicit_fitness_value = static_cast<float>(simulate());
                if (aurora::get_has_fit()) {
                    this->_value = _implicit_fitness_value;
                  } else {
                    this->_value = -1; // FITNESS: constant because we're interested in exploration
                }

                if (aurora::is_algo_hand_coded()) {
                    set_descriptor_novelty_search(ind);
                }
            }

            template<typename Indiv>
            void initialise_ctrl(Indiv &ind) {
                _ctrl.resize(36);

                for (size_t i = 0; i < _ctrl.size(); i++) {
                    _ctrl[i] = round(ind.data(i) * 1000.0) / 1000.0;// limit numerical issues
                }
            }

            double simulate() {
                Corrade::Utility::Error magnum_silence_error{nullptr};
                get_gl_context_with_sleep(gl_context,
                                          20); // this call will sleep 20ms between each failed query

                std::shared_ptr<robot_dart::Robot> g_robot = generate_and_configure_robot();

                std::shared_ptr<robot_dart::RobotDARTSimu> simu{new robot_dart::RobotDARTSimu(0.03)};
                create_environment(simu, g_robot);

                Eigen::VectorXd init_trans = simu->robots().back()->skeleton()->getPositions();

                execute_simu(simu); // generates m_rgb_image

                convert_to_vector(m_rgb_image, _image);

                update_ground_truth(simu);

                g_robot.reset();
                //            cam1.reset();
                release_gl_context(gl_context);

                Eigen::VectorXd pose = simu->robots().back()->skeleton()->getPositions();
                double fitness_value = -1 * orientation_error(pose, init_trans);
                return fitness_value;
            }

            double simulate_with_video(const std::string& video_name)
            {

              std::shared_ptr<robot_dart::Robot> g_robot = generate_and_configure_robot();

              std::shared_ptr<robot_dart::RobotDARTSimu> simu{ new robot_dart::RobotDARTSimu(0.03) };
              create_environment(simu, g_robot, true);

              Eigen::VectorXd init_trans = simu->robots().back()->skeleton()->getPositions();

              execute_simu_with_video(simu, video_name); // generates m_rgb_image

              convert_to_vector(m_rgb_image, _image);

              update_ground_truth(simu);

              g_robot.reset();
              //            cam1.reset();

              Eigen::VectorXd pose = simu->robots().back()->skeleton()->getPositions();
              double fitness_value = -1 * orientation_error(pose, init_trans);
              return fitness_value;
            }

            double orientation_error(const Eigen::VectorXd& pose, const Eigen::VectorXd& init_trans) const
          {
            Eigen::Vector3d final_pos;
            Eigen::Vector3d final_rot;
            double arrival_angle;
            double covered_distance;

            Eigen::Matrix3d rot = dart::math::expMapRot({pose[0], pose[1], pose[2]});
            Eigen::Matrix3d init_rot = dart::math::expMapRot({init_trans[0], init_trans[1], init_trans[2]});
            Eigen::MatrixXd init_homogeneous(4, 4);
            init_homogeneous << init_rot(0, 0), init_rot(0, 1), init_rot(0, 2), init_trans[3], init_rot(1, 0), init_rot(1, 1), init_rot(1, 2), init_trans[4], init_rot(2, 0), init_rot(2, 1), init_rot(2, 2), init_trans[5], 0, 0, 0, 1;
            Eigen::MatrixXd final_homogeneous(4, 4);
            final_homogeneous << rot(0, 0), rot(0, 1), rot(0, 2), pose[3], rot(1, 0), rot(1, 1), rot(1, 2), pose[4], rot(2, 0), rot(2, 1), rot(2, 2), pose[5], 0, 0, 0, 1;
            Eigen::Vector4d pos = {init_trans[3], init_trans[4], init_trans[5], 1.0};
            pos = init_homogeneous.inverse() * final_homogeneous * pos;

            final_pos = pos.head(3);

            covered_distance = std::round(final_pos(0) * 100) / 100.0;

            // Angle computation
            final_rot = dart::math::matrixToEulerXYZ(init_rot.inverse() * rot);

            // roll-pitch-yaw
            arrival_angle = std::round(final_rot(2) * 100) / 100.0;


            // Performance - Angle Difference (desrird angle and obtained angle fomr simulation)
            // Change of orientation of axis in counting the desried angle to account for frontal axis of the newer robot (x-axis:frontal axis)
            double x = final_pos[0] ;
            double y = final_pos[1] ;

            // Computation of desired angle (yaxis-north x-axis(postive))
            double B = std::sqrt((0.25 * x * x) + ( 0.25 * y * y));
            double alpha = std::atan2(y, x);
            double A = B / std::cos(alpha);
            double beta = std::atan2(y, x - A);

            if (x < 0)
              beta = beta - M_PI;
            while (beta < -M_PI)
              beta += 2 * M_PI;
            while (beta > M_PI)
              beta -= 2 * M_PI;

            double angle_diff = std::abs(angle_dist(beta, arrival_angle)); //angle dist was a finction made earlier up in the script

            return angle_diff;
          }


          std::shared_ptr<robot_dart::Robot> generate_and_configure_robot() {
                return stc::exact(this)->generate_and_configure_robot();
            }

            void create_environment(const std::shared_ptr<robot_dart::RobotDARTSimu> &simu,
                                    const std::shared_ptr<robot_dart::Robot> &robot,
                                    bool use_video_settings = false) {
                stc::exact(this)->create_environment(simu, robot, use_video_settings);
            }

            void execute_simu(const std::shared_ptr<robot_dart::RobotDARTSimu> &simu) {
                stc::exact(this)->execute_simu(simu);
            }

            void execute_simu_with_video(const std::shared_ptr<robot_dart::RobotDARTSimu> &simu,
                                         const std::string &video_name) {
              stc::exact(this)->execute_simu_with_video(simu, video_name);
            }

            void update_ground_truth(const std::shared_ptr<robot_dart::RobotDARTSimu> &simu) {
              /*
               * As we have walls of different colors in every direction,
               * we are here interested in calculating all the different Euler angles
               */
              // getting the rotation angle
              dart::dynamics::BodyNode* base_link_body_node = simu->robots().back()->body_node("base_link");
              Eigen::Matrix3d rot = base_link_body_node->getTransform().linear();
              // Eigen::Matrix3d rot = simu->robots().back()->body_rot("base_link");
              auto final_pos = simu->robots().back()->skeleton()->getPositions().head(6).tail(3).cast<float>();


              // We start by adding the x,y,z coord
              this->_gt.push_back(final_pos[0]);
              this->_gt.push_back(final_pos[1]);
              this->_gt.push_back(final_pos[2]);


              // Then Rotation coordinates
              Eigen::Vector3d forward_direction = rot.col(0);
              Eigen::Vector3d left_direction = rot.col(1);
              Eigen::Vector3d up_direction = rot.col(2);

              Eigen::Vector3d u_z{0., 0., 1.};

              // u_r and u_theta -> Cylindrical coordinate system
              Eigen::Vector3d u_r{forward_direction(0), forward_direction(1), 0.};
              u_r = u_r.normalized();
              Eigen::Vector3d u_theta = u_z.cross(u_r);

              // Get absolute values of angles

              auto abs_pitch_angle = static_cast<float>(
                acos(forward_direction.dot(u_z)));
              auto abs_roll_angle = static_cast<float>(
                acos(u_theta.dot(left_direction)));
              auto abs_yaw_angle = static_cast<float>(
                acos(u_r.dot(Eigen::Vector3d(1., 0., 0.)))
              );

              // Get values of angles depending on the direction of the vector

              float pitch_angle;
              float roll_angle;
              float yaw_angle;


              if (u_z.dot(up_direction) > 0.) {
                pitch_angle = abs_pitch_angle;
              } else {
                pitch_angle = -1.f * abs_pitch_angle;
              }

              if (u_theta.dot(up_direction) < 0.) {
                roll_angle = abs_roll_angle;
              } else {
                roll_angle = -1.f * abs_roll_angle;
              }

              if (u_r.dot(Eigen::Vector3d(0., 1., 0.)) > 0) {
                yaw_angle = abs_yaw_angle;
              } else {
                yaw_angle = -1.f * abs_yaw_angle;
              }

              this->_gt.push_back(pitch_angle);
              this->_gt.push_back(roll_angle);
              this->_gt.push_back(yaw_angle);
            }

            template<typename Indiv>
            void set_descriptor_novelty_search(Indiv &ind) {
              if (Params::env == aurora::env::Env::hexa_cam_vertical) {
                std::vector<double> gt_double(this->_gt.begin(), this->_gt.begin() + 2);  // x,y position
                this->set_desc(gt_double);
              } else if (Params::env == aurora::env::Env::hexa_cam_vert_hc_pix) {
                std::vector<double> img_double(this->_image.begin(), this->_image.end());
                this->set_desc(img_double);
              } else if (Params::env == aurora::env::Env::hexa_gen_desc) {
                std::vector<double> gen_double(ind.gen().data().begin(), ind.gen().data().end());
                this->set_desc(gen_double);
              }
            }

            // converts RGB OSG Image to grayscale vector of floats (0 to 1)
            void convert_to_vector(const robot_dart::gui::ImageSerialisable &image, std::vector<float> &img_float) const {
                if (Params::use_colors) {
                    convert_to_rgb_vector(image, img_float);
                } else {
                    convert_to_grayscale_vector(image, img_float);
                }
            }

            void convert_to_grayscale_vector(const robot_dart::gui::ImageSerialisable &image, std::vector<float> &img_float) const {
                const std::vector<uint8_t>& img_uint8 = image.data;

                float pixel;

                for (size_t i = 0; i < img_uint8.size(); i += 3) {
                    pixel = R * (float) img_uint8[i] + G * (float) img_uint8[i + 1] +
                            B * (float) img_uint8[i + 2];
                    img_float.push_back(pixel / 255.0);
                }

                for (int k = Params::times_downsample; k > 1; k /= 2) {
                    downsample(img_float, k);
                }
            }

            // converts RGB OSG Image to grayscale vector of floats (0 to 1)
            void convert_to_rgb_vector(const robot_dart::gui::ImageSerialisable &image,
                                       std::vector<float> &img_float
            ) const {
                std::vector<float> img_red_float;
                std::vector<float> img_green_float;
                std::vector<float> img_blue_float;

                std::vector<uint8_t> img_red_uint8;
                std::vector<uint8_t> img_green_uint8;
                std::vector<uint8_t> img_blue_uint8;

                size_t height = image.height;
                size_t width = image.width;
                size_t nb_channels = image.channels;

                for (size_t h = 0; h < height; h++) {
                    for (size_t w = 0; w < width; w++) {
                        int id_rgb = w * nb_channels + h * (width * nb_channels);

                        img_red_uint8.push_back(image.data[id_rgb + 0]);
                        img_green_uint8.push_back(image.data[id_rgb + 1]);
                        img_blue_uint8.push_back(image.data[id_rgb + 2]);
                    }
                }

                for (size_t i = 0; i < img_red_uint8.size(); ++i) {
                    img_red_float.push_back(img_red_uint8[i] / 255.0);
                    img_green_float.push_back(img_green_uint8[i] / 255.0);
                    img_blue_float.push_back(img_blue_uint8[i] / 255.0);
                }

                for (int k = Params::times_downsample; k > 1; k /= 2) {
                    downsample(img_red_float, k);
                    downsample(img_green_float, k);
                    downsample(img_blue_float, k);
                }

                img_float.insert(img_float.end(), img_red_float.begin(), img_red_float.end());
                img_float.insert(img_float.end(), img_green_float.begin(), img_green_float.end());
                img_float.insert(img_float.end(), img_blue_float.begin(), img_blue_float.end());
            }

            void downsample(std::vector<float> &img, int k) const {
                // std::cout << "downsample: img size " << img.size() << "k: " << k << std::endl;
                int image_height = Params::image_height * k;
                int image_width = Params::image_width * k;

                Mat image;
                image.resize(image_height, image_width);

                // std::cout << "image in matrix form resized: " << image.rows() << "x" << image.cols() << std::endl;

                size_t count = 0;
                for (int i = 0; i < image_height; i++) {
                    for (int j = 0; j < image_width; j++) {
                        image(i, j) = img[count++];
                    }
                }

                img.clear();

                for (int i = 0; i < image_height; i += 2) {
                    for (int j = 0; j < image_width; j += 2) {
                        // Using Max Pooling to reduce dimensionnality
                        img.push_back(std::max(
                                {image(i, j), image(i + 1, j), image(i, j + 1), image(i + 1, j + 1)}));
                        //img.push_back((image(i, j) + image(i + 1, j) + image(i, j + 1) + image(i + 1, j + 1)) / 4); // TODO: update this
                    }
                }
            }

            float &entropy() { return _entropy; }
            const float &entropy() const { return _entropy; }

            float &implicit_fitness_value() { return _implicit_fitness_value; }
            const float &implicit_fitness_value() const { return _implicit_fitness_value; }

            const std::vector<float> &observations() const {
                return _image;
            }

            template<typename block_t>
            void get_flat_observations(block_t &data) const {
                // std::cout << _image.size() << std::endl;
                for (size_t i = 0; i < _image.size(); i++) {
                    data(0, i) = _image[i];
                }
            }

            size_t get_flat_obs_size() const {
                assert(observations().size());
                return observations().size();
            }

            std::vector<float> &gt() { return _gt; }

            const std::vector<float> &successive_gt() const {
                return m_successive_gt;
            }

            robot_dart::gui::ImageSerialisable &get_rgb_image() { return m_rgb_image; }

            void set_ctrl(const std::vector<double>& ctrl) {
              assert(ctrl.size() == 36);
              _ctrl.resize(36);

              for (size_t i = 0; i < _ctrl.size(); i++) {
                _ctrl[i] = round(ctrl[i] * 1000.0) / 1000.0;// limit numerical issues
              }
            }

            // Serialization
            template<class Archive>
            void serialize(Archive & ar, const unsigned int version)
            {
              ar& BOOST_SERIALIZATION_NVP(this->_objs);
              ar& BOOST_SERIALIZATION_NVP(this->_value);
              ar& BOOST_SERIALIZATION_NVP(this->_dead);
              ar& BOOST_SERIALIZATION_NVP(this->_desc);
              ar& BOOST_SERIALIZATION_NVP(this->_novelty);
              ar& BOOST_SERIALIZATION_NVP(this->_curiosity);
              ar& BOOST_SERIALIZATION_NVP(this->_lq);

              ar& BOOST_SERIALIZATION_NVP(this->_ctrl);
              ar& BOOST_SERIALIZATION_NVP(this->_entropy);
              ar& BOOST_SERIALIZATION_NVP(this->_implicit_fitness_value);
              ar& BOOST_SERIALIZATION_NVP(this->_image);
              ar& BOOST_SERIALIZATION_NVP(this->m_rgb_image);
              ar& BOOST_SERIALIZATION_NVP(this->_gt);
              ar& BOOST_SERIALIZATION_NVP(this->m_successive_gt);
            }

          protected:
            std::vector<double> _ctrl;
            float _entropy;
            float _implicit_fitness_value;
            std::vector<float> _image;
            robot_dart::gui::ImageSerialisable m_rgb_image;

            std::vector<float> _gt;
            std::vector<float> m_successive_gt; // TODO: Fill it
        };
    }
}


#endif //AURORA_FIT_HEXAPOD_ABSTRACT_HPP

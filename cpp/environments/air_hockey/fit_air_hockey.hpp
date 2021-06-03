//
// Created by Luca Grillotti on 22/09/2020.
//

#ifndef AURORA_FIT_AIR_HOCKEY_HPP
#define AURORA_FIT_AIR_HOCKEY_HPP

#include <iostream>
#include <Eigen/Core>

#include <box2d/box2d.h>

#include <sferes/fit/fit_qd.hpp>


#include <robox2d/simu.hpp>
#include <robox2d/robot.hpp>
#include <robox2d/common.hpp>
#include <robox2d/gui/magnum/graphics.hpp>
#include <robox2d/gui/magnum/glfw_application.hpp>
#include <robox2d/gui/magnum/base_application.hpp>
#include <robox2d/gui/magnum/windowless_gl_application.hpp>
#include <robox2d/gui/magnum/windowless_graphics.hpp>

#include "environments/image_utils.hpp"
#include "environments/air_hockey/arm_robot.hpp"
#include "environments/air_hockey/arm_descriptor.hpp"

namespace aurora {
  namespace env {

    FIT_QD(AirHockey) {
    public:
      using rgb_t = robox2d::gui::magnum::GraphicsConfiguration::rgb_t;

      AirHockey() : m_entropy(-1), m_implicit_fitness_value(-1) {}

      typedef Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> Mat;

      template<typename Indiv>
      void eval(Indiv &ind) {
        Eigen::VectorXd angle, angle_post;
        std::tie(angle, angle_post) = initialise_ctrl(ind);

        m_implicit_fitness_value = static_cast<float>(simulate(angle, angle_post));
        this->generate_observation();

        if (aurora::get_has_fit()) {
          this->_value = m_implicit_fitness_value;
        } else {
          this->_value = -1; // FITNESS: constant because we're interested in exploration
        }

        if (aurora::is_algo_hand_coded()) {
          set_descriptor_novelty_search(ind);
        }
      }

      void generate_observation()
      {
        if (aurora::does_encode_images()) {
          this->save_vector_image_float(m_rgb_image);
        } else if (aurora::does_encode_sequence()
                   or aurora::is_algo_hand_coded()) {

          std::vector<float> successive_ball_positions_float(m_successive_ball_positions.begin(),
                                                             m_successive_ball_positions.end());
          m_image_float = successive_ball_positions_float;
        }
      }

      template<typename Indiv>
      std::tuple<Eigen::VectorXd, Eigen::VectorXd> initialise_ctrl(Indiv &ind) {
        size_t nb_dofs= ind.size() / 2;
        Eigen::VectorXd angle(nb_dofs);
        Eigen::VectorXd angle_post(nb_dofs);
        assert(ind.size() % 2 == 0);
        for (size_t i = 0; i < ind.size() / 2; ++i) {
          angle[i] = ind.data(i);
        }
        for (size_t i = 0; i < ind.size() / 2; ++i) {
          angle_post[i] = ind.data(i + ind.size() / 2);
        }

        return {angle, angle_post};
      }

      double simulate(const Eigen::VectorXd& ctrl_pos, const Eigen::VectorXd& ctrl_pos_post) {
        Corrade::Utility::Error magnum_silence_error{nullptr};
        get_gl_context_with_sleep_robox2d(gl_context, 20);

        robox2d::Simu* simu = new robox2d::Simu();
        simu->add_floor();

        constexpr int size_image = 256;

        auto rob = std::make_shared<robox2d::Arm>(simu->world(), false);
        auto ctrl = std::make_shared<robox2d::control::ConstantPos>(ctrl_pos);
        rob->add_controller(ctrl);
        simu->add_robot(rob);
        simu->add_descriptor<robox2d::descriptor::ArmDescriptor>();

        // TODO: Modify Size_image to make it fit what we do with hexapod

        robox2d::gui::magnum::GraphicsConfiguration graphics_configuration = robox2d::gui::magnum::WindowlessGraphics::default_configuration();
        graphics_configuration.width = Params::image_width * Params::times_downsample;
        graphics_configuration.height = Params::image_width * Params::times_downsample;


        auto graphics = std::make_shared<robox2d::gui::magnum::WindowlessGraphics>(graphics_configuration);

        simu->set_graphics(graphics);
        graphics->set_enable(false);


        simu->run(5.0);

        simu->robot(0)->remove_controller(0);
        simu->robot(0)->add_controller(std::make_shared<robox2d::control::ConstantPos>(ctrl_pos_post));
        // graphics->refresh();

        b2Body* ball = robox2d::common::createCircle(simu->world(), 0.05f, b2_dynamicBody, {0.55f, 0.5f, 0.f}, 0.2f, 0.8f, 0.8f);

        // Saving several successive positions of the ball
        m_successive_ball_positions.clear();
        m_successive_ball_positions.reserve(100);
        for (int nb_steps = 0; nb_steps < 50; ++nb_steps) {
          simu->run(0.1f);
          m_successive_ball_positions.push_back(ball->GetWorldCenter().x);
          m_successive_ball_positions.push_back(ball->GetWorldCenter().y);
        }

        // Saving Ground Truth (final position of the ball)
        m_gt.clear();
        m_gt.reserve(2);
        m_gt.push_back(m_successive_ball_positions[m_successive_ball_positions.size() - 2]);
        m_gt.push_back(m_successive_ball_positions[m_successive_ball_positions.size() - 1]);


        // graphics->set_enable(true);
        simu->set_graphics(graphics);
        graphics->magnum_app()->init(simu, graphics_configuration);
        graphics->refresh();
        // graphics->set_enable(false);
        // graphics->refresh();
        // graphics->set_recording(true);

        m_rgb_image = graphics->image();

        std::shared_ptr<robox2d::descriptor::ArmDescriptor> arm_descriptor = std::static_pointer_cast<robox2d::descriptor::ArmDescriptor>(simu->descriptor(0));

        float penalty_energy = -1.f * arm_descriptor->energy;
        delete simu;

        release_gl_context_robox2d(gl_context);

        return penalty_energy;
      }

      double simulate_with_video(const Eigen::VectorXd& ctrl_pos, const Eigen::VectorXd& ctrl_pos_post, const std::string& video_name)
      {
        // Corrade::Utility::Error magnum_silence_error{nullptr};
        // get_gl_context_with_sleep_robox2d(gl_context, 20);

        robox2d::Simu* simu = new robox2d::Simu();
        simu->add_floor();

        constexpr int size_image = 256;

        auto rob = std::make_shared<robox2d::Arm>(simu->world(), false);
        auto ctrl = std::make_shared<robox2d::control::ConstantPos>(ctrl_pos);
        rob->add_controller(ctrl);
        simu->add_robot(rob);
        simu->add_descriptor<robox2d::descriptor::ArmDescriptor>();

        robox2d::gui::magnum::GraphicsConfiguration graphics_configuration = robox2d::gui::magnum::WindowlessGraphics::default_configuration();
        graphics_configuration.width = 1024;
        graphics_configuration.height = 1024;
        graphics_configuration.bg_color = Eigen::Vector4d{1., 1., 1., 1.}; // White Background
        graphics_configuration.map_body_color = rob->get_map_body_color();

        auto graphics = std::make_shared<robox2d::gui::magnum::WindowlessGraphics>(graphics_configuration);

        simu->set_graphics(graphics);
        graphics->record_video(video_name, 30);
        // graphics->magnum_app()->init(simu, 1024, 1024);
        // graphics->refresh();
        // graphics->record_video(video_name, 30);

        for (int nb_steps = 0; nb_steps < 10; ++nb_steps) {
          simu->run(0.5f);
          robox2d::gui::Image image = graphics->image();
          robox2d::gui::save_png_image(video_name + "-" + std::to_string(nb_steps) + ".png", image);
        }

        simu->robot(0)->remove_controller(0);
        simu->robot(0)->add_controller(std::make_shared<robox2d::control::ConstantPos>(ctrl_pos_post));
        // graphics->refresh();

        b2Body* ball = robox2d::common::createCircle(simu->world(), 0.05f, b2_dynamicBody, {0.55f, 0.5f, 0.f}, 0.2f, 0.8f, 0.8f);

        std::map<b2Body*, rgb_t> map_body_color = rob->get_map_body_color();
        map_body_color.insert(std::pair<b2Body*, rgb_t>(ball, rgb_t{ 192, 108, 0 }));
        graphics_configuration.map_body_color = map_body_color;

        graphics->magnum_app()->init(simu, graphics_configuration);

        // Saving several successive positions of the ball
        m_successive_ball_positions.clear();
        m_successive_ball_positions.reserve(100);
        for (int nb_steps = 10; nb_steps < 20; ++nb_steps) {
          simu->run(0.5f);
          m_successive_ball_positions.push_back(ball->GetWorldCenter().x);
          m_successive_ball_positions.push_back(ball->GetWorldCenter().y);
          robox2d::gui::Image image = graphics->image();
          robox2d::gui::save_png_image(video_name + "-" + std::to_string(nb_steps) + ".png", image);
        }

        // Saving Ground Truth (final position of the ball)
        m_gt.clear();
        m_gt.reserve(2);
        m_gt.push_back(m_successive_ball_positions[m_successive_ball_positions.size() - 2]);
        m_gt.push_back(m_successive_ball_positions[m_successive_ball_positions.size() - 1]);


        // graphics->set_enable(true);

        // graphics->set_enable(false);
        // graphics->refresh();
        // graphics->set_recording(true);

        m_rgb_image = graphics->image();

        std::shared_ptr<robox2d::descriptor::ArmDescriptor> arm_descriptor = std::static_pointer_cast<robox2d::descriptor::ArmDescriptor>(simu->descriptor(0));

        float penalty_energy = -1.f * arm_descriptor->energy;
        delete simu;

        // release_gl_context_robox2d(gl_context);

        return penalty_energy;
      }


      std::shared_ptr<robox2d::Robot> generate_and_configure_robot() {
        // TODO
      }

      void create_environment(const std::shared_ptr<robox2d::Simu> &simu,
                              const std::shared_ptr<robox2d::Robot> &robot,
                              bool use_video_settings = false) {
        // TODO
      }

      void execute_simu(const std::shared_ptr<robox2d::Simu> &simu) {
        // TODO
      }

      void update_ground_truth(const std::shared_ptr<robox2d::Simu> &simu) {
        // TODO
      }

      template<typename Indiv>
      void set_descriptor_novelty_search(Indiv &ind) {
        if (aurora::get_env() == aurora::env::Env::air_hockey) {
          this->set_desc(m_gt);
        } else {
          this->set_desc(m_successive_ball_positions);
        }
      }

      // converts RGB OSG Image to grayscale vector of floats (0 to 1)
      void save_vector_image_float(const robox2d::gui::ImageSerialisable &image) {
        if (Params::use_colors) {
          save_vector_rgb_image_float(image);
        } else {
          save_vector_grayscale_image_float(image);
        }
      }

      void save_vector_grayscale_image_float(const robox2d::gui::ImageSerialisable &image) {
        const std::vector<uint8_t>& img_uint8 = image.data;

        float pixel;

        for (size_t i = 0; i < img_uint8.size(); i += 3) {
          pixel = R * (float) img_uint8[i] + G * (float) img_uint8[i + 1] +
                  B * (float) img_uint8[i + 2];
          m_image_float.push_back(pixel / 255.0);
        }

        for (int k = Params::times_downsample; k > 1; k /= 2) {
          downsample(m_image_float, k);
        }
      }

      // converts RGB OSG Image to grayscale vector of floats (0 to 1)
      void save_vector_rgb_image_float(const robox2d::gui::ImageSerialisable &image) {
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

        m_image_float.insert(m_image_float.end(), img_red_float.begin(), img_red_float.end());
        m_image_float.insert(m_image_float.end(), img_green_float.begin(), img_green_float.end());
        m_image_float.insert(m_image_float.end(), img_blue_float.begin(), img_blue_float.end());
      }

      /**
       * Downsampling function adapted to images in the format of robox2d
       * (imitating the old version of robot_dart)
       * @param img
       * @param k
       */
      void downsample(std::vector<float> &img, int k) const {
        int image_height = Params::image_height * k;
        int image_width = Params::image_width * k;

        Mat image;
        image.resize(image_height, image_width);

        size_t count = 0;
        for (int i = 0; i < image_height; i++) {
          for (int j = 0; j < image_width; j++) {
            image(i, j) = img[count++];
          }
        }

        img.clear();

        for (int i = 0; i < image_height; i += 2) {
          for (int j = 0; j < image_width; j += 2) {
            // Using Average Pooling to reduce dimensionality
//                        img.push_back(std::max({image(i, j), image(i + 1, j), image(i, j + 1), image(i + 1, j + 1)}));
            img.push_back((image(i, j) + image(i + 1, j) + image(i, j + 1) + image(i + 1, j + 1)) / 4);
          }
        }
      }

      float &entropy() { return m_entropy; }
      const float &entropy() const { return m_entropy; }

      float &implicit_fitness_value() { return this->m_implicit_fitness_value; }
      const float &implicit_fitness_value() const { return this->m_implicit_fitness_value; }

      const std::vector<float> &observations() const {
        return m_image_float;
      }

      template<typename block_t>
      void get_flat_observations(block_t &data) const {
        // std::cout << m_image_float.size() << std::endl;
        for (size_t i = 0; i < m_image_float.size(); i++) {
          data(0, i) = m_image_float[i];
        }
      }

      size_t get_flat_obs_size() const {
        assert(observations().size());
        return observations().size();
      }

      std::vector<double> &gt() { return m_gt; }

      const std::vector<double> &successive_gt() const {
        return m_successive_ball_positions;
      }

      robox2d::gui::ImageSerialisable &get_rgb_image() { return m_rgb_image; }

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
        ar& BOOST_SERIALIZATION_NVP(this->m_entropy);
        ar& BOOST_SERIALIZATION_NVP(this->m_implicit_fitness_value);
        ar& BOOST_SERIALIZATION_NVP(this->m_image_float);
        ar& BOOST_SERIALIZATION_NVP(this->m_rgb_image);
        ar& BOOST_SERIALIZATION_NVP(this->m_gt);
        ar& BOOST_SERIALIZATION_NVP(this->m_successive_ball_positions);
      }

    protected:
      std::vector<double> _ctrl;
      float m_entropy;
      float m_implicit_fitness_value;
      std::vector<float> m_image_float;
      robox2d::gui::ImageSerialisable m_rgb_image;

      std::vector<double> m_gt;
      std::vector<double> m_successive_ball_positions; // TODO: Fill it
    };

  }
}
#endif //AURORA_FIT_AIR_HOCKEY_HPP

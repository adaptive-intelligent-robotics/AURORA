//
// Created by Luca Grillotti on 09/02/2021.
//

#ifndef AURORA_ARM_ROBOT_HPP
#define AURORA_ARM_ROBOT_HPP

#include <robox2d/simu.hpp>
#include <robox2d/robot.hpp>
#include <robox2d/common.hpp>
#include <robox2d/gui/magnum/graphics.hpp>
#include <robox2d/gui/magnum/glfw_application.hpp>
#include <robox2d/gui/magnum/base_application.hpp>
#include <robox2d/gui/magnum/windowless_gl_application.hpp>

namespace robox2d {
  class Arm : public robox2d::Robot {
  public:
    using rgb_t = robox2d::gui::magnum::GraphicsConfiguration::rgb_t;

    Arm(std::shared_ptr<b2World> world, bool add_ball) {

      size_t nb_joints=4;
      float arm_length=0.9f;
      float seg_length = arm_length / (float) nb_joints;

      constexpr bool use_graphics = true;
      constexpr float room_w = 2.f;
      constexpr float room_h = 2.f;

      _walls_v.clear();
      _robot_bodies_v.clear();

      if (use_graphics)
      {
        b2Body* ceiling = robox2d::common::createBox(world, {room_w / 2, 0.01}, b2_staticBody, {0.f, room_h / 2.f, 0.f});
        b2Body* floor = robox2d::common::createBox(world, {room_w / 2, 0.01}, b2_staticBody, {0.f, - room_h / 2.f, 0.f});
        b2Body* right = robox2d::common::createBox(world, {0.01, room_h / 2}, b2_staticBody, {room_w / 2.f, 0.f, 0.f});
        b2Body* left = robox2d::common::createBox(world, {0.01, room_h / 2}, b2_staticBody, { - room_w / 2.f, 0.f, 0.f});

        _walls_v.insert(_walls_v.end(), {ceiling, floor, right, left});
      }
      else // if not using the GUI, create one body with 4 walls for faster sim
      {

        // b2Body* room = robox2d::common::createRoom(world, {room_w, room_h});
      }

      b2Body* body = robox2d::common::createBox( world,{arm_length*0.025f, arm_length*0.025f}, b2_staticBody,  {0.0f,0.0f,0.0f} );
      b2Vec2 anchor = body->GetWorldCenter();
      // body will always represent the body created in the previous iteration

      _robot_bodies_v.push_back(body);

      for(size_t i =0; i < nb_joints; i++)
      {
        float density = 1.0f/std::pow(1.5,i);
        _end_effector = robox2d::common::createBox( world,{seg_length*0.5f , arm_length*0.01f }, b2_dynamicBody, {(0.5f+i)*seg_length,0.0f,0.0f}, density );
        this->_actuators.push_back(std::make_shared<robox2d::actuator::Servo>(world,body, _end_effector, anchor, 0.3));

        body=_end_effector;
        anchor = _end_effector->GetWorldCenter() + b2Vec2(seg_length*0.5 , 0.0f);
        _robot_bodies_v.push_back(body);
      }

      for (auto& actuator: this->_actuators) {
        actuator->set_input(0.);
      }

      _end_effector = robox2d::common::createCircle( world, arm_length*0.015f, b2_dynamicBody, {nb_joints*seg_length,0.0f,0.0f}, 0.1 );
      std::make_shared<robox2d::actuator::Servo>(world,body, _end_effector, anchor); //adding a servo that is not used to glue the end_effector on the arm. Can be improved with a weldjoint or multiple fixture, but currently not supported by robox2d.
    }

    b2Vec2 get_end_effector_pos(){return _end_effector->GetWorldCenter(); }

    float
    get_sum_speed_actuators_squared()
    {
      float sum_speed_actuators_squared = 0.;
      for (const std::shared_ptr<actuator::Actuator>& actuator : _actuators) {
        sum_speed_actuators_squared += std::pow(std::dynamic_pointer_cast<actuator::Servo>(actuator)->get_joint()->GetMotorSpeed(), 2.0f);
      }
      return sum_speed_actuators_squared;
    }

    std::map<b2Body*, rgb_t> get_map_body_color() {
      std::map<b2Body*, rgb_t> map_body_color;

      std::vector<rgb_t> colors_arm_v{ rgb_t{ 0, 138, 198 }, rgb_t{ 17, 0, 192 } };
      for (int index_body_arm = 0; index_body_arm < _robot_bodies_v.size(); ++index_body_arm) {
        b2Body* body_arm = _robot_bodies_v[index_body_arm];

        // alternating between the colors in colors_arm_v
        map_body_color.insert(std::pair<b2Body*, rgb_t>(body_arm,
                                                         colors_arm_v[index_body_arm % colors_arm_v.size()])
                               );
      }

      for (b2Body* body_wall : _walls_v) {
        map_body_color.insert(std::pair<b2Body*, rgb_t>(body_wall, rgb_t{ 104, 104, 104 }));
      }

      map_body_color.insert(std::pair<b2Body*, rgb_t>(_end_effector, rgb_t{ 192, 108, 0 }));

      return map_body_color;
    }

  private:
    b2Body* _end_effector;
    std::vector<b2Body*> _walls_v;
    std::vector<b2Body*> _robot_bodies_v;
  };
}

#endif // AURORA_ARM_ROBOT_HPP

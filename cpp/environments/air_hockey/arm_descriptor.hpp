//
// Created by Luca Grillotti on 09/02/2021.
//

#ifndef AURORA_ARM_DESCRIPTOR_HPP
#define AURORA_ARM_DESCRIPTOR_HPP

#include <iostream>

#include <robox2d/common.hpp>
#include <robox2d/gui/magnum/base_application.hpp>
#include <robox2d/gui/magnum/glfw_application.hpp>
#include <robox2d/gui/magnum/graphics.hpp>
#include <robox2d/gui/magnum/windowless_gl_application.hpp>
#include <robox2d/robot.hpp>
#include <robox2d/descriptor/base_descriptor.hpp>
#include <robox2d/simu.hpp>

#include "environments/air_hockey/arm_robot.hpp"

// for size_t
#include <cstddef>

namespace robox2d {
  class Simu;
  class Robot;

  namespace descriptor {

    struct ArmDescriptor : public BaseDescriptor
    {
    public:
      ArmDescriptor(size_t desc_dump = 1)
        : BaseDescriptor(desc_dump)
        , energy(0.f)
      {}

      virtual void
      operator()()
      {
        const std::shared_ptr<robox2d::Arm>& arm =
          std::static_pointer_cast<robox2d::Arm>(_simu->robots().back());
        energy += arm->get_sum_speed_actuators_squared();
      }

      float energy;
    };
  } // namespace descriptor
} // namespace robox2d

#endif // AURORA_ARM_DESCRIPTOR_HPP

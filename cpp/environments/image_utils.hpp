//
// Created by Luca Grillotti on 23/12/2020.
//

#ifndef AURORA_IMAGE_UTILS_HPP
#define AURORA_IMAGE_UTILS_HPP

#include <robot_dart/gui/helper.hpp>
#include <robox2d/gui/helper.hpp>

namespace robot_dart {
  namespace gui {
    struct ImageSerialisable : robot_dart::gui::Image
    {
      ImageSerialisable() : robot_dart::gui::Image() {}
      explicit ImageSerialisable(const robot_dart::gui::Image& image) : robot_dart::gui::Image(image) {}

      ImageSerialisable& operator=(const robot_dart::gui::Image& image) {
        this->width = image.width;
        this->height = image.height;
        this->channels = image.channels;
        this->data = image.data;
        return *this;
      }

      // Serialization
      template<class Archive>
      void
      serialize(Archive& ar, const unsigned int version)
      {
        ar& BOOST_SERIALIZATION_NVP(this->width);
        ar& BOOST_SERIALIZATION_NVP(this->height);
        ar& BOOST_SERIALIZATION_NVP(this->channels);
        ar& BOOST_SERIALIZATION_NVP(this->data);
      }
    };
  } // namespace gui
} // namespace robot_dart

namespace robox2d {
  namespace gui {
    struct ImageSerialisable : robox2d::gui::Image
    {
      ImageSerialisable() : robox2d::gui::Image() {}
      explicit ImageSerialisable(const robox2d::gui::Image& image) : robox2d::gui::Image(image) {}

      ImageSerialisable& operator=(const robox2d::gui::Image& image) {
        this->width = image.width;
        this->height = image.height;
        this->channels = image.channels;
        this->data = image.data;
        return *this;
      }

      // Serialization
      template<class Archive>
      void
      serialize(Archive& ar, const unsigned int version)
      {
        ar& BOOST_SERIALIZATION_NVP(this->width);
        ar& BOOST_SERIALIZATION_NVP(this->height);
        ar& BOOST_SERIALIZATION_NVP(this->channels);
        ar& BOOST_SERIALIZATION_NVP(this->data);
      }
    };
  } // namespace gui
} // namespace robox2d

#endif // AURORA_IMAGE_UTILS_HPP

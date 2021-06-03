//
// Created by Luca Grillotti on 04/12/2019.
//

#ifndef SFERES2_FIT_MAZE_HPP
#define SFERES2_FIT_MAZE_HPP


#include <SDL/SDL.h>
#include <SDL/SDL_thread.h>

#include <tbb/mutex.h>

#include <modules/libfastsim/src/display.hpp>
#include <modules/libfastsim/src/map.hpp>
#include <modules/libfastsim/src/settings.hpp>
#include "../hexapod/fit_hexapod_utils.hpp"

#include "environments/image_utils.hpp"


// TODO :
/*
 * Downsampling images for neural network
 * Global variable settings map robot
 * Parameters to set
 */



namespace fastsim {
    class DisplaySurface : public Display {
    public:
        DisplaySurface(const boost::shared_ptr<Map> &m, Robot &r) : Display(m, r) {};

        SDL_Surface *screen() const {
            return _screen;
        }

        void update(const Robot &robot) {
            _events();
            // convert to pixel
            unsigned x = _map->real_to_pixel(robot.get_pos().x());
            unsigned y = _map->real_to_pixel(robot.get_pos().y());
            unsigned r = _map->real_to_pixel(robot.get_radius());
            float theta = robot.get_pos().theta();

            // erase robot
            SDL_BlitSurface(_map_bmp, &_prev_bb, _screen, &_prev_bb);
            // erase all
            SDL_BlitSurface(_map_bmp, 0, _screen, 0);

            // lasers
//           _disp_lasers(robot);

            // goals
//           _disp_goals(robot);


            // light sensor
//           _disp_light_sensors(robot);

            // radars
//           _disp_radars(robot);

            // camera
//           _disp_camera(robot);

            // draw the circle again (robot)
            unsigned int col = robot.color();
            const unsigned r_scaled = r;
            _disc(_screen, x, y, r_scaled, _color_from_id(_screen, col));
            _circle(_screen, x, y, r_scaled, 255, 0, 0);
            // direction
            Uint32 color = SDL_MapRGB(_screen->format, 0, 255, 0);


            _line(_screen, x, y, (int) (r * cosf(theta) + x), (int) (r * sinf(theta) + y), color);

            // bumpers
//           _disp_bumpers(robot);

            // illuminated switches
//           _disp_switches();


            SDL_Rect rect;
            _bb_to_sdl(robot.get_bb(), &rect);
            rect.x = std::max(0, std::min((int) rect.x, (int) _prev_bb.x));
            rect.y = std::max(0, std::min((int) rect.y, (int) _prev_bb.y));
            rect.w = std::max(rect.w, _prev_bb.w);
            rect.h = std::max(rect.h, _prev_bb.h);

            if (rect.x + rect.w > _w) rect.w = _w;
            if (rect.y + rect.h > _h) rect.h = _h;


        }

    protected:
        void _disp_bb(const Robot &robot) {
            unsigned x = _map->real_to_pixel(robot.get_bb().x);
            unsigned y = _map->real_to_pixel(robot.get_bb().y);
            unsigned w = _map->real_to_pixel(robot.get_bb().w);
            unsigned h = _map->real_to_pixel(robot.get_bb().h);

            assert(x >= 0);
            assert(y >= 0);
            assert(x + w < (unsigned) _screen->w);
            assert(y + h < (unsigned) _screen->h);
            _line(_screen, x, y, x + w, y, 0);
            _line(_screen, x + w, y, x + w, y + h, 0);
            _line(_screen, x + w, y + h, x, y + h, 0);
            _line(_screen, x, y + h, x, y, 0);
        }

        void _disp_goals(const Robot &robot) {
            for (size_t i = 0; i < _map->get_goals().size(); ++i) {
                const Goal &goal = _map->get_goals()[i];
                unsigned x = _map->real_to_pixel(goal.get_x());
                unsigned y = _map->real_to_pixel(goal.get_y());
                unsigned diam = _map->real_to_pixel(goal.get_diam());
                Uint8 r = 0, g = 0, b = 0;
                switch (goal.get_color()) {
                    case 0:
                        r = 255;
                        break;
                    case 1:
                        g = 255;
                        break;
                    case 2:
                        b = 255;
                        break;
                    default:
                        assert(0);
                }
                _circle(_screen, x, y, diam, r, g, b);
            }
        }

        void _disp_radars(const Robot &robot) {
            unsigned r = _map->real_to_pixel(robot.get_radius()) / 2;
            unsigned x = _map->real_to_pixel(robot.get_pos().x());
            unsigned y = _map->real_to_pixel(robot.get_pos().y());

            for (size_t i = 0; i < robot.get_radars().size(); ++i) {
                const Radar &radar = robot.get_radars()[i];
                if (radar.get_activated_slice() != -1) {
                    float a1 = robot.get_pos().theta() + radar.get_inc() * radar.get_activated_slice();
                    float a2 = robot.get_pos().theta() + radar.get_inc() * (radar.get_activated_slice() + 1);
                    _line(_screen,
                          cos(a1) * r + x, sin(a1) * r + y,
                          cos(a2) * r + x, sin(a2) * r + y,
                          0x0000FF);
                    assert(radar.get_color() < (int) _map->get_goals().size());
                    const Goal &g = _map->get_goals()[radar.get_color()];
                    unsigned gx = _map->real_to_pixel(g.get_x());
                    unsigned gy = _map->real_to_pixel(g.get_y());
                    _line(_screen, x, y, gx, gy, 0x0000FF);
                }

            }

        }

        void _disp_bumpers(const Robot &robot) {
            // convert to pixel
            unsigned x = _map->real_to_pixel(robot.get_pos().x());
            unsigned y = _map->real_to_pixel(robot.get_pos().y());
            unsigned r = _map->real_to_pixel(robot.get_radius());
            float theta = robot.get_pos().theta();
            Uint32 cb_left = SDL_MapRGB(_screen->format, robot.get_left_bumper() ? 255 : 0, 0, 0);
            Uint32 cb_right = SDL_MapRGB(_screen->format, robot.get_right_bumper() ? 255 : 0, 0, 0);
            _line(_screen,
                  (int) (r * cosf(theta + M_PI / 2.0f) + x),
                  (int) (r * sinf(theta + M_PI / 2.0f) + y),
                  (int) (r * cosf(theta) + x),
                  (int) (r * sinf(theta) + y),
                  cb_right);
            _line(_screen,
                  (int) (r * cosf(theta - M_PI / 2.0f) + x),
                  (int) (r * sinf(theta - M_PI / 2.0f) + y),
                  (int) (r * cosf(theta) + x),
                  (int) (r * sinf(theta) + y),
                  cb_left);
        }

        void _disp_lasers(const Robot &robot) {
            _disp_lasers(robot.get_lasers(), robot);
            for (size_t i = 0; i < robot.get_laser_scanners().size(); ++i)
                _disp_lasers(robot.get_laser_scanners()[i].get_lasers(), robot);
        }

        void _disp_lasers(const std::vector<Laser> &lasers, const Robot &robot) {
            for (size_t i = 0; i < lasers.size(); ++i) {
                unsigned x_laser = _map->real_to_pixel(robot.get_pos().x()
                                                       + lasers[i].get_gap_dist() * cosf(robot.get_pos().theta()
                                                                                         + lasers[i].get_gap_angle()));
                unsigned y_laser = _map->real_to_pixel(robot.get_pos().y()
                                                       + lasers[i].get_gap_dist() * sinf(robot.get_pos().theta()
                                                                                         + lasers[i].get_gap_angle()));
                _line(_screen, x_laser, y_laser,
                      lasers[i].get_x_pixel(),
                      lasers[i].get_y_pixel(),
                      0xFF00000);
            }
        }

        void _disp_light_sensors(const Robot &robot) {
            for (size_t i = 0; i < robot.get_light_sensors().size(); ++i) {
                const LightSensor &ls = robot.get_light_sensors()[i];
                unsigned x_ls = _map->real_to_pixel(robot.get_pos().x());
                unsigned y_ls = _map->real_to_pixel(robot.get_pos().y());
                unsigned x_ls1 = _map->real_to_pixel(robot.get_pos().x()
                                                     +
                                                     200. / ((float) ls.get_color() + 1) * cosf(robot.get_pos().theta()
                                                                                                + ls.get_angle() -
                                                                                                ls.get_range() / 2.0));
                unsigned y_ls1 = _map->real_to_pixel(robot.get_pos().y()
                                                     +
                                                     200. / ((float) ls.get_color() + 1) * sinf(robot.get_pos().theta()
                                                                                                + ls.get_angle() -
                                                                                                ls.get_range() / 2.0));
                _line(_screen, x_ls, y_ls, x_ls1, y_ls1, _color_from_id(_screen, ls.get_color()));
                unsigned x_ls2 = _map->real_to_pixel(robot.get_pos().x()
                                                     +
                                                     200. / ((float) ls.get_color() + 1) * cosf(robot.get_pos().theta()
                                                                                                + ls.get_angle() +
                                                                                                ls.get_range() / 2.0));
                unsigned y_ls2 = _map->real_to_pixel(robot.get_pos().y()
                                                     +
                                                     200. / ((float) ls.get_color() + 1) * sinf(robot.get_pos().theta()
                                                                                                + ls.get_angle() +
                                                                                                ls.get_range() / 2.0));
                _line(_screen, x_ls, y_ls, x_ls2, y_ls2, _color_from_id(_screen, ls.get_color()));
                _line(_screen, x_ls1, y_ls1, x_ls2, y_ls2, _color_from_id(_screen, ls.get_color()));

                if (ls.get_activated()) {
                    const IlluminatedSwitch &is = *_map->get_illuminated_switches()[ls.get_num()];
                    unsigned x_is = _map->real_to_pixel(is.get_x());
                    unsigned y_is = _map->real_to_pixel(is.get_y());
                    _line(_screen, x_ls, y_ls, x_is, y_is, _color_from_id(_screen, is.get_color()));
                }
            }
        }

        void _disp_camera(const Robot &robot) {
            static const int pw = 20;
            if (!robot.use_camera())
                return;
            unsigned x_ls = _map->real_to_pixel(robot.get_pos().x());
            unsigned y_ls = _map->real_to_pixel(robot.get_pos().y());
            float a1 = robot.get_pos().theta() + robot.get_camera().get_angular_range() / 2.0;
            _line(_screen, x_ls, y_ls, cos(a1) * 200 + x_ls,
                  sin(a1) * 200 + y_ls, 0x0000ff);
            float a2 = robot.get_pos().theta() - robot.get_camera().get_angular_range() / 2.0;
            _line(_screen, x_ls, y_ls, cos(a2) * 200 + x_ls,
                  sin(a2) * 200 + y_ls, 0x0000ff);

            for (size_t i = 0; i < robot.get_camera().pixels().size(); ++i) {
                int pix = robot.get_camera().pixels()[i];
                Uint32 color = pix == -1 ? 0xffffff : _color_from_id(_screen, pix);
                SDL_Rect r;
                r.x = i * pw;
                r.y = 0;
                r.w = pw;
                r.h = pw;
                SDL_FillRect(_screen, &r, color);
            }

        }

    };

}


namespace aurora {
    namespace env {

        namespace global {
            boost::shared_ptr<fastsim::Settings> settings;
            boost::shared_ptr<fastsim::Map> map;
            boost::shared_ptr<fastsim::DisplaySurface> display;
            tbb::mutex sdl_mutex;
        }

        template<typename Params>
        void init_fastsim_settings() {
            if ((Params::env == aurora::env::Env::hard_maze)
                || (Params::env == aurora::env::Env::hard_maze_sticky)
                || (Params::env == aurora::env::Env::hard_maze_gen_desc)) {
                Params::fit_data::settings = boost::make_shared<fastsim::Settings>(
                        "/git/sferes2/exp/aurora/resources/LS_maze_hard.xml");
            }

            Params::fit_data::map = Params::fit_data::settings->map();
            Params::fit_data::display = boost::make_shared<fastsim::DisplaySurface>(Params::fit_data::map, *(Params::fit_data::settings->robot()));
        }

//        template <typename TParams>
//        int get_one_obs_size() {
//            if (TParams::do_consider_bumpers_in_obs_for_maze) {
//                return global::settings->robot()->get_lasers().size() + 2;
//            } else {
//                return global::settings->robot()->get_lasers().size();
//            }
//        }

        FIT_QD(HardMaze)
        {
        public:
            HardMaze() : m_entropy(-1), m_implicit_fitness_value(-1) {}

            constexpr bool use_sensor_desc() {
                return false;
            }

            template<typename Indiv>
            void eval(Indiv &ind) {
                // TODO : Add Asserts again

                boost::shared_ptr<fastsim::Robot> robot = boost::make_shared<fastsim::Robot>(
                        *Params::fit_data::settings->robot());

                constexpr size_t c_size_input = 5;
                constexpr size_t c_number_iterations = 2000;
                const int c_period_add_successive_sensor_measure = Params::step_measures;
                constexpr size_t c_period_add_successive_action_measure = 50;

                std::vector<float> in(c_size_input);
                std::vector<float> successive_sensor_measures;
                std::vector<float> successive_action_measures;
                m_successive_gt.clear();
                float power_used = 0;

                for (size_t i = 0; i < c_number_iterations; ++i) {
                    // NN indiv policies stay the same across different environments
                    for (size_t index_laser = 0; index_laser < 3; ++index_laser) {
                        in[index_laser] = robot->get_lasers()[index_laser].get_dist();
                    }

                    in[3] = static_cast<float>(robot->get_left_bumper());
                    in[4] = static_cast<float>(robot->get_right_bumper());

                    // If using videos => Store successive sensor measures
                    // If NS -> still save successive measures for future tests
                    if ((Params::use_videos and use_sensor_desc()) or aurora::is_algo_hand_coded()) {
                        // condition satisfied when i=99, 199, 299... 1999
                        if (i % c_period_add_successive_sensor_measure == c_period_add_successive_sensor_measure - 1) {
                            // Adding ALL inputs (including Bumpers)
                            for (const auto & laser : robot->get_lasers()) {
                                successive_sensor_measures.push_back(laser.get_dist());
                            }
                            // TODO: See the impact of adding the left and right bumber in the descriptors
                            if (Params::do_consider_bumpers_in_obs_for_maze) {
                                successive_sensor_measures.push_back(100 * static_cast<float>(robot->get_left_bumper()));
                                successive_sensor_measures.push_back(100 * static_cast<float>(robot->get_right_bumper()));
                            }
                        }
                    }

                    // Store successive GT
                    if (i % c_period_add_successive_sensor_measure == c_period_add_successive_sensor_measure - 1) {
                        // Adding ALL inputs (including Bumpers)
                        // TODO: See the impact of adding the left and right bumber in the descriptors
                        m_successive_gt.push_back(robot->get_pos().get_x());
                        m_successive_gt.push_back(robot->get_pos().get_y());
                        m_successive_gt.push_back(robot->get_pos().theta());
                    }

                    for (int index_laser = 0; index_laser < in.size() - 2; ++index_laser) {
                      // index_laser goes until in.size() - 2 as we do not want to consider the bumpers
                      if ((in[index_laser] < 0.f) or (in[index_laser] > 100.f)) {
                        in[index_laser] = 100.f;
                      }
                    }


                    ind.nn().step(in);
                    constexpr bool use_sticky_walls = (aurora::get_env() == aurora::env::Env::hard_maze_sticky);
                    robot->move(ind.nn().get_outf(0) * 5, ind.nn().get_outf(1) * 5, Params::fit_data::map, use_sticky_walls);
                    // std::cout << ind.nn().get_outf(0) << "  " << ind.nn().get_outf(1) << std::endl;

                    power_used += std::pow(ind.nn().get_outf(0), 2) + std::pow(ind.nn().get_outf(1), 2);
                }

                const int c_size(Params::fit_data::display->screen()->w * Params::fit_data::display->screen()->h * 4);
                uint8_t array_pixels[c_size];
                Params::fit_data::sdl_mutex.lock();

                Params::fit_data::display->update(*robot);
                //fastsim::DisplaySurface d(map, *robot);
                //d.update();

                memcpy(array_pixels, Params::fit_data::display->screen()->pixels, c_size);
                Params::fit_data::sdl_mutex.unlock();

                for (const auto &laser : robot->get_lasers()) {
                    m_lasers_dists.push_back(laser.get_dist());
                }

                this->_update_gt(robot);

                // Observation that will be encoded by the AE
                this->generate_observation(array_pixels, successive_sensor_measures, successive_action_measures);

                m_implicit_fitness_value = -1 * power_used;
                if (aurora::get_has_fit()) {
                  this->_value = m_implicit_fitness_value; // the goal is to minimise the power used
                } else {
                  this->_value = -1; // FITNESS: constant because we're interested in exploration
                }

                // assign descriptor if novelty search algorithm
                if (aurora::is_algo_hand_coded()) {
                    this->assign_descriptor_for_novelty_search_algorithm<Indiv>(successive_sensor_measures, successive_action_measures, ind);
                }
            }

            void generate_observation(uint8_t array_pixels[],
                    const std::vector<float>& successive_sensor_measures,
                    const std::vector<float>& successive_action_measures) {
                /*
                 * Generate Observation that will be encoded by the AE
                 * Depending on the network used storing different kinds of observations
                 */
                if (aurora::does_encode_images()) {
                    // Create Image from top of terrain
                    this->_create_image(array_pixels, Params::fit_data::display->screen()->h, Params::fit_data::display->screen()->w);
                } else if (aurora::does_encode_sequence()) {

                    if (use_sensor_desc()) {
                        // add succession of measures
                        m_image_float = successive_sensor_measures;
                    } else if (Params::env == aurora::env::Env::hard_maze) {
                        // if in standard environment, add succession of GT
                        m_image_float = m_successive_gt;
                    }
                } else if (aurora::is_algo_hand_coded()) {
                    // If NS -> still save successive measures for future tests
                    m_image_float = successive_sensor_measures;
                }
            }

            template<typename Indiv>
            void assign_descriptor_for_novelty_search_algorithm(const std::vector<float>& successive_sensor_measures,
                                                                const std::vector<float>& successive_action_measures,
                                                                Indiv& ind) {
                if ((Params::env == aurora::env::Env::hard_maze)
                    or (Params::env == aurora::env::Env::hard_maze_sticky)) { // HARD MAZE ENV
                    if (not Params::use_videos) {
                        std::vector<double> gt_double(gt().begin(), gt().begin() + 2); // x,y position
                        this->set_desc(gt_double);
                    } else {
                        std::vector<double> successive_gt_double(m_successive_gt.begin(), m_successive_gt.end()); // x,y position
                        this->set_desc(successive_gt_double);
                    }
                } else if (Params::env == aurora::env::Env::hard_maze_gen_desc) {
                    std::vector<float> gen_data = ind.gen().data();
                    std::vector<double> gen_desc_double(gen_data.begin(), gen_data.end()); // x,y position
                    this->set_desc(gen_desc_double);
                } else if (use_sensor_desc()) { // HARD MAZE 3 LASERS
                    if (!Params::use_videos) { // Not using videos => only using last laser desc
                        std::vector<double> lasers_desc(lasers_dists().begin(), lasers_dists().end());
                        this->set_desc(lasers_desc);
                    } else { // Using all sensor measures from beginning to end
                        std::vector<double> sensors_desc(successive_sensor_measures.begin(),
                                                         successive_sensor_measures.end());
                        this->set_desc(sensors_desc);
                    }
                } else { // Using actions as descriptor
                    if (!Params::use_videos) {
                        std::vector<double> actions_desc = {static_cast<double>(ind.nn().get_outf(0)),
                                                            static_cast<double>(ind.nn().get_outf(1))};
                        this->set_desc(actions_desc);

                    } else {
                        std::vector<double> actions_desc(successive_action_measures.begin(),
                                                         successive_action_measures.end());
                        this->set_desc(actions_desc);
                    }
                }
            }

            std::vector<float> &gt() { return m_gt; }

            const robot_dart::gui::ImageSerialisable &get_rgb_image() const { return m_rgb_image; }

            float &entropy() { return m_entropy; }
            const float &entropy() const { return m_entropy; }

            float &implicit_fitness_value() { return this->m_implicit_fitness_value; }
            const float &implicit_fitness_value() const { return this->m_implicit_fitness_value; }

            const std::vector<float> &observations() const {
                return m_image_float;
            }

            const std::vector<float> &lasers_dists() const {
                return m_lasers_dists;
            }

            const std::vector<float> &successive_gt() const {
                return m_successive_gt;
            }

            size_t get_flat_obs_size() const {
                return observations().size();
            }

            template<typename block_t>
            void get_flat_observations(block_t &data) const {
                // std::cout << _image.size() << std::endl;
                for (size_t i = 0; i < m_image_float.size(); i++) {
                    data(0, i) = m_image_float[i];
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

            ar& BOOST_SERIALIZATION_NVP(this->m_entropy);
            ar& BOOST_SERIALIZATION_NVP(this->m_implicit_fitness_value);
            ar& BOOST_SERIALIZATION_NVP(this->m_image_float);
            ar& BOOST_SERIALIZATION_NVP(this->m_rgb_image);
            ar& BOOST_SERIALIZATION_NVP(this->m_gt);
            ar& BOOST_SERIALIZATION_NVP(this->m_lasers_dists);
            ar& BOOST_SERIALIZATION_NVP(this->m_successive_gt);
          }


        protected:
            float m_entropy;
            float m_implicit_fitness_value;
            std::vector<float> m_image_float;
            robot_dart::gui::ImageSerialisable m_rgb_image;
            std::vector<float> m_gt;
            std::vector<float> m_lasers_dists;
            std::vector<float> m_successive_gt;

            void _update_gt(const boost::shared_ptr<fastsim::Robot> &robot) {
                constexpr size_t c_size_gt = 3;
                m_gt.clear();
                m_gt.reserve(c_size_gt);
                m_gt.push_back(robot->get_pos().get_x());
                m_gt.push_back(robot->get_pos().get_y());
                m_gt.push_back(robot->get_pos().theta());
            }

            void _create_image(uint8_t array_pixels[], const int w, const int h, const bool do_compute_obs_vector=true) {
              
                _store_image(array_pixels, w, h);

                if (do_compute_obs_vector) {
                  if (Params::use_colors) {
                      _store_rgb_vector();
                  } else {
                      _store_grayscale_vector();
                  }
                }
            }

            void _store_image(uint8_t array_pixels[], const int w, const int h) {
                m_rgb_image.width = w;
                m_rgb_image.height = h;
                m_rgb_image.data.clear();
                m_rgb_image.data.reserve(3 * w * h);
                for (int index_row = 0; index_row < h; ++index_row) {
                    std::vector<std::vector<uint8_t>> v_row;
                    for (int index_col = 0; index_col < w; ++index_col) {
                        // std::vector<uint8_t> v_pixel;
                        // v_pixel.assign(&array_pixels[4 * (index_row + index_col * w)], &array_pixels[4 * (index_row + index_col * w)] + 3);
                        m_rgb_image.data.insert(m_rgb_image.data.end(), &array_pixels[4 * (index_row + index_col * w)], &array_pixels[4 * (index_row + index_col * w)] + 3);
                        // v_row.push_back(v_pixel);
                    }
                    // m_rgb_image.push_back(v_row);
                }
            }

            void _store_rgb_vector() {
                std::vector<float> img_red_float;
                std::vector<float> img_green_float;
                std::vector<float> img_blue_float;

                std::vector<uint8_t> img_red_uint8;
                std::vector<uint8_t> img_green_uint8;
                std::vector<uint8_t> img_blue_uint8;

                size_t height = m_rgb_image.height;
                size_t width = m_rgb_image.width;
                size_t nb_channels = m_rgb_image.channels;

                for (size_t h = 0; h < height; h++) {
                    for (size_t w = 0; w < width; w++) {
                        int id_rgb = w * nb_channels + h * (width * nb_channels);

                        img_red_uint8.push_back(m_rgb_image.data[id_rgb + 0]);
                        img_green_uint8.push_back(m_rgb_image.data[id_rgb + 1]);
                        img_blue_uint8.push_back(m_rgb_image.data[id_rgb + 2]);
                    }
                }

                for (size_t i = 0; i < img_red_uint8.size(); ++i) {
                    img_red_float.push_back(img_red_uint8[i] / 255.0);
                    img_green_float.push_back(img_green_uint8[i] / 255.0);
                    img_blue_float.push_back(img_blue_uint8[i] / 255.0);
                }

                for (int k = Params::times_downsample; k > 1; k /= 2) {
                    _downsample(img_red_float, k);
                    _downsample(img_green_float, k);
                    _downsample(img_blue_float, k);
                }

                m_image_float.insert(m_image_float.end(), img_red_float.begin(), img_red_float.end());
                m_image_float.insert(m_image_float.end(), img_green_float.begin(), img_green_float.end());
                m_image_float.insert(m_image_float.end(), img_blue_float.begin(), img_blue_float.end());
            }

            void _store_grayscale_vector() {
                const std::vector<uint8_t>& img_uint8 = m_rgb_image.data;

                float pixel;

                for (size_t i = 0; i < img_uint8.size(); i += 3) {
                    pixel = R * (float) img_uint8[i] + G * (float) img_uint8[i + 1] +
                            B * (float) img_uint8[i + 2];
                    m_image_float.push_back(pixel / 255.0);
                }

                for (int k = Params::times_downsample; k > 1; k /= 2) {
                    _downsample(m_image_float, k);
                }
            }

            void _downsample(std::vector<float> &img, int k) const {
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
                        // Using Average Pooling to reduce dimensionality
//                        img.push_back(std::max({image(i, j), image(i + 1, j), image(i, j + 1), image(i + 1, j + 1)}));
                        img.push_back((image(i, j) + image(i + 1, j) + image(i, j + 1) + image(i + 1, j + 1)) / 4);
                    }
                }
            }

            int _get_number_sensors_to_consider() {
                if (Params::do_consider_bumpers_in_obs_for_maze) {
                    return 5; // 3 lasers + 2 bumpers
                } else {
                    return 3; // 3 lasers, usual case
                }
            }
        };
    }
}
#endif //SFERES2_FIT_MAZE_HPP

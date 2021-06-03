#ifndef INTERACTIVE_MAP_HPP
#define INTERACTIVE_MAP_HPP

#define _USE_MATH_DEFINES

#include <cmath>
#include <iostream>
#include <fstream>
#include <sferes/run.hpp>
#include <sferes/stc.hpp>

using namespace std;

typedef std::vector <std::vector<double>> descriptors_archive_t;
typedef std::vector<double> performance_archive_t;
typedef std::vector <std::vector<double>> controllers_archive_t;
typedef std::vector <std::vector<double>> gt_archive_t;
typedef std::vector <std::vector<float>> succession_measures_archive_t;

typedef std::tuple <std::vector<size_t>, descriptors_archive_t, performance_archive_t, controllers_archive_t> archive_content_t;
typedef std::tuple <descriptors_archive_t, gt_archive_t, std::vector<size_t>> stat_projection_t;


template<typename Params>
class Interactive_map {

public:

    Interactive_map() {}

    void load_archive(const std::string &path,
                      archive_content_t &archive_all_individuals,
                      const size_t behav_dim_in_file) {
        std::vector<size_t> &indexes = std::get<0>(archive_all_individuals);
        descriptors_archive_t &descriptors_archive = std::get<1>(archive_all_individuals);
        performance_archive_t &performances_archive = std::get<2>(archive_all_individuals);
        controllers_archive_t &controllers_archive = std::get<3>(archive_all_individuals);

        descriptors_archive.clear();
        performances_archive.clear();
        controllers_archive.clear();

        std::ifstream file(path);
        if (!file.is_open()) {
            std::cout << " error: Impossible to open " << path << std::endl;
            exit(2);
        }

        std::string line;
        std::cout << "Behavioural dimension (considered for reading): " << behav_dim_in_file << std::endl;
        while (std::getline(file, line)) {
            std::istringstream iss(line);
            double data;
            iss >> data; //first value is index
            indexes.push_back(data);

            std::vector<double> desc;
            for (size_t i = 0; i < behav_dim_in_file; i++) {
                iss >> data;
                desc.push_back(data);
            }
            descriptors_archive.push_back(desc);
            iss >> data;
            performances_archive.push_back(data);
            std::vector<double> ctrl;
            while (iss >> data) // while the stream is not empty
                ctrl.push_back(data);
            controllers_archive.push_back(ctrl);
        }
    }

    void load_archive(const std::string &path,
                      archive_content_t &archive_all_individuals) {
        load_archive(path, archive_all_individuals, Params::qd::behav_dim);
    }

    void load_stat_projection(const std::string &path, stat_projection_t &stat_projection_all_individuals, size_t behav_dim) {
        descriptors_archive_t &descriptors_archive = std::get<0>(stat_projection_all_individuals);
        gt_archive_t &gt_archive = std::get<1>(stat_projection_all_individuals);
        std::vector<size_t> &indexes = std::get<2>(stat_projection_all_individuals);

        descriptors_archive.clear();
        gt_archive.clear();

        std::ifstream file(path);
        if (!file.is_open()) {
            std::cout << " error: Impossible to open " << path << std::endl;
            exit(2);
        }

        std::string line;
        std::cout << "Behavioural dimension: " << behav_dim << std::endl;
        while (std::getline(file, line)) {
            std::istringstream iss(line);
            float data;
            iss >> data; //first value is index
            indexes.push_back(data);
            iss >> data; //entropy is useless
            std::vector<double> desc;
            for (int i = 0; i < behav_dim; i++) {
                iss >> data;
                desc.push_back(data);
            }
            descriptors_archive.push_back(desc);

            std::vector<double> gt_ind;
            while (iss >> data) // while the stream is not empty
                gt_ind.push_back(data);
            gt_archive.push_back(gt_ind);
        }
    }

    void load_stat_projection(const std::string &path, stat_projection_t &stat_projection_all_individuals) {
        load_stat_projection(path, stat_projection_all_individuals, Params::qd::behav);
    }

    void load_stat_sequence_observations(const std::string &path, succession_measures_archive_t &succession_measures) {
        succession_measures.clear();

        std::ifstream file(path);
        if (!file.is_open()) {
            std::cout << " error: Impossible to open " << path << std::endl;
            exit(2);
        }

        std::string line;
        while (std::getline(file, line)) {
            std::istringstream iss(line);
            double data;
            iss >> data; //first value is index, so we skip it
            std::vector<float> one_succession_measures;
            while (iss >> data) // while the stream is not empty
                one_succession_measures.push_back(static_cast<float>(data));
            succession_measures.push_back(one_succession_measures);
        }
    }

    void get_gt_from_descriptor(const std::vector<double> &descriptor,
                                const stat_projection_t &stat_projection_all_individuals,
                                std::vector<double> &gt) {
        const descriptors_archive_t &descriptors_archive = std::get<0>(stat_projection_all_individuals);
        const gt_archive_t &gt_archive = std::get<1>(stat_projection_all_individuals);

        for (size_t i = 0; i < gt_archive.size(); ++i) {
            if (descriptors_archive[i] == descriptor) {
                gt = gt_archive[i];
                break;
            }
        }
    }

    double get_distance_from_origin(const std::vector<double> &gt_individual) {
        return std::sqrt(std::pow(gt_individual[0], 2) + std::pow(gt_individual[1], 2));
    }

    double get_angle_with_x_axis(std::vector<double> &gt_individual) {
        const double &x = gt_individual[0];
        const double &y = gt_individual[1];

        if (x >= 0) {
            return atan(y / x);
        } else if (y >= 0) {
            return M_PI - atan(std::abs(y / x));
        } else {
            return -1. * atan(std::abs(y / x));
        }
    }

private:
    std::string xLabel = "";
    std::string yLabel = "";

    bool isMappedX = false;
    bool isMappedY = false;

    double orgXlow, orgXhig, desXlow, desXhig;
    double orgYlow, orgYhig, desYlow, desYhig;

    double maximalSquaredDistance = 0.25;

    bool is_manual_control;

    double mapValue(double a, double b, double c, double d, double x) {
        return c + ((d - c) / (b - a)) * (x - a);
    }


    double get_point_size() {
        double val1 = 1;
        for (size_t i = 0; i < Params::qd::grid_shape_size(); i += 2)
            val1 *= Params::qd::grid_shape(i);
        double val2 = 1;
        for (size_t i = 1; i < Params::qd::grid_shape_size(); i += 2)
            val2 *= Params::qd::grid_shape(i);

        double val = std::max(val1, val2);
        return 100000.0 / (val * val);
    }

    double squared_dist(double x1, double y1, double x2, double y2) {
        return (x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2);
    }

    size_t get_index(const std::vector<double> &x, std::vector<double> y, std::vector <std::array<double, 2>> click) {
        assert(x.size() == y.size());
        for (size_t it = 0; it < x.size(); it++)
            if (squared_dist(x[it], y[it], click[0][0], click[0][1]) <=
                maximalSquaredDistance) //maximal squared distance from the center within a cell of radius 1.
                return it;
        return x.size();
    }


public:
    void setXLabel(std::string xlabel) {
        xLabel = xlabel;
    }

    void setYLabel(std::string ylabel) {
        yLabel = ylabel;
    }

    //NOTE: the functions below are experimental and should be use with a 2D behaviour descriptor only.
    void mapX(double _orgXlow, double _orgXhig, double _desXlow, double _desXhig) {
        isMappedX = true;
        orgXlow = _orgXlow;
        orgXhig = _orgXhig;

        desXlow = _desXlow;
        desXhig = _desXhig;

        double d = (desXhig - desXlow) / 200;
        maximalSquaredDistance = d * d;
    }

    void mapY(double _orgYlow, double _orgYhig, double _desYlow, double _desYhig) {
        isMappedY = true;
        orgYlow = _orgYlow;
        orgYhig = _orgYhig;

        desYlow = _desYlow;
        desYhig = _desYhig;
    }

};

#endif

//
// Created by Luca Grillotti on 31/10/2019.
//

#ifndef SFERES2_PREPROCESSOR_HPP
#define SFERES2_PREPROCESSOR_HPP

#include <iostream>

template<typename TParams>
class RescaleFeature {
public:
    RescaleFeature<TParams>(float epsilon_std) : no_prep(true), m_epsilon_std(epsilon_std) {}

    typedef Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> Mat;

    void init() {
        no_prep = true;
    }

    void init(const Mat &data) {
        no_prep = false;
        m_mean_dataset = data.colwise().mean();
        m_std_dataset = (data.array().rowwise() - m_mean_dataset.transpose().array()).pow(2).colwise().mean().sqrt();
    }

    void apply(const Mat &data, Mat &res) const {
        if (no_prep) {
            res = data;
        } else {
            // res = a + (data.array() - _min) * (b - a) / (_max - _min);

            res = data.array().rowwise() - m_mean_dataset.transpose().array();
            if (data.rows() > 1) { // If strictly more than one individual, we can supposedly divide by std
              res = res.array().rowwise() / (m_std_dataset.transpose().array() + m_epsilon_std);
            } else {
              res = data;
            }
        }
    }

    void deapply(const Mat &data, Mat &res, const bool do_clip_zero_one) const {
        if (no_prep) {
            res = data;
        } else {
            res = data.array().rowwise() * (m_std_dataset.transpose().array() + m_epsilon_std);
            res = res.array().rowwise() + m_mean_dataset.transpose().array();
            if (do_clip_zero_one) {
                res = res.cwiseMax(0.).cwiseMin(1.);
            }
        }
    }

    // Serialization
    template<class Archive>
    void
    save(Archive& ar, const unsigned int version) const
    {
      ar& BOOST_SERIALIZATION_NVP(this->no_prep);
      ar& BOOST_SERIALIZATION_NVP(this->m_epsilon_std);

      std::vector<float> mean_dataset_vector(this->m_mean_dataset.data(),
                                             this->m_mean_dataset.data()
                                             + this->m_mean_dataset.rows() * this->m_mean_dataset.cols());

      std::vector<float> std_dataset_vector(this->m_std_dataset.data(),
                                            this->m_std_dataset.data()
                                            + this->m_std_dataset.rows() * this->m_std_dataset.cols());

      ar& BOOST_SERIALIZATION_NVP(mean_dataset_vector);
      ar& BOOST_SERIALIZATION_NVP(std_dataset_vector);
    }

    // Serialization
    template<class Archive>
    void
    load(Archive& ar, const unsigned int version)
    {
      ar& BOOST_SERIALIZATION_NVP(this->no_prep);
      ar& BOOST_SERIALIZATION_NVP(this->m_epsilon_std);

      std::vector<float> mean_dataset_vector, std_dataset_vector;

      ar& BOOST_SERIALIZATION_NVP(mean_dataset_vector);
      ar& BOOST_SERIALIZATION_NVP(std_dataset_vector);

      this->m_mean_dataset = Eigen::Map<Eigen::VectorXf>(mean_dataset_vector.data(), mean_dataset_vector.size());
      this->m_std_dataset = Eigen::Map<Eigen::VectorXf>(std_dataset_vector.data(), std_dataset_vector.size());
    }

    BOOST_SERIALIZATION_SPLIT_MEMBER();

  protected:
    bool no_prep;
    Eigen::VectorXf m_mean_dataset;
    Eigen::VectorXf m_std_dataset;
    float m_epsilon_std;
};

#endif //SFERES2_PREPROCESSOR_HPP

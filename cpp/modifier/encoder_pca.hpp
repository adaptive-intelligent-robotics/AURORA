//
// Created by Luca Grillotti on 05/12/2020.
//

#ifndef AURORA_ENCODER_PCA_HPP
#define AURORA_ENCODER_PCA_HPP

#include <fstream>

#include <Eigen/Core>


namespace Eigen{
  template<class Matrix>
  void write_binary(const char* filename, const Matrix& matrix){
    std::ofstream out(filename, std::ios::out | std::ios::binary | std::ios::trunc);
    typename Matrix::Index rows=matrix.rows(), cols=matrix.cols();
    out.write((char*) (&rows), sizeof(typename Matrix::Index));
    out.write((char*) (&cols), sizeof(typename Matrix::Index));
    out.write((char*) matrix.data(), rows*cols*sizeof(typename Matrix::Scalar) );
    out.close();
  }
  template<class Matrix>
  void read_binary(const char* filename, Matrix& matrix){
    std::ifstream in(filename, std::ios::in | std::ios::binary);
    typename Matrix::Index rows=0, cols=0;
    in.read((char*) (&rows),sizeof(typename Matrix::Index));
    in.read((char*) (&cols),sizeof(typename Matrix::Index));
    matrix.resize(rows, cols);
    in.read( (char *) matrix.data() , rows*cols*sizeof(typename Matrix::Scalar) );
    in.close();
  }
} // Eigen::

namespace aurora {
  template<typename TParams, typename Exact = stc::Itself>
  class EncoderPCA : public stc::Any<Exact>
  {
  public:
    typedef Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> MatrixXf_rm;

    explicit EncoderPCA() = default;

    void
    eval(const MatrixXf_rm& data,
         MatrixXf_rm& descriptors,
         MatrixXf_rm& reconstruction_loss,
         MatrixXf_rm& reconstructed_data)
    {
      if (m_mean_dataset.cols() == 0) {
        this->training(data);
      }
      MatrixXf_rm centered_data = data.rowwise() - m_mean_dataset;
      descriptors = (m_eigenvector_matrix.transpose() * centered_data.transpose()).transpose();
      reconstructed_data = (m_eigenvector_matrix * descriptors.transpose()).transpose();
      reconstruction_loss = (data - (reconstructed_data.rowwise() + m_mean_dataset)).rowwise().lpNorm<2>();
    }

    /**
     * Used to calculate the mean and the eigenvector matrix that will be used to project the data to
     * the latent space.
     * @param data
     * @param full_train
     * @param generation
     * @return
     */
    float
    training(const MatrixXf_rm& data)
    {
      m_mean_dataset = data.colwise().mean();
      MatrixXf_rm centered_data = data.rowwise() - m_mean_dataset;

      Eigen::SelfAdjointEigenSolver<MatrixXf_rm> eigensolver(centered_data.transpose() * centered_data);
      if (eigensolver.info() != 0) {
        abort();
      }

      m_eigenvector_matrix = eigensolver.eigenvectors().rightCols<TParams::qd::behav_dim>();
      const float c_recon_loss_data = this->get_avg_recon_loss(data);

      return c_recon_loss_data;
    }

    void
    get_reconstruction(const MatrixXf_rm& data, MatrixXf_rm& reconstruction)
    {
      MatrixXf_rm desc, recon_loss;
      eval(data, desc, recon_loss, reconstruction);
    }

    float
    get_avg_recon_loss(const MatrixXf_rm& data)
    {
      MatrixXf_rm descriptors, recon_loss, reconst;
      eval(data, descriptors, recon_loss, reconst);
      return recon_loss.mean();
    }

    void save_model(const std::string& name_file) const {
      Eigen::write_binary((name_file + "-mean-dataset").data(), m_mean_dataset);
      Eigen::write_binary((name_file + "-eigenvector-matrix").data(), m_eigenvector_matrix);
    }

    void load_model(const std::string& name_file) {
      Eigen::read_binary((name_file + "-mean-dataset").data(), m_mean_dataset);
      Eigen::read_binary((name_file + "-eigenvector-matrix").data(), m_eigenvector_matrix);
    }

    // Serialization
    template<class Archive>
    void
    save(Archive& ar, const unsigned int version) const
    {
      std::vector<float> mean_dataset_vector(this->m_mean_dataset.data(),
                                             this->m_mean_dataset.data()
                                             + this->m_mean_dataset.rows() * this->m_mean_dataset.cols());

      std::vector<float> eigenvector_matrix_vector(this->m_eigenvector_matrix.data(),
                                            this->m_eigenvector_matrix.data()
                                            + this->m_eigenvector_matrix.rows() * this->m_eigenvector_matrix.cols());

      ar& BOOST_SERIALIZATION_NVP(mean_dataset_vector);
      ar& BOOST_SERIALIZATION_NVP(eigenvector_matrix_vector);
    }

    // Serialization
    template<class Archive>
    void
    load(Archive& ar, const unsigned int version)
    {
      std::vector<float> mean_dataset_vector, eigenvector_matrix_vector;

      ar& BOOST_SERIALIZATION_NVP(mean_dataset_vector);
      ar& BOOST_SERIALIZATION_NVP(eigenvector_matrix_vector);

      this->m_mean_dataset = Eigen::Map<Eigen::RowVectorXf>(mean_dataset_vector.data(), mean_dataset_vector.size());
      this->m_eigenvector_matrix = Eigen::Map<Eigen::MatrixXf>(eigenvector_matrix_vector.data(), mean_dataset_vector.size(), TParams::qd::behav_dim);
    }

    BOOST_SERIALIZATION_SPLIT_MEMBER();

  protected:
    Eigen::RowVectorXf m_mean_dataset;
    Eigen::MatrixXf m_eigenvector_matrix;
  };
} // namespace aurora

#endif // AURORA_ENCODER_PCA_HPP

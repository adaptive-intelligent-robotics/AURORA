#ifndef __NETWORK__LOADER__HPP__
#define __NETWORK__LOADER__HPP__

#include <torch/torch.h>
#include <c10/cuda/CUDACachingAllocator.h>

#include <sferes/misc/rand.hpp>

#include <chrono>
#include <iomanip>

#include <Eigen/Core>

// #include <sferes/dbg/dbg.hpp>
#include <sferes/stc.hpp>

#include "dbg_tools/dbg.hpp"
#include "autoencoder/encoder.hpp"
#include "autoencoder/decoder.hpp"
#include "autoencoder/autoencoder.hpp"
#include "autoencoder/lstm_auto_encoder.hpp"
#include "autoencoder/video_auto_encoder.hpp"
#include "autoencoder/convolutional_sequence_autoencoder.hpp"
#include "autoencoder/mlp_autoencoder.hpp"

namespace aurora {
    template <typename TParams, typename Exact = stc::Itself>
    class AbstractLoader : public stc::Any<Exact> {
    public:
        typedef Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> MatrixXf_rm;

        explicit AbstractLoader(std::size_t latent_size, torch::nn::AnyModule auto_encoder_module) :
                m_global_step(0),
                m_one_obs_size(TParams::get_one_obs_size()),
                m_use_colors(TParams::use_colors),
                m_auto_encoder_module(std::move(auto_encoder_module)),
                m_device(torch::cuda::is_available() ? torch::kCUDA : torch::kCPU){
            if (torch::cuda::is_available()) {
//                const char* cuda_visible_devices = std::getenv("CUDA_VISIBLE_DEVICES");
//                std::string str_cuda_visible_devices = (cuda_visible_devices == NULL) ? std::string("") : std::string(cuda_visible_devices);
//                if (not str_cuda_visible_devices.empty()) {
//                    int index_device_to_use = std::stoi(str_cuda_visible_devices);
//                    m_device = torch::Device(m_device.type(), index_device_to_use);
//                    std::cout << "Torch -> Using CUDA ; index device: " << index_device_to_use << std::endl;
//                } else {
//                    std::cout << "Torch -> Using CUDA ; no specified index device " << std::endl;
//                }
                std::cout << "Torch -> Using CUDA" << std::endl;
            } else {
                std::cout << "Torch -> Using CPU" << std::endl;
            }

            this->m_auto_encoder_module.ptr()->to(this->m_device);
        }

        void eval(const MatrixXf_rm &data,
                  MatrixXf_rm &descriptors,
                  MatrixXf_rm &recon_loss,
                  MatrixXf_rm &reconstructed_data) {
            stc::exact(this)->eval(data, descriptors, recon_loss, reconstructed_data);
        }

        void prepare_batches(std::vector<torch::Tensor> &batches, const MatrixXf_rm &data) const {
            stc::exact(this)->prepare_batches(batches, data);
        }

        void split_dataset(const MatrixXf_rm &data, MatrixXf_rm &train, MatrixXf_rm &valid) {
            Eigen::PermutationMatrix<Eigen::Dynamic, Eigen::Dynamic> perm(data.rows());
            perm.setIdentity();
            std::random_shuffle(perm.indices().data(), perm.indices().data() + perm.indices().size());
            MatrixXf_rm tmp;
            if (data.rows() > 10000) {
              tmp = (perm * data).topRows(10000);
            } else {
              tmp = (perm * data);
            }
            size_t l_train{0}, l_valid{0};
            if (tmp.rows() > 500) {
                l_train = floor(tmp.rows() * Options::CV_fraction);
                l_valid = tmp.rows() - l_train;
            } else {
                l_train = tmp.rows();
                l_valid = tmp.rows();
            }
            assert(l_train != 0 && l_valid != 0);

            train = tmp.topRows(l_train);
            valid = tmp.bottomRows(l_valid);
        }

        float training(const MatrixXf_rm &data, bool full_train = false, int generation = 1000) {
            airl::dbg::trace trace(AIRL_DBG_HERE);
            torch::optim::Adam m_adam_optimiser = torch::optim::Adam(m_auto_encoder_module.ptr()->parameters(),
                                                  torch::optim::AdamOptions(1e-3)
                                                          .betas({0.9, 0.999}));
            MatrixXf_rm train_db, valid_db;
            this->split_dataset(data, train_db, valid_db);

            float init_tr_recon_loss = this->get_avg_recon_loss(train_db);
            float init_vl_recon_loss = this->get_avg_recon_loss(valid_db);

            std::cout << "INIT recon train loss: " << init_tr_recon_loss << "   valid recon loss: " << init_vl_recon_loss;

            bool _continue = true;
            const int size_previous_avg = 5 * std::max(1, static_cast<int>(data.rows() / Options::batch_size));
            Eigen::VectorXd previous_avg = Eigen::VectorXd::Ones(size_previous_avg) * 100;

            int nb_epochs = Options::nb_epochs;

            int epoch(0);

            while (_continue && (epoch < nb_epochs)) {
                airl::dbg::out(airl::dbg::debug) << "Split dataset\n";
                this->split_dataset(data, train_db, valid_db);
                std::vector<torch::Tensor> batches;
                airl::dbg::out(airl::dbg::debug) << "Prepare batches\n";
                prepare_batches(batches, train_db);
                airl::dbg::out(airl::dbg::debug) << "Training\n";

                for (auto &batche : batches) {
                    // Get the names below with the inspect_graph.py script applied on the generated graph_text.pb file.
                    this->m_auto_encoder_module.ptr()->zero_grad();
                    m_adam_optimiser.zero_grad();
                    torch::Tensor reconstruction_tensor = this->m_auto_encoder_module.forward(batche);
                    torch::Tensor loss_reconstruction = torch::mse_loss(reconstruction_tensor, batche);
                    loss_reconstruction.backward();
                    m_adam_optimiser.step();

                    // early stopping
                    const float c_recon_loss_t = this->get_avg_recon_loss(train_db);
                    const float c_recon_loss_v = this->get_avg_recon_loss(valid_db);

                    if (!full_train) {
                        previous_avg[epoch % size_previous_avg] = c_recon_loss_v;

                        if ((previous_avg.array() - previous_avg[(epoch + 1) % size_previous_avg]).mean() > 0
                                and c_recon_loss_t < init_tr_recon_loss) {

                            _continue = false;
                            break;
                        }
                    }

                    ++epoch;

                    std::cout.precision(5);
                    std::cout << "training dataset: " << train_db.rows() << "  valid dataset: " << valid_db.rows() << " - ";
                    std::cout << std::setw(5) << epoch << "/" << std::setw(5) << nb_epochs;
                    std::cout << " recon loss (t): " << std::setw(8) << c_recon_loss_t;
                    std::cout << " (v): " << std::setw(8) << c_recon_loss_v;
                    std::cout << std::flush << '\r';
                }

                this->m_global_step++;
            }

            const float c_recon_loss_data = this->get_avg_recon_loss(data);
            std::cout << "Final recon loss: " << c_recon_loss_data << '\n';


          c10::cuda::CUDACachingAllocator::emptyCache();

          return c_recon_loss_data;
        }

        void get_reconstruction(const MatrixXf_rm &data, MatrixXf_rm &reconstruction) {
            MatrixXf_rm desc, recon_loss;
            eval(data, desc, recon_loss, reconstruction);
        }


        float get_avg_recon_loss(const MatrixXf_rm &data) {
            MatrixXf_rm descriptors, recon_loss, reconst;
            eval(data, descriptors, recon_loss, reconst);
            return recon_loss.mean();
        }

        torch::nn::AnyModule get_auto_encoder() {
            return this->m_auto_encoder_module;
        }

        torch::nn::AnyModule& auto_encoder() {
            return this->m_auto_encoder_module;
        }

        void
        save_model(const std::string& name_file) const
        {
          torch::save(m_auto_encoder_module.ptr(), name_file);
        }

        void
        load_model(const std::string& name_file) const
        {
          torch::load(m_auto_encoder_module, name_file.data());
        }

      // Serialization
      template<class Archive>
      void
      save(Archive& ar, const unsigned int version) const
      {
        ar& BOOST_SERIALIZATION_NVP(m_global_step);
        ar& BOOST_SERIALIZATION_NVP(this->m_one_obs_size);
        ar& BOOST_SERIALIZATION_NVP(this->m_use_colors);

        std::map<std::string, std::vector<float>> map;
        for (auto& pair: m_auto_encoder_module.ptr()->named_parameters()) {
          // std::cout << pair.key() << " - " << pair.value() << " - " << pair.value().numel() << " - " << pair.value().data_ptr<float>() << std::endl;
          torch::Tensor t = pair.value().cpu();
          map.insert(
            std::pair<std::string, std::vector<float>>
              (pair.key(), std::vector<float>(t.data_ptr<float>(), t.data_ptr<float>() + t.numel())
              ));
        }

        ar& BOOST_SERIALIZATION_NVP(map);
      }

      // Serialization
      template<class Archive>
      void
      load(Archive& ar, const unsigned int version)
      {
        ar& BOOST_SERIALIZATION_NVP(m_global_step);
        ar& BOOST_SERIALIZATION_NVP(this->m_one_obs_size);
        ar& BOOST_SERIALIZATION_NVP(this->m_use_colors);

        std::map<std::string, std::vector<float>> map;

        ar& BOOST_SERIALIZATION_NVP(map);

        for (auto& pair: m_auto_encoder_module.ptr()->named_parameters()) {
          torch::Tensor tensor_cpu = pair.value().cpu();
          float *data = tensor_cpu.data_ptr<float>();
          std::vector<float> vector_data = map[pair.key()];
          memcpy(data, vector_data.data(), vector_data.size() * sizeof(float));
          pair.value() = tensor_cpu.to(this->m_device);
          // std::cout << pair.key() << " - " << pair.value() << std::endl;
        }
      }

      BOOST_SERIALIZATION_SPLIT_MEMBER();


    protected:
        int32_t m_global_step;
        int m_one_obs_size;
        bool m_use_colors;

        torch::nn::AnyModule m_auto_encoder_module;
        torch::Device m_device;


        struct Options {
            // config setting
            static constexpr int batch_size = TParams::batch_size;
            static constexpr int nb_epochs = TParams::nb_epochs;
            static constexpr float convergence_epsilon = 0.0000001;
            SFERES_CONST float CV_fraction = 0.75;
        };

        void get_torch_tensor_from_eigen_matrix(const MatrixXf_rm &M, torch::Tensor &T) const {

            T = torch::rand({M.rows(), M.cols()});
            float *data = T.data_ptr<float>();
            memcpy(data, M.data(), M.cols() * M.rows() * sizeof(float));
        }

        void get_eigen_matrix_from_torch_tensor(const torch::Tensor &T, MatrixXf_rm &M) const {
            if (T.dim() == 0) {
                M = MatrixXf_rm(1, 1); //scalar
                float *data = T.data_ptr<float>();
                M = Eigen::Map<MatrixXf_rm>(data, 1, 1);
            } else {
                size_t total_size_individual_tensor = 1;
                for (size_t dim = 1; dim < T.dim(); ++dim) {
                    total_size_individual_tensor *= T.size(dim);
                }
                M = MatrixXf_rm(T.size(0), total_size_individual_tensor);
                float *data = T.data_ptr<float>();
                M = Eigen::Map<MatrixXf_rm>(data, T.size(0), total_size_individual_tensor);
            }
        }
    };

    template <typename TParams, typename Exact = stc::Itself>
    class NetworkLoaderLSTM : public AbstractLoader<TParams, typename stc::FindExact<NetworkLoaderLSTM<TParams, Exact>, Exact>::ret> {
    public:
        typedef AbstractLoader<TParams, typename stc::FindExact<NetworkLoaderLSTM<TParams, Exact>, Exact>::ret> TParentLoader;

        explicit NetworkLoaderLSTM() : TParentLoader(TParams::lstm::latent_size_per_layer * TParams::lstm::number_layers,
                                                     torch::nn::AnyModule(aurora::nn::LSTMAutoencoderImpl(TParams::get_one_obs_size(), TParams::lstm::latent_size_per_layer, TParams::lstm::number_layers)))
                                       {

            if (this->m_use_colors) {
                std::cout << "Using COLORS" << std::endl;
            } else {
                std::cout << "Using GRAYSCALE Images" << std::endl;
            }
        }

        typedef Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> MatrixXf_rm;

        void prepare_batches(std::vector<torch::Tensor> &batches, const MatrixXf_rm &data) const {
            /*
             * Generate all the batches for training
             * */
            if (data.rows() <= TParentLoader::Options::batch_size) {
                batches = std::vector<torch::Tensor>(1);
            } else {
                batches = std::vector<torch::Tensor>(floor(data.rows() / (TParentLoader::Options::batch_size)));
            }

            if (batches.size() == 1) {
                this->get_torch_tensor_from_eigen_matrix(data, batches[0]);
                batches[0] = batches[0].view({-1, static_cast<int>(data.cols() / this->m_one_obs_size), this->m_one_obs_size}).to(this->m_device);
            } else {
                for (size_t ind = 0; ind < batches.size(); ind++) {
                    this->get_torch_tensor_from_eigen_matrix(data.middleRows(ind * TParentLoader::Options::batch_size, TParentLoader::Options::batch_size),
                                                             batches[ind]);
                    batches[ind] = batches[ind].view({-1, static_cast<int>(data.cols() / this->m_one_obs_size), this->m_one_obs_size}).to(this->m_device);
                }
            }

        }



        void eval(const MatrixXf_rm &data,
                  MatrixXf_rm &descriptors,
                  MatrixXf_rm &recon_loss,
                  MatrixXf_rm &reconstructed_data) {
            
            torch::NoGradGuard no_grad;

            aurora::nn::LSTMAutoencoder auto_encoder = std::static_pointer_cast<aurora::nn::LSTMAutoencoderImpl>(this->m_auto_encoder_module.ptr());

            torch::Tensor eval_data;
            this->get_torch_tensor_from_eigen_matrix(data, eval_data);

            std::vector<torch::Tensor> outputs;


            // Get the names below with the inspect_graph.py script applied on the generated graph_text.pb file.
            eval_data = torch::reshape(eval_data, {-1, static_cast<int>(data.cols() / this->m_one_obs_size), this->m_one_obs_size}); //TODO : CHANGE THIS
            torch::Tensor descriptors_tensor;
            torch::Tensor reconstruction_tensor;
            std::vector<torch::Tensor> v_loss_tensor;
            std::vector<torch::Tensor> v_descriptors_tensor;
            std::vector<torch::Tensor> v_reconstruction_tensor;

            for (long index = 0; index * TParentLoader::Options::batch_size < eval_data.size(0); ++index) {
                const torch::Tensor& c_slice_eval_data = eval_data.slice(0, index * TParentLoader::Options::batch_size, std::min((index + 1) * TParentLoader::Options::batch_size, eval_data.size(0))).to(this->m_device);
                reconstruction_tensor = auto_encoder->forward_get_latent(c_slice_eval_data, descriptors_tensor).to(this->m_device);
                v_loss_tensor.push_back(torch::norm((reconstruction_tensor - c_slice_eval_data), 2, {1, 2}).cpu());
                v_descriptors_tensor.push_back(descriptors_tensor.cpu());
                v_reconstruction_tensor.push_back(reconstruction_tensor.cpu());
            }
            torch::TensorList tensor_list_loss_tensor{v_loss_tensor};
            torch::TensorList tensor_list_descriptors_tensor{v_descriptors_tensor};
            torch::TensorList tensor_list_reconstruction_tensor{v_reconstruction_tensor};

            torch::Tensor loss_tensor = torch::cat(tensor_list_loss_tensor);
            descriptors_tensor = torch::cat(tensor_list_descriptors_tensor);
            reconstruction_tensor = torch::cat(tensor_list_reconstruction_tensor);
            //std::cout << "eval (reconstruction tensor sizes) - " << reconstruction_tensor.sizes() << std::endl;
            // TODO put those lines in another function
            // std::cout << descriptors_tensor.size(0) << " " << descriptors_tensor.size(1) << std::endl;
            this->get_eigen_matrix_from_torch_tensor(descriptors_tensor.cpu(), descriptors);
            this->get_eigen_matrix_from_torch_tensor(loss_tensor.cpu(), recon_loss);
            // TODO To avoid next line if not needed
            this->get_eigen_matrix_from_torch_tensor(reconstruction_tensor.cpu(), reconstructed_data);
        }
    };

    template <typename TParams, typename Exact = stc::Itself>
    class NetworkLoaderVideoAutoEncoder : public AbstractLoader<TParams, typename stc::FindExact<NetworkLoaderVideoAutoEncoder<TParams, Exact>, Exact>::ret> {
    public:
        typedef NetworkLoaderVideoAutoEncoder<TParams, typename stc::FindExact<NetworkLoaderVideoAutoEncoder<TParams, Exact>, Exact>::ret> TParentLoader;

        explicit NetworkLoaderVideoAutoEncoder() : TParentLoader(TParams::lstm::latent_size_per_layer * TParams::lstm::number_layers,
                                                     torch::nn::AnyModule(
                                                             aurora::nn::VideoAutoEncoderImpl(
                                                             32,
                                                             32,
                                                             TParams::latent_size_cnn_ae,
                                                             TParams::use_colors,
                                                             TParams::lstm::latent_size_per_layer,
                                                             TParams::lstm::number_layers)))
        {

            if (this->m_use_colors) {
                std::cout << "Using COLORS" << std::endl;
            } else {
                std::cout << "Using GRAYSCALE Images" << std::endl;
            }
        }

        typedef Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> MatrixXf_rm;

        void prepare_batches(std::vector<torch::Tensor> &batches, const MatrixXf_rm &data) const {
            /*
             * Generate all the batches for training
             * */
            if (data.rows() <= TParentLoader::Options::batch_size) {
                batches = std::vector<torch::Tensor>(1);
            } else {
                batches = std::vector<torch::Tensor>(floor(data.rows() / (TParentLoader::Options::batch_size)));
            }

            const int c_number_channels = this->m_use_colors ? 3 : 1;

            if (batches.size() == 1) {
                this->get_torch_tensor_from_eigen_matrix(data, batches[0]);
                batches[0] = torch::upsample_bilinear2d(
                        batches[0].view({-1, c_number_channels, TParams::image_height, TParams::image_width}).to(this->m_device),
                        {32, 32}, false);
                batches[0] = batches[0].view({-1, static_cast<int>(data.cols() / this->m_one_obs_size), c_number_channels, 32, 32});

            } else {
                for (size_t ind = 0; ind < batches.size(); ind++) {
                    this->get_torch_tensor_from_eigen_matrix(data.middleRows(ind * TParentLoader::Options::batch_size, TParentLoader::Options::batch_size),
                                                             batches[ind]);
                    batches[ind] = torch::upsample_bilinear2d(
                            batches[ind].view({-1, c_number_channels, TParams::image_height, TParams::image_width}).to(this->m_device),
                            {32, 32}, false);
                    batches[ind] = batches[ind].view({-1, static_cast<int>(data.cols() / this->m_one_obs_size), c_number_channels, 32, 32});
                }
            }
        }

        void eval(const MatrixXf_rm &data,
                  MatrixXf_rm &descriptors,
                  MatrixXf_rm &recon_loss,
                  MatrixXf_rm &reconstructed_data) {
            torch::NoGradGuard no_grad;


            aurora::nn::VideoAutoEncoder auto_encoder = std::static_pointer_cast<aurora::nn::VideoAutoEncoderImpl>(this->m_auto_encoder_module.ptr());

            torch::Tensor eval_data;
            this->get_torch_tensor_from_eigen_matrix(data, eval_data);

            std::vector<torch::Tensor> outputs;

            const int c_number_channels = this->m_use_colors ? 3 : 1;

            // Get the names below with the inspect_graph.py script applied on the generated graph_text.pb file.
            eval_data = torch::upsample_bilinear2d(torch::reshape(eval_data, {-1, c_number_channels, TParams::image_height, TParams::image_width}).to(this->m_device), {32, 32}, false);
            eval_data = torch::reshape(eval_data, {-1, static_cast<int>(data.cols() / this->m_one_obs_size), c_number_channels, TParams::image_height, TParams::image_width}); // TODO : CHANGE THIS

            torch::Tensor descriptors_tensor;
            torch::Tensor reconstruction_tensor = auto_encoder->forward_get_latent(eval_data, descriptors_tensor);
            torch::Tensor loss_tensor{torch::norm(reconstruction_tensor - eval_data, 2, {1, 2, 3, 4})};

            //std::cout << "eval (reconstruction tensor sizes) - " << reconstruction_tensor.sizes() << std::endl;
            // TODO put those lines in another function
            // std::cout << descriptors_tensor.size(0) << " " << descriptors_tensor.size(1) << std::endl;
            this->get_eigen_matrix_from_torch_tensor(descriptors_tensor.cpu(), descriptors);
            this->get_eigen_matrix_from_torch_tensor(loss_tensor.cpu(), recon_loss);
            // TODO To avoid next line if not needed
            this->get_eigen_matrix_from_torch_tensor(reconstruction_tensor.cpu(), reconstructed_data);
        }
    };

    template <typename TParams, typename Exact = stc::Itself>
    class NetworkLoaderAutoEncoder : public AbstractLoader<TParams, typename stc::FindExact<NetworkLoaderAutoEncoder<TParams, Exact>, Exact>::ret> {
    public:
        typedef AbstractLoader<TParams, typename stc::FindExact<NetworkLoaderAutoEncoder<TParams, Exact>, Exact>::ret> TParentLoader;

        // IF aurora::get_encoder_type() == EncoderType::strg_cnn
        // -> using strong autoencoder
        explicit NetworkLoaderAutoEncoder() :
                TParentLoader(TParams::qd::behav_dim,
                              torch::nn::AnyModule(aurora::nn::AutoEncoder(32, 32, TParams::qd::behav_dim, TParams::use_colors, aurora::get_encoder_type() == EncoderType::strg_cnn)))
        {

            if (this->m_use_colors) {
                std::cout << "Using COLORS" << std::endl;
            } else {
                std::cout << "Using GRAYSCALE Images" << std::endl;
            }
        }

        typedef Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> MatrixXf_rm;


        void prepare_batches(std::vector<torch::Tensor> &batches, const MatrixXf_rm &data) const {
            /*
             * Generate all the batches for training
             * */
            if (data.rows() <= TParentLoader::Options::batch_size) {
                batches = std::vector<torch::Tensor>(1);
            } else {
                batches = std::vector<torch::Tensor>(floor(data.rows() / (TParentLoader::Options::batch_size)));
            }

            if (batches.size() == 1) {
                this->get_torch_tensor_from_eigen_matrix(data, batches[0]);
                if (this->m_use_colors) {
                    batches[0] = torch::upsample_bilinear2d(batches[0].view({-1, 3, TParams::image_height, TParams::image_width}).to(this->m_device), {32, 32}, false);
                } else {
                    batches[0] = torch::upsample_bilinear2d(batches[0].view({-1, 1, TParams::image_height, TParams::image_width}).to(this->m_device), {32, 32}, false);
                }
            } else {
                for (size_t ind = 0; ind < batches.size(); ind++) {
                    this->get_torch_tensor_from_eigen_matrix(data.middleRows(ind * TParentLoader::Options::batch_size, TParentLoader::Options::batch_size),
                                                             batches[ind]);
                    if (this->m_use_colors) {
                        batches[ind] = torch::upsample_bilinear2d(batches[ind].view({-1, 3, TParams::image_height, TParams::image_width}).to(this->m_device), {32, 32}, false);
                    } else {
                        batches[ind] = torch::upsample_bilinear2d(batches[ind].view({-1, 1, TParams::image_height, TParams::image_width}).to(this->m_device), {32, 32}, false);
                    }
                }
            }

        }



        void eval(const MatrixXf_rm &data,
                  MatrixXf_rm &descriptors,
                  MatrixXf_rm &recon_loss,
                  MatrixXf_rm &reconstructed_data) {
            torch::NoGradGuard no_grad;

            aurora::nn::AutoEncoder auto_encoder = std::static_pointer_cast<aurora::nn::AutoEncoderImpl>(this->m_auto_encoder_module.ptr());

            torch::Tensor eval_data;
            this->get_torch_tensor_from_eigen_matrix(data, eval_data);

            std::vector<torch::Tensor> outputs;

            // Get the names below with the inspect_graph.py script applied on the generated graph_text.pb file.
            if (this->m_use_colors) {
                eval_data = torch::upsample_bilinear2d(torch::reshape(eval_data, {-1, 3, TParams::image_height, TParams::image_width}).to(this->m_device), {32, 32}, false);
            } else {
                eval_data = torch::upsample_bilinear2d(torch::reshape(eval_data, {-1, 1, TParams::image_height, TParams::image_width}).to(this->m_device), {32, 32}, false);
            }

            torch::Tensor descriptors_tensor;
            torch::Tensor reconstruction_tensor = auto_encoder->forward_get_latent(eval_data, descriptors_tensor);
            torch::Tensor loss_tensor{torch::norm(reconstruction_tensor - eval_data, 2, {1, 2, 3})};

            //std::cout << "eval (reconstruction tensor sizes) - " << reconstruction_tensor.sizes() << std::endl;
            // TODO put those lines in another function
            this->get_eigen_matrix_from_torch_tensor(descriptors_tensor.cpu(), descriptors);
            this->get_eigen_matrix_from_torch_tensor(loss_tensor.cpu(), recon_loss);
            // TODO To avoid next line if not needed
            this->get_eigen_matrix_from_torch_tensor(torch::upsample_bilinear2d(reconstruction_tensor.cpu(), {TParams::image_height , TParams::image_width}, false), reconstructed_data);
            c10::cuda::CUDACachingAllocator::emptyCache();
        }
    };

    template <typename TParams, typename Exact = stc::Itself>
    class NetworkLoaderConvolutionalSequenceAutoencoder : public AbstractLoader<TParams, typename stc::FindExact<NetworkLoaderConvolutionalSequenceAutoencoder<TParams, Exact>, Exact>::ret> {
    public:
        typedef AbstractLoader<TParams, typename stc::FindExact<NetworkLoaderConvolutionalSequenceAutoencoder<TParams, Exact>, Exact>::ret> TParentLoader;

        explicit NetworkLoaderConvolutionalSequenceAutoencoder() : TParentLoader(TParams::qd::behav_dim,
                                                     torch::nn::AnyModule(aurora::nn::ConvolutionalSequenceAutoencoderImpl(TParams::qd::behav_dim)))
        {

            if (this->m_use_colors) {
                std::cout << "Using COLORS" << std::endl;
            } else {
                std::cout << "Using GRAYSCALE Images" << std::endl;
            }
        }

        typedef Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> MatrixXf_rm;

        void prepare_batches(std::vector<torch::Tensor> &batches, const MatrixXf_rm &data) const {
            /*
             * Generate all the batches for training
             * */
            if (data.rows() <= TParentLoader::Options::batch_size) {
                batches = std::vector<torch::Tensor>(1);
            } else {
                batches = std::vector<torch::Tensor>(floor(data.rows() / (TParentLoader::Options::batch_size)));
            }

            if (batches.size() == 1) {
                this->get_torch_tensor_from_eigen_matrix(data, batches[0]);
                batches[0] = batches[0].view({-1, static_cast<int>(data.cols() / this->m_one_obs_size), this->m_one_obs_size}).permute({0, 2, 1}).to(this->m_device);
            } else {
                for (size_t ind = 0; ind < batches.size(); ind++) {
                    this->get_torch_tensor_from_eigen_matrix(data.middleRows(ind * TParentLoader::Options::batch_size, TParentLoader::Options::batch_size),
                                                             batches[ind]);
                    batches[ind] = batches[ind].view({-1, static_cast<int>(data.cols() / this->m_one_obs_size), this->m_one_obs_size}).permute({0, 2, 1}).to(this->m_device);
                }
            }

        }



        void eval(const MatrixXf_rm &data,
                  MatrixXf_rm &descriptors,
                  MatrixXf_rm &recon_loss,
                  MatrixXf_rm &reconstructed_data) {
            torch::NoGradGuard no_grad;

            aurora::nn::ConvolutionalSequenceAutoencoder auto_encoder = std::static_pointer_cast<aurora::nn::ConvolutionalSequenceAutoencoderImpl>(this->m_auto_encoder_module.ptr());

            torch::Tensor eval_data;
            this->get_torch_tensor_from_eigen_matrix(data, eval_data);

            std::vector<torch::Tensor> outputs;


            // Get the names below with the inspect_graph.py script applied on the generated graph_text.pb file.
            eval_data = torch::reshape(eval_data, {-1, static_cast<int>(data.cols() / this->m_one_obs_size), this->m_one_obs_size}).permute({0, 2, 1}); //TODO : CHANGE THIS
            torch::Tensor descriptors_tensor;
            torch::Tensor reconstruction_tensor;
            std::vector<torch::Tensor> v_loss_tensor;
            std::vector<torch::Tensor> v_descriptors_tensor;
            std::vector<torch::Tensor> v_reconstruction_tensor;
            for (long index = 0; index * TParentLoader::Options::batch_size < eval_data.size(0); ++index) {
                const torch::Tensor& c_slice_eval_data = eval_data.slice(0, index * TParentLoader::Options::batch_size, std::min((index + 1) * TParentLoader::Options::batch_size, eval_data.size(0))).to(this->m_device);
                reconstruction_tensor = auto_encoder->forward_get_latent(c_slice_eval_data, descriptors_tensor).to(this->m_device);
                v_loss_tensor.push_back(torch::norm((reconstruction_tensor - c_slice_eval_data), 2, {1, 2}).cpu());
                v_descriptors_tensor.push_back(descriptors_tensor.cpu());
                v_reconstruction_tensor.push_back(reconstruction_tensor.permute({0, 2, 1}).cpu());
            }
            torch::TensorList tensor_list_loss_tensor{v_loss_tensor};
            torch::TensorList tensor_list_descriptors_tensor{v_descriptors_tensor};
            torch::TensorList tensor_list_reconstruction_tensor{v_reconstruction_tensor};

            torch::Tensor loss_tensor = torch::cat(tensor_list_loss_tensor);
            descriptors_tensor = torch::cat(tensor_list_descriptors_tensor);
            reconstruction_tensor = torch::cat(tensor_list_reconstruction_tensor);
            //std::cout << "eval (reconstruction tensor sizes) - " << reconstruction_tensor.sizes() << std::endl;
            // TODO put those lines in another function
            // std::cout << descriptors_tensor.size(0) << " " << descriptors_tensor.size(1) << std::endl;
            this->get_eigen_matrix_from_torch_tensor(descriptors_tensor.cpu(), descriptors);
            this->get_eigen_matrix_from_torch_tensor(loss_tensor.cpu(), recon_loss);
            // TODO To avoid next line if not needed
            this->get_eigen_matrix_from_torch_tensor(reconstruction_tensor.cpu(), reconstructed_data);
        }
    };

  template <typename TParams, typename Exact = stc::Itself>
  class NetworkLoaderMLPAutoEncoder : public AbstractLoader<TParams, typename stc::FindExact<NetworkLoaderMLPAutoEncoder<TParams, Exact>, Exact>::ret> {
  public:
    typedef AbstractLoader<TParams, typename stc::FindExact<NetworkLoaderMLPAutoEncoder<TParams, Exact>, Exact>::ret> TParentLoader;

    // IF aurora::get_encoder_type() == EncoderType::strg_cnn
    // -> using strong autoencoder
    explicit NetworkLoaderMLPAutoEncoder() :
      TParentLoader(TParams::qd::behav_dim,
                    torch::nn::AnyModule(aurora::nn::MLPAutoEncoder(TParams::qd::behav_dim, 100)))
    {}

    typedef Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> MatrixXf_rm;


    void prepare_batches(std::vector<torch::Tensor> &batches, const MatrixXf_rm &data) const {
      /*
       * Generate all the batches for training
       * */
      if (data.rows() <= TParentLoader::Options::batch_size) {
        batches = std::vector<torch::Tensor>(1);
      } else {
        batches = std::vector<torch::Tensor>(floor(data.rows() / (TParentLoader::Options::batch_size)));
      }

      if (batches.size() == 1) {
        this->get_torch_tensor_from_eigen_matrix(data, batches[0]);
        batches[0] = batches[0].to(this->m_device);
      } else {
        for (size_t ind = 0; ind < batches.size(); ind++) {
          this->get_torch_tensor_from_eigen_matrix(data.middleRows(ind * TParentLoader::Options::batch_size, TParentLoader::Options::batch_size),
                                                   batches[ind]);
          batches[ind] = batches[ind].to(this->m_device);
        }
      }
    }


    void eval(const MatrixXf_rm &data,
              MatrixXf_rm &descriptors,
              MatrixXf_rm &recon_loss,
              MatrixXf_rm &reconstructed_data) {
      torch::NoGradGuard no_grad;

      aurora::nn::MLPAutoEncoder auto_encoder = std::static_pointer_cast<aurora::nn::MLPAutoEncoderImpl>(this->m_auto_encoder_module.ptr());

      torch::Tensor eval_data;
      this->get_torch_tensor_from_eigen_matrix(data, eval_data);

      eval_data = eval_data.to(this->m_device);

      std::vector<torch::Tensor> outputs;

      torch::Tensor descriptors_tensor;
      torch::Tensor reconstruction_tensor = auto_encoder->forward_get_latent(eval_data, descriptors_tensor);
      torch::Tensor loss_tensor{torch::norm(reconstruction_tensor - eval_data, 2, {1})};

      //std::cout << "eval (reconstruction tensor sizes) - " << reconstruction_tensor.sizes() << std::endl;
      // TODO put those lines in another function
      this->get_eigen_matrix_from_torch_tensor(descriptors_tensor.cpu(), descriptors);
      this->get_eigen_matrix_from_torch_tensor(loss_tensor.cpu(), recon_loss);
      // TODO To avoid next line if not needed
      this->get_eigen_matrix_from_torch_tensor(reconstruction_tensor.cpu(), reconstructed_data);
      c10::cuda::CUDACachingAllocator::emptyCache();
    }
  };
}




#endif

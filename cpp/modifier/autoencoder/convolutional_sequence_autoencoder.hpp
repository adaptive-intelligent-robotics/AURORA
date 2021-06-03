//
// Created by Luca Grillotti on 22/06/2020.
//

#ifndef AURORA_CONVOLUTIONAL_SEQUENCE_AUTOENCODER_HPP
#define AURORA_CONVOLUTIONAL_SEQUENCE_AUTOENCODER_HPP

#include <torch/torch.h>

namespace aurora {
    namespace nn {
        struct ConvolutionalSequenceAutoencoderImpl : torch::nn::Module {

            explicit ConvolutionalSequenceAutoencoderImpl(int latent_dim)  {
                m_conv_1 = torch::nn::Conv1d(torch::nn::Conv1dOptions(3, 4, 3).padding(1));
                m_conv_2 = torch::nn::Conv1d(torch::nn::Conv1dOptions(4, 4, 3).padding(1));
                m_conv_3 = torch::nn::Conv1d(torch::nn::Conv1dOptions(4, 4, 3).padding(1));

                m_max_pool_1 = torch::nn::AvgPool1d(2);
                m_max_pool_2 = torch::nn::AvgPool1d(2);
                m_max_pool_3 = torch::nn::AvgPool1d(2);

                m_linear_enc_1 = torch::nn::Linear(100, 32);
                m_linear_enc_2 = torch::nn::Linear(32, latent_dim);

                m_linear_dec_1 = torch::nn::Linear(latent_dim, 32);
                m_linear_dec_2 = torch::nn::Linear(32, 100);

                m_deconv_1 = torch::nn::Conv1d(torch::nn::Conv1dOptions(4, 4, 3).padding(1));
                m_deconv_2 = torch::nn::Conv1d(torch::nn::Conv1dOptions(4, 4, 3).padding(1));
                m_deconv_3 = torch::nn::Conv1d(torch::nn::Conv1dOptions(4, 4, 3).padding(1));
                m_deconv_4 = torch::nn::Conv1d(torch::nn::Conv1dOptions(4, 3, 3).padding(1));

                register_module("linear_enc_1", m_linear_enc_1);
                register_module("linear_enc_2", m_linear_enc_2);

                register_module("linear_dec_1", m_linear_dec_1);
                register_module("linear_dec_2", m_linear_dec_2);

                register_module("conv_1", m_conv_1);
                register_module("conv_2", m_conv_2);
                register_module("conv_3", m_conv_3);

                register_module("m_max_pool_1", m_max_pool_1);
                register_module("m_max_pool_2", m_max_pool_2);
                register_module("m_max_pool_3", m_max_pool_3);

                register_module("deconv_1", m_deconv_1);
                register_module("deconv_2", m_deconv_2);
                register_module("deconv_3", m_deconv_3);
                register_module("deconv_4", m_deconv_4);

            }

            torch::Tensor forward(const torch::Tensor &x)  {
                torch::Tensor output;
                output = m_max_pool_1(torch::relu(m_conv_1(x)));
                output = m_max_pool_2(torch::relu(m_conv_2(output)));
                output = m_max_pool_3(torch::relu(m_conv_3(output)));
                output = torch::reshape(output, {x.size(0), 100});

                output = torch::relu(m_linear_enc_1(output));
                output = m_linear_enc_2(output);

                output = torch::relu(m_linear_dec_1(output));
                output = torch::relu(m_linear_dec_2(output));

                output = torch::reshape(output, {-1, 4, 25});
                output = torch::upsample_nearest1d(output, {50});
                output = torch::relu(m_deconv_1(output));
                output = torch::upsample_nearest1d(output, {100});
                output = torch::relu(m_deconv_2(output));
                output = torch::upsample_nearest1d(output, {200});
                output = torch::relu(m_deconv_3(output));
                output = m_deconv_4(output);
                return output;
            }

            torch::Tensor forward_get_latent(const torch::Tensor &x, torch::Tensor& latent)  {
                torch::Tensor output;
                output = m_max_pool_1(torch::relu(m_conv_1(x)));
                output = m_max_pool_2(torch::relu(m_conv_2(output)));
                output = m_max_pool_3(torch::relu(m_conv_3(output)));
                output = torch::reshape(output, {x.size(0), 100});

                output = torch::relu(m_linear_enc_1(output));
                latent = m_linear_enc_2(output);

                output = torch::relu(m_linear_dec_1(latent));
                output = torch::relu(m_linear_dec_2(output));

                output = torch::reshape(output, {-1, 4, 25});
                output = torch::upsample_nearest1d(output, {50});
                output = torch::relu(m_deconv_1(output));
                output = torch::upsample_nearest1d(output, {100});
                output = torch::relu(m_deconv_2(output));
                output = torch::upsample_nearest1d(output, {200});
                output = torch::relu(m_deconv_3(output));
                output = m_deconv_4(output);
                return output;
            }

            torch::nn::Linear m_linear_enc_1{nullptr}, m_linear_enc_2{nullptr}, m_linear_dec_1{nullptr}, m_linear_dec_2{nullptr};
            torch::nn::AvgPool1d m_max_pool_1{nullptr}, m_max_pool_2{nullptr}, m_max_pool_3{nullptr};
            torch::nn::Conv1d m_deconv_1{nullptr}, m_deconv_2{nullptr}, m_deconv_3{nullptr}, m_deconv_4{nullptr};
            torch::nn::Conv1d m_conv_1{nullptr}, m_conv_2{nullptr}, m_conv_3{nullptr};
        };

        TORCH_MODULE(ConvolutionalSequenceAutoencoder);
    }
}



#endif //AURORA_CONVOLUTIONAL_SEQUENCE_AUTOENCODER_HPP

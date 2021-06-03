//
// Created by Luca Grillotti
//

#ifndef AURORA_ENCODER_HPP
#define AURORA_ENCODER_HPP

#include <torch/torch.h>

#include "utils_autoencoder.hpp"

/* TODO add padding to get a same padding */

namespace aurora {
    namespace nn {
        struct EncoderImpl : torch::nn::Module {
            EncoderImpl(int image_width, int image_height, int latent_dim, bool use_colors=false, bool strong=false) :
                    m_conv_1(torch::nn::Conv2d(
                            torch::nn::Conv2dOptions(use_colors ? 3 : 1, strong ? 16 : 4, 3)
                                    .padding(1))),
                    m_conv_2(torch::nn::Conv2d(
                            torch::nn::Conv2dOptions(strong ? 16 : 4, strong ? 16 : 4, 3)
                                    .padding(1))),
                    m_conv_3(torch::nn::Conv2d(
                            torch::nn::Conv2dOptions(strong ? 16 : 4, strong ? 16 : 4, 3)
                                    .padding(1))),

                    m_max_pool_1(torch::nn::MaxPool2d(2)),
                    m_max_pool_2(torch::nn::MaxPool2d(2)),
                    m_max_pool_3(torch::nn::MaxPool2d(2)),
                    m_linear_1(torch::nn::Linear((strong ? 16 : 4) * 4 * 4, (strong ? 64 : 32))),
                    m_linear_latent(torch::nn::Linear((strong ? 64 : 32), latent_dim)) {
                register_module("conv_1", m_conv_1);
                register_module("conv_2", m_conv_2);
                register_module("conv_3", m_conv_3);
                register_module("max_pool_1", m_max_pool_1);
                register_module("max_pool_2", m_max_pool_2);
                register_module("max_pool_3", m_max_pool_3);
                register_module("flatten", m_flatten);
                register_module("linear_1", m_linear_1);
                register_module("linear_latent", m_linear_latent);
            }

            torch::Tensor forward(const torch::Tensor &x) {
                torch::Tensor output;
                output = m_max_pool_1(torch::relu(m_conv_1(x)));
                output = m_max_pool_2(torch::relu(m_conv_2(output)));
                output = m_max_pool_3(torch::relu(m_conv_3(output)));
                output = m_flatten(output);
                output = torch::relu(m_linear_1(output));
                output = m_linear_latent(output);
                return output;
            }

            torch::nn::Conv2d m_conv_1, m_conv_2, m_conv_3;
            torch::nn::MaxPool2d m_max_pool_1, m_max_pool_2, m_max_pool_3;
            torch::nn::Linear m_linear_1, m_linear_latent;
            aurora::nn::Flatten m_flatten;
        };

        TORCH_MODULE(Encoder);

    }
}

#endif //AURORA_ENCODER_HPP

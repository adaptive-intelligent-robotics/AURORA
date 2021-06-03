//
// Created by Luca Grillotti
//

#ifndef AURORA_AUTOENCODER_HPP
#define AURORA_AUTOENCODER_HPP

#include <torch/torch.h>

#include "encoder.hpp"
#include "decoder.hpp"

namespace aurora {
    namespace nn {
        struct AutoEncoderImpl : torch::nn::Module {
            AutoEncoderImpl(int image_width, int image_height, int latent_dim, bool use_colors = false, bool strong = false) :
                    m_encoder(aurora::nn::Encoder(image_width, image_height, latent_dim, use_colors, strong)),
                    m_decoder(aurora::nn::Decoder(image_width, image_height, latent_dim, use_colors, strong)) {
                register_module("encoder", m_encoder);
                register_module("decoder", m_decoder);
            }

            torch::Tensor forward(const torch::Tensor &x) {
                return m_decoder(m_encoder(x));
            }

            torch::Tensor forward_get_latent(const torch::Tensor &input, torch::Tensor &corresponding_latent) {
                corresponding_latent = m_encoder(input);
                return m_decoder(corresponding_latent);
            }

            aurora::nn::Encoder m_encoder;
            aurora::nn::Decoder m_decoder;
        };

        TORCH_MODULE(AutoEncoder);
    }
}

#endif //AURORA_AUTOENCODER_HPP

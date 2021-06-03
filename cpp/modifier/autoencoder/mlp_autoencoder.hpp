//
// Created by Luca Grillotti on 21/02/2021.
//

#ifndef AURORA_MLP_AUTOENCODER_HPP
#define AURORA_MLP_AUTOENCODER_HPP

#include <torch/torch.h>

namespace aurora {
  namespace nn {
    struct MLPEncoderImpl : torch::nn::Module {
      MLPEncoderImpl(int latent_dim, int sequence_length) {
        m_linear_enc_1 = torch::nn::Linear(sequence_length, 32);
        m_linear_enc_2 = torch::nn::Linear(32, 8);
        m_linear_enc_3 = torch::nn::Linear(8, latent_dim);

        register_module("linear_enc_1", m_linear_enc_1);
        register_module("linear_enc_2", m_linear_enc_2);
        register_module("linear_enc_3", m_linear_enc_3);
      }

      torch::Tensor forward(const torch::Tensor &x)  {
        torch::Tensor output;
        output = torch::relu(m_linear_enc_1(x));
        output = torch::relu(m_linear_enc_2(output));
        output = m_linear_enc_3(output);

        return output;
      }

      torch::nn::Linear m_linear_enc_1{nullptr}, m_linear_enc_2{nullptr}, m_linear_enc_3{nullptr};
    };

    TORCH_MODULE(MLPEncoder);


    struct MLPDecoderImpl : torch::nn::Module {

      MLPDecoderImpl(int latent_dim, int sequence_length)  {

        m_linear_dec_1 = torch::nn::Linear(latent_dim, 8);
        m_linear_dec_2 = torch::nn::Linear(8, 32);
        m_linear_dec_3 = torch::nn::Linear(32, sequence_length);

        register_module("linear_dec_1", m_linear_dec_1);
        register_module("linear_dec_2", m_linear_dec_2);
        register_module("linear_dec_3", m_linear_dec_3);
      }

      torch::Tensor forward(const torch::Tensor &x)  {
        torch::Tensor output;

        output = torch::relu(m_linear_dec_1(x));
        output = torch::relu(m_linear_dec_2(output));
        output = m_linear_dec_3(output);

        return output;
      }

      torch::nn::Linear m_linear_dec_1{nullptr}, m_linear_dec_2{nullptr}, m_linear_dec_3{nullptr};
    };

    TORCH_MODULE(MLPDecoder);

    struct MLPAutoEncoderImpl : torch::nn::Module {

      MLPAutoEncoderImpl(int latent_dim, int sequence_length)  {

        m_encoder = MLPEncoder(latent_dim, sequence_length);
        m_decoder = MLPDecoder(latent_dim, sequence_length);

        register_module("encoder", m_encoder);
        register_module("decoder", m_decoder);
      }

      torch::Tensor forward(const torch::Tensor &x)  {
        return m_decoder(m_encoder(x));
      }

      torch::Tensor forward_get_latent(const torch::Tensor &input, torch::Tensor &corresponding_latent) {
        corresponding_latent = m_encoder(input);
        return m_decoder(corresponding_latent);
      }

      MLPEncoder m_encoder{nullptr};
      MLPDecoder m_decoder{nullptr};
    };

    TORCH_MODULE(MLPAutoEncoder);
  }
}



#endif // AURORA_MLP_AUTOENCODER_HPP

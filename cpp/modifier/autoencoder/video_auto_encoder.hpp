//
// Created by Luca Grillotti on 30/05/2020.
//

#ifndef AURORA_VIDEO_AUTO_ENCODER_HPP
#define AURORA_VIDEO_AUTO_ENCODER_HPP

#include <torch/torch.h>

#include "lstm_auto_encoder.hpp"
#include "encoder.hpp"
#include "decoder.hpp"

namespace aurora {
    namespace nn {
        struct VideoAutoEncoderImpl : torch::nn::Module {

            VideoAutoEncoderImpl(int image_width, int image_height, int latent_dim_images, bool use_colors,
                                 size_t latent_space_per_layer, size_t number_layers)  {
                m_cnn_encoder = Encoder(image_width, image_height, latent_dim_images, use_colors);
                m_lstm_encoder = LSTMEncoder(static_cast<size_t>(latent_dim_images), latent_space_per_layer, number_layers);
                m_lstm_decoder = LSTMDecoder(static_cast<size_t>(latent_dim_images), latent_space_per_layer, number_layers);
                m_cnn_decoder = Decoder(image_width, image_height, latent_dim_images, use_colors);

                register_module("ImageEncoder", m_cnn_encoder);
                register_module("LSTMEncoder", m_lstm_encoder);
                register_module("LSTMDecoder", m_lstm_decoder);
                register_module("ImageDecoder", m_cnn_decoder);
            }

            torch::Tensor forward(const torch::Tensor &x)  { // shape = {num_batches, nb_in_seq, number_chanels, w, h}
                torch::Tensor temp = x.reshape({-1, x.size(2), x.size(3), x.size(4)});

                temp = m_cnn_encoder->forward(temp).reshape(
                        {x.size(0), x.size(1), m_lstm_encoder->m_lstm->options.input_size()}
                );

                temp = m_lstm_decoder->forward(temp, m_lstm_encoder->forward(temp)).reshape({
                                                                                                    x.size(0) * x.size(1), m_lstm_encoder->m_lstm->options.input_size()
                                                                                            });

                temp = m_cnn_decoder->forward(temp).reshape(x.sizes()); // Putting it back at x shape

                return temp;
            }

            torch::Tensor forward_get_latent(const torch::Tensor &x, torch::Tensor &latent) {
                torch::Tensor encoding = m_lstm_encoder->forward(m_cnn_encoder->forward(x));
                latent = encoding.reshape({x.size(0), m_lstm_encoder->m_lstm->options.hidden_size()}); // shape = {batch_size, hidden_size}
                return m_cnn_decoder->forward(m_lstm_decoder->forward(x, encoding));
            }

            Encoder m_cnn_encoder{nullptr};
            LSTMEncoder m_lstm_encoder{nullptr};
            LSTMDecoder m_lstm_decoder{nullptr};
            Decoder m_cnn_decoder{nullptr};
        };

        TORCH_MODULE(VideoAutoEncoder);
    }
}

#endif //AURORA_VIDEO_AUTO_ENCODER_HPP

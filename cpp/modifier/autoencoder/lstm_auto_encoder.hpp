//
// Created by Luca Grillotti on 20/05/2020.
//

#ifndef AURORA_LSTM_AUTO_ENCODER_HPP
#define AURORA_LSTM_AUTO_ENCODER_HPP

#include <torch/torch.h>

namespace aurora {
    namespace nn {
        struct LSTMEncoderImpl : torch::nn::Module {
          typedef std::tuple<torch::Tensor, std::tuple<torch::Tensor, torch::Tensor>> RNNOutput;

            LSTMEncoderImpl(size_t input_size, size_t latent_space_per_layer, size_t number_layers) {
                torch::nn::LSTMOptions lstm_options(input_size, latent_space_per_layer);
                lstm_options.batch_first(true);
                lstm_options.num_layers(number_layers);
                m_lstm = torch::nn::LSTM(lstm_options);

                register_module("LSTM", m_lstm);
            }

            torch::Tensor forward(const torch::Tensor &x) {
                RNNOutput lstm_output = m_lstm->forward(x);
                const std::tuple<torch::Tensor, torch::Tensor>& tuple_state = std::get<1>(lstm_output);
                return torch::stack({std::get<0>(tuple_state), std::get<1>(tuple_state)});
            }

            torch::nn::LSTM m_lstm{nullptr};
        };

        TORCH_MODULE(LSTMEncoder);

        struct LSTMDecoderImpl : torch::nn::Module {
          
          typedef std::tuple<torch::Tensor, std::tuple<torch::Tensor, torch::Tensor>> RNNOutput;

            LSTMDecoderImpl(size_t input_size, size_t latent_space_per_layer, size_t number_layers) {
                torch::nn::LSTMOptions lstm_options(input_size, latent_space_per_layer);
                lstm_options.batch_first(true);
                lstm_options.num_layers(number_layers);
                m_lstm = torch::nn::LSTM(lstm_options);

                m_linear = torch::nn::Linear(latent_space_per_layer, input_size);

                register_module("LSTM", m_lstm);
                register_module("Linear", m_linear);
            }

            torch::Tensor forward(const torch::Tensor &x,
                                  const torch::Tensor &lstm_state) {
                torch::Tensor input_tensor = torch::zeros({x.size(0), 1, x.size(2)}).to(x.device());
                RNNOutput rnn_output;

//                torch::Tensor hidden_cell_state = torch::zeros({2, m_lstm->options.num_layers(), x.size(0), m_lstm->options.hidden_size()}).to(x.device());
//                hidden_cell_state.select(0,0).select(0, 0) = latent.squeeze(1);
                torch::Tensor hidden_cell_state = lstm_state.to(x.device());

                torch::Tensor result = torch::empty(x.sizes()).to(x.device());

                for (int index = 0; index < x.size(1); ++index) {
                    rnn_output = m_lstm->forward(input_tensor, std::make_tuple(hidden_cell_state.select(0,0), hidden_cell_state.select(0,1)));
                    input_tensor = m_linear->forward(std::get<0>(rnn_output));
                    const std::tuple<torch::Tensor, torch::Tensor>& tuple_state = std::get<1>(rnn_output);
                    hidden_cell_state = torch::stack({std::get<0>(tuple_state), std::get<1>(tuple_state)});
                    result.select(1, x.size(1) - index - 1) = input_tensor.squeeze(1);
                }
                return result;
            }

            torch::nn::LSTM m_lstm{nullptr};
            torch::nn::Linear m_linear{nullptr};
        };

        TORCH_MODULE(LSTMDecoder);


        struct LSTMAutoencoderImpl : torch::nn::Module {

            LSTMAutoencoderImpl(size_t input_size, size_t latent_space_per_layer, size_t number_layers)  {
                m_encoder = LSTMEncoder(input_size, latent_space_per_layer, number_layers);
                m_decoder = LSTMDecoder(input_size, latent_space_per_layer, number_layers);

                register_module("LSTMEncoder", m_encoder);
                register_module("LSTMDecoder", m_decoder);
            }

            torch::Tensor forward(const torch::Tensor &x)  {
                return m_decoder->forward(x, m_encoder->forward(x));
            }

            torch::Tensor forward_get_latent(const torch::Tensor &x, torch::Tensor &latent) {
                torch::Tensor encoding = m_encoder->forward(x);

                // TODO: To change
                latent = encoding.select(0,1).view(
                        {x.size(0),
                         m_encoder->m_lstm->options.num_layers() * m_encoder->m_lstm->options.hidden_size()}
                         ); // shape = {batch_size, hidden_size * number_layers}

                return m_decoder->forward(x, encoding);
            }

            LSTMEncoder m_encoder{nullptr};
            LSTMDecoder m_decoder{nullptr};
        };

        TORCH_MODULE(LSTMAutoencoder);
    }
}



#endif //AURORA_LSTM_AUTO_ENCODER_HPP

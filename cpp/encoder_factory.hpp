//
// Created by Luca Grillotti on 08/02/2020.
//

#ifndef AURORA_ENCODER_TYPE_FACTORY_HPP
#define AURORA_ENCODER_TYPE_FACTORY_HPP


#include "modifier/network_loader_pytorch.hpp"
#include "modifier/encoder_pca.hpp"

#include "compilation_variables.hpp"


namespace aurora {
    template<typename TParams, EncoderType>
    struct EncoderTypeFactory {};

    template<typename TParams>
    struct EncoderTypeFactory<TParams, EncoderType::cnn_ae> {
        typedef aurora::NetworkLoaderAutoEncoder<TParams> network_loader_t;
    };

    template<typename TParams>
    struct EncoderTypeFactory<TParams, EncoderType::strg_cnn> {
      typedef aurora::NetworkLoaderAutoEncoder<TParams> network_loader_t;
    };

    template<typename TParams>
    struct EncoderTypeFactory<TParams, EncoderType::lstm_ae> {
        typedef aurora::NetworkLoaderLSTM<TParams> network_loader_t;
    };

    template<typename TParams>
    struct EncoderTypeFactory<TParams, EncoderType::video_ae> {
        typedef aurora::NetworkLoaderVideoAutoEncoder<TParams> network_loader_t;
    };

    template<typename TParams>
    struct EncoderTypeFactory<TParams, EncoderType::conv_seq_ae> {
        typedef aurora::NetworkLoaderConvolutionalSequenceAutoencoder<TParams> network_loader_t;
    };

    template<typename TParams>
    struct EncoderTypeFactory<TParams, EncoderType::pca> {
      typedef aurora::EncoderPCA<TParams> network_loader_t;
    };

    template<typename TParams>
    struct EncoderTypeFactory<TParams, EncoderType::mlp_ae> {
      typedef aurora::NetworkLoaderMLPAutoEncoder<TParams> network_loader_t;
    };
}

#endif //AURORA_PARAMETERS_FACTORY_HPP

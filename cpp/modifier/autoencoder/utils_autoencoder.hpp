//
// Created by Luca Grillotti on 15/12/2019.
//

#ifndef AURORA_UTILS_AUTOENCODER_HPP
#define AURORA_UTILS_AUTOENCODER_HPP

#include <torch/torch.h>

namespace aurora {
    namespace nn {
        struct FlattenImpl : torch::nn::Module {
            FlattenImpl() = default;

            torch::Tensor forward(const torch::Tensor &x) {
                return x.view({x.size(0), -1});
            }
        };

        TORCH_MODULE(Flatten);

        struct PrintImpl : torch::nn::Module {
            PrintImpl() = default;

            torch::Tensor forward(const torch::Tensor &x) {
                std::cout << x.sizes() << '\n';
                return x;
            }
        };

        TORCH_MODULE(Print);
    }
}
#endif //AURORA_UTILS_AUTOENCODER_HPP

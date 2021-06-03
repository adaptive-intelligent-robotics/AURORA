//
// Created by Szymon Brych on 06/08/19.
//

#ifndef AURORA_GEN_MLP_HPP
#define AURORA_GEN_MLP_HPP

#include "modules/nn2/pf.hpp"
#include <iostream>
#include <vector>

#include <modules/nn2/connection.hpp>
#include <modules/nn2/neuron.hpp>
#include <modules/nn2/nn.hpp>

#include <modules/nn2/gen_dnn_ff.hpp>

namespace sferes {
  namespace gen {
    // a basic multi-layer perceptron (feed-forward neural network)
    // multiple hidden layer in this version

    template<typename N, typename C, typename Params>
    class GenMlp : public DnnFF<N, C, Params>
    {
    public:
      typedef nn::NN<N, C> nn_t;
      typedef typename nn_t::io_t io_t;
      typedef typename nn_t::vertex_desc_t vertex_desc_t;
      typedef typename nn_t::graph_t graph_t;

      GenMlp()
        : _hidden_neurons{}
      {}

      GenMlp&
      operator=(const GenMlp& o)
      {
        static_cast<nn::NN<N, C>&>(*this) = static_cast<const nn::NN<N, C>&>(o);
        return *this;
      }

      GenMlp(const GenMlp& o) { *this = o; }

      void
      init()
      {
        DnnFF<N, C, Params>::init();
      }

      void
      random()
      {
        assert(Params::dnn::init == dnn::ff);

        std::vector<size_t> nbs_hidden{};

        // for some reason cannot directly push back param without causing a linker error
        if (Params::mlp::layer_0_size > 0) {
          size_t layer_0_size = Params::mlp::layer_0_size;
          nbs_hidden.push_back(layer_0_size);
        };

        if (Params::mlp::layer_1_size > 0) {
          size_t layer_1_size = Params::mlp::layer_1_size;
          nbs_hidden.push_back(layer_1_size);
        }

        _random(Params::dnn::nb_inputs, nbs_hidden, Params::dnn::nb_outputs);

        this->_random_neuron_params();
        this->_make_all_vertices();
      }

      void
      _random(size_t nb_inputs, const std::vector<size_t>& nbs_hidden, size_t nb_outputs)
      {

        //                std::cout << std::endl;
        //                std::cout << nb_inputs;
        //                std::cout << std::endl;

        // neurons
        // add +1 for bias trick
        this->set_nb_inputs(nb_inputs); //+1);
        this->set_nb_outputs(nb_outputs);

        for (size_t j = 0; j < nbs_hidden.size(); ++j) {

          _hidden_neurons.push_back(std::vector<vertex_desc_t>{});

          for (size_t i = 0; i < nbs_hidden[j]; ++i)
            _hidden_neurons[j].push_back(this->add_neuron(std::string("layer") + std::to_string(j) +
                                                          std::string("_h") + std::to_string(i)));
        }
        // connections

        if (nbs_hidden.size() > 0) {
          this->full_connect_random(this->_inputs, this->_hidden_neurons[0]);

          for (size_t j = 0; j < nbs_hidden.size() - 1; ++j) {
            this->full_connect_random(this->_hidden_neurons[j], this->_hidden_neurons[j + 1]);
          }

          this->full_connect_random(this->_hidden_neurons[_hidden_neurons.size() - 1], this->_outputs);
        } else {
          this->full_connect_random(this->_inputs, this->_outputs);
        }

        // below is bias addition, it was commented out in favor of biased activations
        //                //bias every hidden layer above first too
        //                for (size_t j = 1; j < nbs_hidden.size(); ++j) {
        //
        //                    for (size_t i = 0; i < nbs_hidden[j]; ++i) {
        //                        w.random();
        //                        this->add_connection(this->get_input(nb_inputs),
        //                        this->_hidden_neurons[j][i],w);
        //                    }
        //                }
        //                // bias outputs too
        //                for (size_t i = 0; i < nb_outputs; ++i) {
        //                    w.random();
        //                    this->add_connection(this->get_input(nb_inputs), this->get_output(i),w);
        //                }
      }

      void
      full_connect_random(const std::vector<vertex_desc_t> v1, const std::vector<vertex_desc_t> v2)
      {
        BOOST_FOREACH (vertex_desc_t x, v1)
          BOOST_FOREACH (vertex_desc_t y, v2)
            this->add_connection(x, y, this->_random_weight());
      }

      void
      mutate()
      {
        DnnFF<N, C, Params>::mutate();
      }

      std::vector<float>
      data()
      {
        std::vector<float> data;

        // Getting weights on connections (edges of the graph)
        BGL_FORALL_EDGES_T(e, this->_g, graph_t)
        {
          data.insert(data.end(),
                      this->_g[e].get_weight().gen().data().begin(),
                      this->_g[e].get_weight().gen().data().end());
        }

        // Getting neurons parameters (only af params as pf params were considered already in the connection
        // weights)
        BGL_FORALL_VERTICES_T(v, this->_g, graph_t)
        {
          data.insert(data.end(),
                      this->_g[v].get_afparams().gen().data().begin(),
                      this->_g[v].get_afparams().gen().data().end());
        }

        return data;
      }

      void
      cross(const GenMlp& o, GenMlp& c1, GenMlp& c2)
      {
        if (misc::flip_coin()) {
          c1 = *this;
          c2 = o;
        } else {
          c2 = *this;
          c1 = o;
        }
      }

      // bias trick continuation, pay attention to non-virtual inheritance if recovering those
      //            unsigned get_nb_inputs() const {
      //                return this->_inputs.size() - 1;
      //            }
      //
      //            void step(const std::vector<io_t> &in) {
      //                assert(in.size() == this->get_nb_inputs());
      //                std::vector<io_t> inf = in;
      //                inf.push_back(1.0f);
      //                nn_t::_step(inf);
      //            }

    protected:
      std::vector<std::vector<vertex_desc_t>> _hidden_neurons;
    };
  } // namespace gen
} // namespace sferes

#endif // AURORA_GEN_MLP_HPP

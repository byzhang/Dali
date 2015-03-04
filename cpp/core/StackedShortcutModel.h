#ifndef STACKED_SHORTCUT_MAT_H
#define STACKED_SHORTCUT_MAT_H

#include <fstream>
#include <gflags/gflags.h>
#include <iostream>
#include <map>
#include <sstream>
#include <unordered_map>

#include "CrossEntropy.h"
#include "Layers.h"
#include "Mat.h"
#include "Softmax.h"
#include "utils.h"
#include "StackedGatedModel.h"
#include "core/RecurrentEmbeddingModel.h"


/**
StackedShortcutModel
--------------------

A Model for making sequence predictions using stacked LSTM cells.

The network uses an embedding layer, and can reconstruct a sequence.

The objective function is built using masked cross entropy (only certain
input channels collect error over small intervals).

The Stacked **Shortcut** Model allows inputs to travel directly to all
stacks of the model, thereby allowing "shortcuts" of information upwards
in the model.

**/

#ifndef SHORTCUT_DECODE_ACROSS_LAYERS
#define SHORTCUT_DECODE_ACROSS_LAYERS
#endif

template<typename T>
class StackedShortcutModel : public RecurrentEmbeddingModel<T> {
        typedef LSTM<T>                    lstm;
        typedef ShortcutLSTM<T>   shortcut_lstm;
        #ifdef SHORTCUT_DECODE_ACROSS_LAYERS
            typedef StackedInputLayer<T>  classifier_t;
        #else
            typedef Layer<T>              classifier_t;
        #endif
        typedef Mat<T>                      mat;
        typedef std::shared_ptr<mat> shared_mat;
        typedef Graph<T>                graph_t;
        typedef std::map<std::string, std::vector<std::string>> config_t;

        inline void name_parameters();
        /**
        Construct LSTM Cells (private)
        ------------------------------

        Construct LSTM cells using the provided hidden sizes and
        the input size to the Stacked LSTMs.

        **/
        inline void construct_LSTM_cells();

        /**
        Construct LSTM Cells (private)
        ------------------------------

        Constructs cells using either deep or shallow copies from
        other cells.

        Inputs
        ------

        const std::vector<LSTM<T>>& cells : cells for copy
                              bool copy_w : should each LSTM copy the parameters or share them
                             bool copy_dw : should each LSTM copy the gradient memory `dw` or share it.

        **/
        inline void construct_LSTM_cells(const std::vector<shortcut_lstm>&, bool, bool);

        public:
                typedef std::pair<std::vector<shared_mat>, std::vector<shared_mat>> state_type;
                typedef std::pair<state_type, shared_mat > activation_t;
                typedef T value_t;

                lstm                   base_cell;
                std::vector<shortcut_lstm> cells;
                shared_mat    embedding;
                typedef Eigen::Matrix<uint, Eigen::Dynamic, Eigen::Dynamic> index_mat;
                typedef std::shared_ptr< index_mat > shared_index_mat;

                const classifier_t decoder;
                virtual std::vector<shared_mat> parameters() const;
                void save(std::string) const;

                /**
                Decoder initialization
                ----------------------

                Prepare sequence of input sizes to
                parametrize the decoder for this shorcut
                stacked LSTM model.

                Inputs
                ------

                   int input_size : size of input embedding
                  int hidden_size : size of internal layers
                   int stack_size : how many stacked layers
                                    are used.

                Outputs
                -------

                std::vector<int> init_list : sizes needed for decoder init.

                **/
                static std::vector<int> decoder_initialization(int, int, int);


                /**
                Decoder initialization
                ----------------------

                Prepare sequence of input sizes to
                parametrize the decoder for this shorcut
                stacked LSTM model.

                Inputs
                ------

                       int input_size : size of input embedding
std::vector<int> hidden_sizes : size of internal layers

                Outputs
                -------

                std::vector<int> init_list : sizes needed for decoder init.

                **/
                static std::vector<int> decoder_initialization(int, std::vector<int>);
                static std::vector<int> decoder_initialization(int, const std::vector<std::string>&);

                /**
                Load
                ----

                Load a saved copy of this model from a directory containing the
                configuration file named "config.md", and from ".npy" saves of
                the model parameters in the same directory.

                Inputs
                ------

                std::string dirname : directory where the model is currently saved

                Outputs
                -------

                StackedShortcutModel<T> model : the saved model

                **/
                static StackedShortcutModel<T> load(std::string);
                static StackedShortcutModel<T> build_from_CLI(std::string load_location,
                                                                                                          int vocab_size,
                                                                                                          int output_size,
                                                                                                          bool verbose);

                StackedShortcutModel(int, int, int, int, int);
                StackedShortcutModel(int, int, int, std::vector<int>&);

                /**
                StackedShortcutModel Constructor from configuration map
                ----------------------------------------------------

                Construct a model from a map of configuration parameters.
                Useful for reinitializing a model that was saved to a file
                using the `utils::file_to_map` function to obtain a map of
                configurations.

                Inputs
                ------

                std::map<std::string, std::vector<std::string>& config : model hyperparameters

                **/
                StackedShortcutModel(const config_t&);

                /**
                StackedShortcutModel<T>::StackedShortcutModel
                -----------------------------

                Copy constructor with option to make a shallow
                or deep copy of the underlying parameters.

                If the copy is shallow then the parameters are shared
                but separate gradients `dw` are used for each of
                thread StackedShortcutModel<T>.

                Shallow copies are useful for Hogwild and multithreaded
                training

                See `Mat<T>::shallow_copy`, `examples/character_prediction.cpp`,
                `StackedShortcutModel<T>::shallow_copy`

                Inputs
                ------

                      StackedShortcutModel<T> l : StackedShortcutModel from which to source parameters and dw
                            bool copy_w : whether parameters for new StackedShortcutModel should be copies
                                          or shared
                           bool copy_dw : whether gradients for new StackedShortcutModel should be copies
                                          shared (Note: sharing `dw` should be used with
                                          caution and can lead to unpredictable behavior
                                          during optimization).

                Outputs
                -------

                StackedShortcutModel<T> out : the copied StackedShortcutModel with deep or shallow copy of parameters

                **/
                StackedShortcutModel(const StackedShortcutModel<T>&, bool, bool);
                T masked_predict_cost(graph_t&, shared_index_mat, shared_index_mat, shared_eigen_index_vector, shared_eigen_index_vector, uint offset=0,  T drop_prob = 0.0);
                T masked_predict_cost(graph_t&, shared_index_mat, shared_index_mat, uint, shared_eigen_index_vector, uint offset=0, T drop_prob = 0.0);
                std::vector<int> reconstruct(Indexing::Index, int, int symbol_offset = 0) const;

                state_type get_final_activation(graph_t&, Indexing::Index, T drop_prob=0.0) const;

                /**
                Activate
                --------

                Run Stacked Model by 1 timestep by observing
                the element from embedding with index `index`
                and report the activation, cell, and hidden
                states

                Inputs
                ------

                Graph<T>& G : computation graph
                std::pair<std::vector<std::shared_ptr<Mat<T>>>, std::vector<std::shared_ptr<Mat<T>>>>& : previous state
                uint index : embedding observation

                Outputs
                -------

                std::pair<std::pair<vector<shared_ptr<Mat<T>>>, vector<shared_ptr<Mat<T>>>>, shared_ptr<Mat<T>> > out :
                    pair of LSTM hidden and cell states, and probabilities from the decoder.

                **/
                activation_t activate(graph_t&, state_type&, const uint&) const;

                activation_t activate(graph_t&, state_type&, const eigen_index_block) const;

                std::vector<utils::OntologyBranch::shared_branch> reconstruct_lattice(Indexing::Index, utils::OntologyBranch::shared_branch, int) const;

                /**
                Shallow Copy
                ------------

                Perform a shallow copy of a StackedShortcutModel<T> that has
                the same parameters but separate gradients `dw`
                for each of its parameters.

                Shallow copies are useful for Hogwild and multithreaded
                training

                See `StackedShortcutModel<T>::shallow_copy`, `examples/character_prediction.cpp`.

                Outputs
                -------

                StackedShortcutModel<T> out : the copied layer with sharing parameters,
                                           but with separate gradients `dw`

                **/
                StackedShortcutModel<T> shallow_copy() const;
};

#endif

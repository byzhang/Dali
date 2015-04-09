#include <gflags/gflags.h>
#include <tuple>
#include <vector>
#include <string>
#include <functional>

#include "dali/core.h"
#include "dali/data_processing/machine_comprehension.h"
#include "dali/data_processing/Glove.h"
#include "dali/utils.h"


using mc::Section;
using mc::Question;
using std::vector;
using std::string;
using utils::Vocab;
using utils::random_minibatches;
using utils::vsum;
using utils::assert2;
using std::function;

vector<Section> train_data, validate_data, test_data;
Vocab vocab;
ThreadPool* pool;

DEFINE_int32(j, 9, "Number of threads");
DEFINE_int32(minibatch, 50, "Number of sections considered in every minibatch gradient step.");
DEFINE_string(pretrained_vectors, "", "Load pretrained word vectors?");

// TODO(szymon): add gloved
// TODO(szymon): add dropout
// TODO(szymon): use ranking loss
// TODO(szymon): make vocab sane

template<typename R>
class NeuralNetworkLayer {
    public:
        typedef function<Mat<R>(Mat<R>)> activation_t;

        vector<int> hidden_sizes;
        vector<activation_t> activations;

        vector<Layer<R>> layers;

        NeuralNetworkLayer() {
        }

        NeuralNetworkLayer(vector<int> hidden_sizes) :
                NeuralNetworkLayer(hidden_sizes, identities(hidden_sizes.size() - 1)) {
        }

        NeuralNetworkLayer(vector<int> hidden_sizes, vector<activation_t> activations) :
                hidden_sizes(hidden_sizes),
                activations(activations) {
            assert2(activations.size() == hidden_sizes.size() - 1,
                    "Wrong number of activations for NeuralNetworkLayer");

            for (int lidx = 0; lidx < hidden_sizes.size() - 1; ++lidx) {
                layers.push_back(Layer<R>(hidden_sizes[lidx], hidden_sizes[lidx + 1]));
            }
        }

        NeuralNetworkLayer(const NeuralNetworkLayer& other, bool copy_w, bool copy_dw) :
                hidden_sizes(other.hidden_sizes),
                activations(other.activations) {
            for (auto& other_layer: other.layers) {
                layers.emplace_back(other_layer, copy_w, copy_dw);
            }
        }

        NeuralNetworkLayer shallow_copy() {
            return NeuralNetworkLayer(*this, false, true);
        }

        Mat<R> activate(Mat<R> input) {
            Mat<R> last_output = input;
            for (int i = 0; i < hidden_sizes.size() - 1; ++i)
                last_output = activations[i](layers[i].activate(last_output));

            return last_output;
        }

        vector<Mat<R>> parameters() {
            vector<Mat<R>> params;
            for (auto& layer: layers) {
                auto layer_params = layer.parameters();
                params.insert(params.end(), layer_params.begin(), layer_params.end());
            }
            return params;
        }

        static Mat<R> identity(Mat<R> m) { return m; }

        static vector<activation_t> identities(int n) {
            std::cout << n << std::endl;
            vector<activation_t> res;
            while(n--) res.push_back(identity);
            return res;
        }
};


template<typename R>
class DragonModel {
    public:
        int EMBEDDING_SIZE = 50;
        int HIDDEN_SIZE = 150;
        bool SEPARATE_EMBEDDINGS = false;
        bool SVD_INIT = true;
        vector<int> LSTM_STACKS = { 100 , 50 };
        vector<int> OUTPUT_NN_SIZES = {HIDDEN_SIZE, 100, 1};

        double DROPOUT_VALUE = 0.3;

        vector<typename NeuralNetworkLayer<R>::activation_t> OUTPUT_NN_ACTIVATIONS =
            { MatOps<R>::tanh, NeuralNetworkLayer<R>::identity };

        Mat<R> embedding;

        Mat<R> text_embedding;
        Mat<R> question_embedding;
        Mat<R> answer_embedding;

        StackedLSTM<R> text_model;
        StackedLSTM<R> question_model;
        StackedLSTM<R> answer_model;

        StackedInputLayer<R> words_repr_to_hidden;

        NeuralNetworkLayer<R> output_classifier;

        DragonModel(const DragonModel& other, bool copy_w, bool copy_dw) {
            if (SEPARATE_EMBEDDINGS) {
                text_embedding = Mat<R>(other.text_embedding, copy_w, copy_dw);
                question_embedding = Mat<R>(other.question_embedding, copy_w, copy_dw);
                answer_embedding = Mat<R>(other.answer_embedding, copy_w, copy_dw);
            } else {
                embedding = Mat<R>(other.embedding, copy_w, copy_dw);
            }
            text_model = StackedLSTM<R>(other.text_model, copy_w, copy_dw);
            question_model = StackedLSTM<R>(other.answer_model, copy_w, copy_dw);
            answer_model = StackedLSTM<R>(other.answer_model, copy_w, copy_dw);

            words_repr_to_hidden =
                    StackedInputLayer<R>(other.words_repr_to_hidden, copy_w, copy_dw);

            output_classifier = NeuralNetworkLayer<R>(other.output_classifier, copy_w, copy_dw);
        }

        DragonModel() {
            auto weight_init = weights<R>::uniform(1.0/EMBEDDING_SIZE);
            if (!FLAGS_pretrained_vectors.empty()) {
                assert(!SEPARATE_EMBEDDINGS);
                int num_loaded = glove::load_relevant_vectors(
                        FLAGS_pretrained_vectors, embedding, vocab, 1000000);
                std::cout << num_loaded << " out of " << vocab.word2index.size()
                          << " word embeddings preloaded from glove." << std::endl;
                assert (embedding.dims(1) == EMBEDDING_SIZE);
            }

            if (SEPARATE_EMBEDDINGS) {
                text_embedding = Mat<R>(vocab.word2index.size(), EMBEDDING_SIZE, weight_init);
                question_embedding = Mat<R>(vocab.word2index.size(), EMBEDDING_SIZE, weight_init);
                answer_embedding = Mat<R>(vocab.word2index.size(), EMBEDDING_SIZE, weight_init);
            } else {
                embedding = Mat<R>(vocab.word2index.size(), EMBEDDING_SIZE, weight_init);
            }
            text_model     = StackedLSTM<R>(EMBEDDING_SIZE, LSTM_STACKS, true, true);
            question_model = StackedLSTM<R>(EMBEDDING_SIZE, LSTM_STACKS, true, true);
            answer_model   = StackedLSTM<R>(EMBEDDING_SIZE, LSTM_STACKS, true, true);

            const int LSTM_OUTPUT_SIZE = utils::vsum(LSTM_STACKS);
            words_repr_to_hidden = StackedInputLayer<R>({ LSTM_OUTPUT_SIZE,
                                                          LSTM_OUTPUT_SIZE,
                                                          LSTM_OUTPUT_SIZE
                                                         }, HIDDEN_SIZE);

            output_classifier = NeuralNetworkLayer<R>(OUTPUT_NN_SIZES, OUTPUT_NN_ACTIVATIONS);

            if (SVD_INIT) {
                // Don't use SVD for embeddings!
                auto params = words_repr_to_hidden.parameters();
                auto params2 = output_classifier.parameters();
                params.insert(params.end(), params2.begin(), params2.end());
                for (auto param: params) {
                    weights<R>::svd(weights<R>::gaussian(1.0))(param);
                }
                std::cout << "Initialized weights with SVD!" << std::endl;
            }
        }

        DragonModel shallow_copy() const {
            return DragonModel(*this, false, true);
        }

        Mat<R> words_to_repr(const vector<string>& words,
                             StackedLSTM<R> model,
                             Mat<R> embedding,
                             bool use_dropout) {
            auto word_idxs = vocab.transform(words);
            Seq<Mat<R>> words_embeddings;
            for (auto word_idx: word_idxs) {
                assert (word_idx < vocab.index2word.size());
                words_embeddings.push_back(embedding[word_idx]);
            }
            auto out_states = model.activate_sequence(model.initial_states(),
                                                      words_embeddings,
                                                      use_dropout ? DROPOUT_VALUE : 0.0);
            return MatOps<R>::vstack(LSTM<R>::State::hiddens(out_states));
        }

        Mat<R> answer_score(const vector<string>& text,
                            const vector<string>& question,
                            const vector<string>& answer,
                            bool use_dropout) {
            Mat<R> text_repr     = words_to_repr(text, text_model,
                    SEPARATE_EMBEDDINGS ? text_embedding : embedding, use_dropout);
            Mat<R> question_repr = words_to_repr(question, question_model,
                    SEPARATE_EMBEDDINGS ? question_embedding : embedding, use_dropout);
            Mat<R> answer_repr   = words_to_repr(answer, answer_model,
                    SEPARATE_EMBEDDINGS ? answer_embedding : embedding, use_dropout);

            Mat<R> hidden = words_repr_to_hidden.activate({text_repr,
                                                           question_repr,
                                                           answer_repr}).tanh();

            return output_classifier.activate(hidden);
        }

        vector<Mat<R>> parameters() {
            vector<Mat<R>> params;
            if (SEPARATE_EMBEDDINGS) {
                params = { text_embedding, question_embedding, answer_embedding };
            } else {
                params = { embedding };
            }
            auto temp = words_repr_to_hidden.parameters();
            params.insert(params.end(), temp.begin(), temp.end());
            temp = output_classifier.parameters();
            params.insert(params.end(), temp.begin(), temp.end());
            return params;
        }

        vector<Mat<R>> answer_scores(const vector<string>& text,
                                     const vector<string>& question,
                                     const vector<vector<string>>& answers,
                                     bool use_dropout) {
            vector<Mat<R>> scores;
            for (auto& answer: answers) {
                scores.push_back(answer_score(text, question, answer, use_dropout));
            }

            return scores;
        }

        Mat<R> error(const vector<string>& text,
                     const vector<string>& question,
                     const vector<vector<string>>& answers,
                     int correct_answer) {
            auto scores = answer_scores(text, question, answers, true);

            R margin = 0.1;

            Mat<R> error(1,1);
            for (int aidx=0; aidx < answers.size(); ++aidx) {
                if (aidx == correct_answer) continue;
                error = error + MatOps<R>::max(scores[aidx] - scores[correct_answer] + margin, 0.0);
            }

            return error;
        }

        int predict(const vector<string>& text,
                    const vector<string>& question,
                    const vector<vector<string>>& answers) {
            auto scores = answer_scores(text, question, answers, false);

            return MatOps<R>::vstack(scores).argmax();
        }
};

class ThreadError {
    public:
        const int num_threads;
        vector<double> thread_error;
        vector<int>    thread_error_updates;

        ThreadError(int num_threads) :
                num_threads(num_threads),
                thread_error(num_threads),
                thread_error_updates(num_threads) {
            reset();
        }

        void update(double error) {
            thread_error[ThreadPool::get_thread_number()] += error;
            thread_error_updates[ThreadPool::get_thread_number()] += 1;
        }

        double average() {
            return vsum(thread_error) / vsum(thread_error_updates);
        }

        void reset() {
            for (int tidx = 0; tidx < num_threads; ++tidx) {
                thread_error[tidx] = 0;
                thread_error_updates[tidx] = 0;
            }
        }

};

typedef DragonModel<double> model_t;

double calculate_accuracy(model_t& model, const vector<Section>& data) {
    int correct = 0;
    int total = 0;

    graph::NoBackprop nb;
    for (auto& section : data) {
        for (auto& question: section.questions) {
            int ans = model.predict(section.text, question.text, question.answers);
            if (ans == question.correct_answer) ++correct;
            ++total;
        }
    }
    return (double)correct/total;
}


int main(int argc, char** argv) {
    sane_crashes::activate();

    GFLAGS_NAMESPACE::SetUsageMessage(
        "\nMicrosoft Machine Comprehension Task"
    );

    GFLAGS_NAMESPACE::ParseCommandLineFlags(&argc, &argv, true);

    Eigen::setNbThreads(0);
    Eigen::initParallel();

    const double VALIDATION_FRACTION = 0.1;

    // Load the data set
    std::tie(train_data, test_data) = mc::load();
    // shuffle examples
    std::random_shuffle(train_data.begin(), train_data.end());
    std::random_shuffle(test_data.begin(), test_data.end());
    // separate validation dataset
    int num_validation = train_data.size() * VALIDATION_FRACTION;
    validate_data = vector<Section>(train_data.begin(), train_data.begin() + num_validation);
    train_data.erase(train_data.begin(), train_data.begin() + num_validation);
    // extract vocabulary
    // only consider common words.
    vector<vector<string>> wrapper;
    wrapper.emplace_back(mc::extract_vocabulary(train_data));
    auto index2word = utils::get_vocabulary(wrapper, 2);
    vocab = Vocab(index2word);

    std::cout << "Datasets : " << "train (" << train_data.size() << " items), "
                               << "validate (" << validate_data.size() << " items), "
                               << "test (" << test_data.size() << " items)" << std::endl;
    std::cout << "vocabulary size : " << vocab.word2index.size() << std::endl;

    pool = new ThreadPool(FLAGS_j);

    auto training = LSTV(0.9, 0.3, 3);

    model_t model;
    auto params = model.parameters();
    auto solver = Solver::Adam<double>(params);

    vector<model_t> thread_models;

    for (int tmidx = 0; tmidx < FLAGS_j; tmidx++)
        thread_models.emplace_back(model.shallow_copy());

    auto thread_error = ThreadError(FLAGS_j);

    model_t best_model(model, false, false);
    float best_accuracy = 0.0;
    while(true) {
        thread_error.reset();

        auto minibatches = random_minibatches(train_data.size(), FLAGS_minibatch);

        ReportProgress<double> journalist("Training", train_data.size());
        std::atomic<int> processed_sections(0);

        for (int bidx = 0; bidx < minibatches.size(); ++bidx) {
            pool->run([bidx, &minibatches, &thread_models, &solver, &thread_error,
                       &processed_sections, &journalist]() {
                auto& batch = minibatches[bidx];
                model_t& thread_model = thread_models[ThreadPool::get_thread_number()];

                double partial_error = 0.0;
                int partial_error_updates = 0;
                for (auto& example_idx: batch) {
                    Section& section = train_data[example_idx];
                    for (auto& question: section.questions) {
                        auto e = thread_model.error(section.text,
                                                    question.text,
                                                    question.answers,
                                                    question.correct_answer);
                        partial_error += e.w()(0,0);
                        partial_error_updates += 1;
                        e.grad();
                        graph::backward();
                    }
                    processed_sections += 1;
                    journalist.tick(processed_sections);
                }
                thread_error.update(partial_error / (double)partial_error_updates);
                auto params = thread_model.parameters();
                solver.step(params);
            });
        }
        pool->wait_until_idle();
        journalist.done();
        double val_acc = 100.0 * calculate_accuracy(model, validate_data);
        std::cout << "Training error = " << thread_error.average()
                  << ", Validation accuracy = " << val_acc << "%" << std::endl;

        if (val_acc > best_accuracy) {
            std::cout << "NEW WORLD RECORD!" << std::endl;
            best_accuracy = val_acc;
            best_model = model_t(model, true, true);
        }

        if (training.should_stop(100.0-val_acc)) break;
    }
    double test_acc = 100.0 * calculate_accuracy(best_model, test_data);
    std::cout << "Test accuracy = " << test_acc << "%" << std::endl;
}

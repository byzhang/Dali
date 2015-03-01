#include <algorithm>
#include <atomic>
#include <Eigen/Eigen>
#include <fstream>
#include <iterator>
#include <thread>
#include <chrono>

#include "core/gzstream.h"
#include "core/NlpUtils.h"
#include "core/SST.h"
#include "core/StackedModel.h"
#include "core/utils.h"
#include "core/Reporting.h"
#include "core/ThreadPool.h"
#include "core/SequenceProbability.h"

using std::vector;
using std::make_shared;
using std::shared_ptr;
using std::ifstream;
using std::istringstream;
using std::stringstream;
using std::string;
using std::min;
using std::thread;
using std::ref;
using utils::Vocab;
using utils::from_string;
using utils::OntologyBranch;
using utils::tokenized_uint_labeled_dataset;
using std::atomic;
using std::chrono::seconds;


typedef float REAL_t;
typedef Graph<REAL_t> graph_t;
typedef Mat<REAL_t> mat;
typedef float price_t;
typedef Eigen::Matrix<uint, Eigen::Dynamic, Eigen::Dynamic> index_mat;
typedef Eigen::Matrix<REAL_t, Eigen::Dynamic, 1> float_vector;
typedef std::pair<vector<string>, uint> labeled_pair;

const string START = "**START**";

ThreadPool* pool;

DEFINE_int32(minibatch, 100, "What size should be used for the minibatches ?");
DEFINE_double(cutoff, 2.0, "KL Divergence error where stopping is acceptable");
DEFINE_int32(patience,             5,    "How many unimproving epochs to wait through before witnessing progress ?");


/**
Databatch
---------

Datastructure handling the storage of training
data, length of each example in a minibatch,
and total number of prediction instances
within a single minibatch.

**/
class Databatch {
    typedef shared_ptr<index_mat> shared_index_mat;
    public:
        shared_index_mat data;
        shared_eigen_index_vector targets;
        shared_eigen_index_vector codelens;
        int total_codes;
        Databatch(int n, int d) {
            data        = make_shared<index_mat>(n, d);
            targets     = make_shared<eigen_index_vector>(n);
            codelens    = make_shared<eigen_index_vector>(n);
            total_codes = 0;
            data->fill(0);
        };
};

void insert_example_indices_into_matrix(
    Vocab& word_vocab,
    Databatch& databatch,
    labeled_pair& example,
    size_t& row) {
    auto description_length = example.first.size();
    (*databatch.data)(row, 0) = word_vocab.word2index[START];
    for (size_t j = 0; j < description_length; j++)
        (*databatch.data)(row, j + 1) = word_vocab.word2index.find(example.first[j]) != word_vocab.word2index.end() ? word_vocab.word2index[example.first[j]] : word_vocab.unknown_word;
    (*databatch.data)(row, description_length + 1) = word_vocab.word2index[utils::end_symbol];
    (*databatch.codelens)(row) = description_length + 1;
    databatch.total_codes += description_length + 1;
    (*databatch.targets)(row) = example.second;
}

Databatch convert_sentences_to_indices(
    tokenized_uint_labeled_dataset& examples,
    Vocab& word_vocab,
    size_t num_elements,
    vector<size_t>::iterator indices,
    vector<size_t>::iterator lengths_sorted) {

    auto indices_begin = indices;
    Databatch databatch(num_elements, *std::max_element(lengths_sorted, lengths_sorted + num_elements));
    for (size_t k = 0; k < num_elements; k++)
        insert_example_indices_into_matrix(
            word_vocab,
            databatch,
            examples[*(indices++)],
            k);
    return databatch;
}

vector<Databatch> create_labeled_dataset(
    tokenized_uint_labeled_dataset& examples,
    Vocab& word_vocab,
    size_t minibatch_size) {

    vector<Databatch> dataset;
    vector<size_t> lengths = vector<size_t>(examples.size());
    for (size_t i = 0; i != lengths.size(); ++i) lengths[i] = examples[i].first.size() + 2;
    vector<size_t> lengths_sorted(lengths);

    auto shortest = utils::argsort(lengths);
    std::sort(lengths_sorted.begin(), lengths_sorted.end());
    size_t piece_size = minibatch_size;
    size_t so_far = 0;

    auto shortest_ptr = lengths_sorted.begin();
    auto end_ptr = lengths_sorted.end();
    auto indices_ptr = shortest.begin();

    while (shortest_ptr != end_ptr) {
        dataset.emplace_back( convert_sentences_to_indices(
            examples,
            word_vocab,
            min(piece_size, lengths.size() - so_far),
            indices_ptr,
            shortest_ptr) );
        shortest_ptr += min(piece_size,          lengths.size() - so_far);
        indices_ptr  += min(piece_size,          lengths.size() - so_far);
        so_far        = min(so_far + piece_size, lengths.size());
    }
    return dataset;
}

/**
get word vocab
--------------

Collect a mapping from words to unique indices
from a collection of Annnotate Parse Trees
from the Stanford Sentiment Treebank, and only
keep words ocurring more than some threshold
number of times `min_occurence`

Inputs
------

std::vector<SST::AnnotatedParseTree::shared_tree>& trees : Stanford Sentiment Treebank trees
                                       int min_occurence : cutoff appearance of words to include
                                                           in vocabulary.


Outputs
-------

Vocab vocab : the vocabulary extracted from the trees with the
              addition of a special "**START**" word.

**/
Vocab get_word_vocab(vector<SST::AnnotatedParseTree::shared_tree>& trees, int min_occurence) {
    tokenized_uint_labeled_dataset examples;
    for (auto& tree : trees)
        examples.emplace_back(tree->to_labeled_pair());
    auto index2word  = utils::get_vocabulary(examples, min_occurence);
    Vocab vocab(index2word);
    vocab.word2index[START] = vocab.index2word.size();
    vocab.index2word.emplace_back(START);
    return vocab;
}


template<typename model_t>
REAL_t average_error(const vector<model_t>& models, const vector<vector<Databatch>>& validation_sets, const int& total_valid_examples) {
    atomic<int> correct(0);
    ReportProgress<double> journalist("Average error", total_valid_examples);
    atomic<int> total(0);

    for (int validation_set_num = 0; validation_set_num < validation_sets.size(); validation_set_num++) {
        for (int minibatch_num =0 ; minibatch_num < validation_sets[validation_set_num].size(); minibatch_num++) {
            for (int row_num = 0; row_num < validation_sets[validation_set_num][minibatch_num].data->rows(); row_num++) {
                pool->run([&journalist, &models, &correct, &total, &validation_sets, validation_set_num, minibatch_num, row_num] {
                    int best_model = -1;
                    auto& valid_set = validation_sets[validation_set_num][minibatch_num];
                    REAL_t best_prob = -std::numeric_limits<REAL_t>::infinity();
                    for (int k = 0; k < models.size();k++) {
                        auto log_prob = sequence_probability::sequence_probability(
                            models[k],
                            valid_set.data->row(row_num).head((*valid_set.codelens)(row_num)));
                        if (log_prob > best_prob) {
                            best_prob = log_prob;
                            best_model = k;
                        }
                    }
                    if (best_model == (*valid_set.targets)(row_num)) {
                        correct ++;
                    }
                    total++;
                    journalist.tick(total, (REAL_t) correct / (REAL_t) total);
                });
            }
        }
    }
    pool->wait_until_idle();
    return ((REAL_t) 100.0 * correct / (REAL_t) total_valid_examples);
};


int main( int argc, char* argv[]) {
    GFLAGS_NAMESPACE::SetUsageMessage(
        "\n"
        "Sentiment Analysis as Competition amongst Language Models\n"
        "---------------------------------------------------------\n"
        "\n"
        "We present a dual formulation of the word sequence classification\n"
        "task: we treat each label’s examples as originating from different\n"
        "languages and we train language models for each label; at test\n"
        "time we compare the likelihood of a sequence under each label’s\n"
        "language model to find the most likely assignment.\n"
        "\n"
        " @author Jonathan Raiman\n"
        " @date February 13th 2015"
    );


    GFLAGS_NAMESPACE::ParseCommandLineFlags(&argc, &argv, true);

    auto epochs              = FLAGS_epochs;
    auto sentiment_treebank  = SST::load(FLAGS_train);

    auto word_vocab          = get_word_vocab(sentiment_treebank, FLAGS_min_occurence);
    auto vocab_size          = word_vocab.index2word.size();

    // Load Dataset of Trees:
    std::cout << "Unique Treees Loaded : " << sentiment_treebank.size() << std::endl
              << "        Example tree : " << *sentiment_treebank[sentiment_treebank.size()-1] << std::endl
              << "     Vocabulary size : " << vocab_size << std::endl;

    // Put trees into matrices:
    int total_valid_examples = 0;
    const int NUM_SENTIMENTS = 5;
    const int BATCHES_PER_EPOCH = 30;
    vector<vector<Databatch>> datasets(NUM_SENTIMENTS);
    vector<vector<Databatch>> validation_sets(NUM_SENTIMENTS);

    {
        vector<tokenized_uint_labeled_dataset> tree_types(NUM_SENTIMENTS);
        vector<tokenized_uint_labeled_dataset> validation_tree_types(NUM_SENTIMENTS);

        for (auto& tree : sentiment_treebank) {
            if (((int) tree->label) > 4)
                utils::exit_with_message("Error: One of the trees has a label other than 0-4");
            tree_types[tree->label].emplace_back(tree->to_labeled_pair());
            for (auto& child : tree->general_children) {
                if (((int)child->label) > 4)
                    utils::exit_with_message("Error: One of the trees's children has a label other than 0-4");
                tree_types[(int) child->label].emplace_back(child->to_labeled_pair());
            }
        }
        auto validation_treebank = SST::load(FLAGS_validation);
        for (auto& tree : validation_treebank) {
            if (((int) tree->label) > 4)
                utils::exit_with_message("Error: One of the trees has a label other than 0-4");
            validation_tree_types[tree->label].emplace_back(tree->to_labeled_pair());
            for (auto& child : tree->general_children) {
                if (((int)child->label) > 4)
                    utils::exit_with_message("Error: One of the trees's children has a label other than 0-4");
                validation_tree_types[(int) child->label].emplace_back(child->to_labeled_pair());
            }
        }
        int i = 0;
        for (auto& tree_type : tree_types)
            std::cout << "Label type " << i++ << " has " << tree_type.size() << " different examples" << std::endl;
        i = 0;

        for (auto& tree_type : validation_tree_types) {
            std::cout << "Label type " << i++ << " has " << tree_type.size() << " validation examples" << std::endl;
            total_valid_examples += tree_type.size();
        }

        i = 0;
        for (auto& tree_type : tree_types)
            datasets[i++] = create_labeled_dataset(tree_type, word_vocab, FLAGS_minibatch);
        i = 0;
        for (auto& tree_type : validation_tree_types)
            validation_sets[i++] = create_labeled_dataset(tree_type, word_vocab, FLAGS_minibatch);
    }

    std::cout << "Max training epochs = " << FLAGS_epochs << std::endl;
    std::cout << "Training cutoff     = " << FLAGS_cutoff << std::endl;

    pool = new ThreadPool(FLAGS_j);

    int patience = 0;

    std::vector<StackedShortcutModel<REAL_t>> models;

    vector<vector<StackedShortcutModel<REAL_t>>> thread_models;
    vector<Solver::AdaDelta<REAL_t>> solvers;


    for (int sentiment = 0; sentiment < NUM_SENTIMENTS; sentiment++) {
        models.emplace_back(
            word_vocab.index2word.size(),
            FLAGS_input_size,
            FLAGS_hidden,
            FLAGS_stack_size < 1 ? 1 : FLAGS_stack_size,
            word_vocab.index2word.size()
        );
        thread_models.emplace_back();
        for (int thread_no = 0; thread_no < FLAGS_j; ++thread_no) {
            thread_models[sentiment].push_back(models[sentiment].shallow_copy());
        }
        auto params = models[sentiment].parameters();
        solvers.emplace_back(params, FLAGS_rho, 1e-9, 5.0);
    }
    int epoch = 0;
    auto cost = std::numeric_limits<REAL_t>::infinity();
    REAL_t new_cost;
    while (cost > FLAGS_cutoff && patience < FLAGS_patience) {
        stringstream ss;
        ss << "Epoch " << ++epoch;
        atomic<int> batches_processed(0);

        ReportProgress<double> journalist(ss.str(), NUM_SENTIMENTS * BATCHES_PER_EPOCH);

        for (int sentiment = 0; sentiment < NUM_SENTIMENTS; sentiment++) {
            for (int batch_id = 0; batch_id < BATCHES_PER_EPOCH; ++batch_id) {
                pool->run([&thread_models, &journalist, &solvers, &datasets, sentiment, &cost, &batches_processed]() {
                    auto& thread_model = thread_models[sentiment][ThreadPool::get_thread_number()];
                    auto& solver = solvers[sentiment];

                    auto thread_parameters = thread_model.parameters();
                    auto& minibatch = datasets[sentiment][utils::randint(0, datasets[sentiment].size()-1)];

                    auto G = graph_t(true);      // create a new graph for each loop
                    cost += thread_model.masked_predict_cost(
                        G,
                        minibatch.data, // the sequence to draw from
                        minibatch.data, // what to predict (the words offset by 1)
                        1,
                        minibatch.codelens,
                        0
                    );
                    G.backward(); // backpropagate
                    solver.step(thread_parameters, 0.0); // One step of gradient descent

                    journalist.tick(++batches_processed, cost);
                });
            }
        }

        while(true) {
            journalist.pause();
            std::cout << "Here be reconstructions" << std::endl;
            journalist.resume();
            // TODO(jonathan): reconstructions go here..
            if (pool->wait_until_idle(seconds(5)))
                break;
        }

        journalist.done();
        new_cost = average_error(models, validation_sets, total_valid_examples);

        if (new_cost > cost) {
            patience +=1;
        } else {
            patience = 0;
        }
        cost = new_cost;
    }

    return 0;
}

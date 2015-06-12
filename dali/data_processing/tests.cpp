/*#include <gtest/gtest.h>

#include "dali/data_processing/Glove.h"
#include "dali/data_processing/Arithmetic.h"
#include "dali/data_processing/NER.h"
#include "dali/data_processing/Paraphrase.h"
#include <vector>

using std::vector;

TEST(Glove, load) {
    auto embedding = glove::load<double>( STR(DALI_DATA_DIR) "/glove/test_data.txt");
    ASSERT_EQ(std::get<1>(embedding).size(), 21);
    ASSERT_EQ(std::get<0>(embedding).dims(0), 21);
    ASSERT_EQ(std::get<0>(embedding).dims(1), 300);
}

// exposing internal functions from arithmetic for testing.
namespace arithmetic {
    std::tuple<std::vector<int>, std::vector<std::string>> remove_multiplies(const std::vector<int>& numbers, const std::vector<std::string>& ops);
    std::tuple<std::vector<int>, std::vector<std::string>> generate_example(int expression_length, int& min, int& max);
    std::vector<std::string> convert_to_chars(const std::vector<int>& numbers, const std::vector<std::string>& ops);
    int compute_result(const std::vector<int>& numbers, const std::vector<std::string>& ops);
}

TEST(arithmetic, generate) {
    int min = 0;
    int max = 9;
    auto example = arithmetic::generate_example(5, min, max);
    ASSERT_EQ(std::get<0>(example).size(), 3);
    ASSERT_EQ(std::get<1>(example).size(), 2);

    auto example2 = std::make_tuple(
        std::vector<int>({12, 9, 3}),
        std::vector<std::string>({"*", "+"})
    );
    auto demultiplied = arithmetic::remove_multiplies(
        std::get<0>(example2),
        std::get<1>(example2)
    );
    auto example3     = std::make_tuple(
        std::vector<int>({108, 3}),
        std::vector<std::string>({"+"})
    );

    ASSERT_EQ(demultiplied, example3);
    ASSERT_EQ(arithmetic::compute_result(
        std::get<0>(example3),
        std::get<1>(example3)),
        111);
    example2     = std::make_tuple(
        std::vector<int>({12, 9, 3, 5, 5}),
        std::vector<std::string>({"*", "-","+", "*"})
    );
    demultiplied = arithmetic::remove_multiplies(
        std::get<0>(example2),
        std::get<1>(example2)
    );
    example3     = std::make_tuple(
        std::vector<int>({108, 3, 25}),
        std::vector<std::string>({"-", "+"})
    );
    ASSERT_EQ(demultiplied, example3);
    ASSERT_EQ(
        arithmetic::compute_result(
            std::get<0>(example3),
            std::get<1>(example3)
        ),
        130
    );
}

TEST(NER, load) {
    auto NER_data = NER::load( STR(DALI_DATA_DIR) "/Stanford_NER/NER_dummy_dataset.tsv");
    ASSERT_EQ(NER_data.size(), 2);

    auto loader = NER::NER_Loader();
    loader.data_column = 0;
    loader.label_column = -1;
    NER_data = loader.convert_tsv(
        utils::load_tsv(
            STR(DALI_DATA_DIR) "/CoNLL_NER/NER_dummy_dataset.tsv", 4, '\t'
        )
    );
    // how many examples
    ASSERT_EQ(NER_data.size(), 1);
    // how many words in first example
    ASSERT_EQ(NER_data.front().first.size(), 7);
    // last word of first example is period
    ASSERT_EQ(NER_data.front().first.back(), ".");
    // before last word of first example is a location
    ASSERT_EQ(NER_data.front().second[5], "I-LOC");
}

TEST(paraphrase, load) {
    auto paraphrase_data = paraphrase::STS_2015::load_train( STR(DALI_DATA_DIR) "/paraphrase_STS_2015/paraphrase_dummy_data.tsv");
    ASSERT_EQ(paraphrase_data.size(), 4);

    ASSERT_EQ(std::get<2>(paraphrase_data[0]), 1.0);
    ASSERT_EQ(std::get<2>(paraphrase_data[1]), 0.0);
    ASSERT_EQ(std::get<2>(paraphrase_data[2]), 1.0);
    ASSERT_EQ(std::get<2>(paraphrase_data[3]), 0.5);

    auto expected = utils::tokenize(
        "But my bro from the 757 EJ Manuel is the 1st QB gone"
    );

    ASSERT_EQ(std::get<1>(paraphrase_data[0]), expected);

    auto vocab       = paraphrase::get_vocabulary(paraphrase_data, 1);
    auto minibatches = paraphrase::convert_to_indexed_minibatches(vocab, paraphrase_data, 2);

    ASSERT_EQ(minibatches.size(), 2);

    auto char_vocab = utils::CharacterVocab(10, 255);
    auto char_minibatches = paraphrase::convert_to_indexed_minibatches(char_vocab, paraphrase_data, 2);

    ASSERT_EQ(char_minibatches.size(), 2);
    auto sentence = char_vocab.decode(std::get<0>(char_minibatches[0][0]));
    ASSERT_EQ(sentence, std::get<0>(paraphrase_data[0]));
}


TEST(paraphrase, generator_load) {
    auto gen_data = paraphrase::STS_2015::generate_train(STR(DALI_DATA_DIR) "/paraphrase_STS_2015/paraphrase_dummy_data.tsv");
    int ex_seen = 0;
    for (auto ex : gen_data) ex_seen++;
    ASSERT_EQ(ex_seen, 4);
    // TODO: find why when gen_data is overwritten a segfault occurs!!
    auto gen_data2 = paraphrase::STS_2015::generate_train(STR(DALI_DATA_DIR) "/paraphrase_STS_2015/paraphrase_dummy_data.tsv");
    auto paraphrase_data = paraphrase::STS_2015::load_train( STR(DALI_DATA_DIR) "/paraphrase_STS_2015/paraphrase_dummy_data.tsv");
    auto vocab_gen = paraphrase::get_vocabulary(gen_data2, 1);
    auto vocab     = paraphrase::get_vocabulary(paraphrase_data, 1);
    ASSERT_EQ(vocab_gen, vocab);
}

TEST(paraphrase, convert_to_indexed_minibatches) {
    auto gen_data = paraphrase::STS_2015::generate_train(STR(DALI_DATA_DIR) "/paraphrase_STS_2015/paraphrase_dummy_data.tsv");
    auto vocab_gen = utils::Vocab(paraphrase::get_vocabulary(gen_data, 1));

    // deadlock when generator inside generator is run.
    auto gen_minibatches = paraphrase::convert_to_indexed_minibatches(vocab_gen, gen_data, 3);

    vector<vector<paraphrase::numeric_example_t>> minibatches;
    for (auto minibatch : gen_minibatches) {
        minibatches.emplace_back(minibatch);
    }

    ASSERT_EQ(minibatches.size(), 2);

}


*/

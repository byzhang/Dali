/*#include <vector>
#include <gtest/gtest.h>
#include "dali/test_utils.h"
#include "dali/models/StackedGatedModel.h"
#include "dali/core.h"

typedef double R;

TEST(utils, shallow_copy) {
    int vocab_size = 6,
        input_size = 2,
        hidden_size = 5,
        stack_size = 4,
        output_size = 2;

    auto base_model = StackedGatedModel<R>(
        vocab_size,
        input_size,
        hidden_size,
        stack_size,
        output_size,
        true,
        true,
        0.01
    );

    int num_copies = 3;
    auto copies = utils::shallow_copy(base_model, num_copies);
    ASSERT_EQ(std::get<0>(copies).size(), num_copies);
}


TEST(utils, shallow_copy_multi_params) {
    int vocab_size = 6,
        input_size = 2,
        hidden_size = 5,
        stack_size = 4,
        output_size = 2;

    auto base_model = StackedGatedModel<R>(
        vocab_size,
        input_size,
        hidden_size,
        stack_size,
        output_size,
        true,
        true,
        0.01
    );

    int num_copies = 3;
    auto copies = utils::shallow_copy_multi_params(base_model, num_copies, [&base_model](const Mat<R>& mat)->bool {
        return mat.id() == base_model.embedding.id();
    });

    auto params = base_model.parameters();

    ASSERT_EQ(std::get<0>(copies).size(), num_copies);
    ASSERT_EQ(std::get<1>(copies)[0].size(), 1);
    ASSERT_EQ(std::get<2>(copies)[0].size(), params.size() - 1);
}
*/

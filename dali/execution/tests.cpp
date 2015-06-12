/*#include <gtest/gtest.h>
#include <map>
#include <stdexcept>

#include "dali/mat/Mat.h"
#include "dali/mat/math/MatOps.h"
#include "dali/execution/BeamSearch.h"

using std::make_tuple;
using std::map;
using std::string;
using std::tuple;
using std::vector;
using ::testing::AssertionResult;
using ::testing::AssertionSuccess;
using ::testing::AssertionFailure;

typedef float R;

struct AutomataState {
    typedef float REAL_t;

    int state;
    AutomataState() : state(0) {}
    AutomataState(int j) : state(j) {};
    Mat<REAL_t> predict() const {
        // automata machine
        // 0 goes to [0, 1, 2], 1 goes to [1, 2], and 2 goes to 2.
        auto new_states = Mat<REAL_t>(3, 1);
        if (state == 0) {
            (*new_states.w())(0, 0) = 0.5;
            (*new_states.w())(1, 0) = 0.3;
            (*new_states.w())(2, 0) = 0.2;
        }
        if (state == 1) {
            (*new_states.w())(1, 0) = 0.5;
            (*new_states.w())(2, 0) = 0.5;
        }
        if (state == 2) {
            assert(false);
        }
        return new_states;
    }
};

typedef AutomataState state_t;

TEST(beam_search, beam_search_automata) {
    typedef float REAL_t;

    // test whether going forward in beam yields right solutions.
    auto prob_next_states = [](state_t state) -> Mat<REAL_t> {
        auto new_probs = state.predict();
        return new_probs.log();
    };
    auto make_choice = [](state_t state, uint candidate) -> state_t {
        return state_t((int)candidate);
    };

    auto initial_state = state_t(0);
    int max_size = 20;

    // Take greedy approach:
    auto beam_search_results = beam_search::beam_search2<REAL_t, state_t>(
            initial_state,
            1,
            prob_next_states,
            make_choice,
            2,
            max_size);

    ASSERT_EQ(beam_search_results.size(), 1);
    auto comp = vector<uint>(max_size, 0);
    ASSERT_EQ(beam_search_results[0].solution.size(), max_size);
    ASSERT_EQ(beam_search_results[0].solution, comp);

    // Take less greedy approach:
    int beam_width = 7;
    beam_search_results = beam_search::beam_search2<REAL_t, state_t>(
            initial_state,
            beam_width,
            prob_next_states,
            make_choice,
            2,
            max_size);

    ASSERT_EQ(beam_search_results.size(), beam_width);
    ASSERT_EQ(beam_search_results[0].solution, vector<uint>{2});
    ASSERT_NEAR(std::exp(beam_search_results[0].score), 0.2, 1e-6);
    ASSERT_EQ(beam_search_results[1].solution, vector<uint>({1, 2}));
    ASSERT_NEAR(std::exp(beam_search_results[1].score), 0.3 * 0.5, 1e-6);
    ASSERT_EQ(beam_search_results[2].solution, vector<uint>({0, 2}));
    ASSERT_NEAR(std::exp(beam_search_results[2].score), 0.5 * 0.2, 1e-6);
    ASSERT_EQ(beam_search_results.back().solution.size(), max_size);
    ASSERT_NEAR(beam_search_results.back().score, std::log(0.5) * 20, 1e-5);
}

TEST(beam_search, beam_search_score_test) {
    using beam_search::beam_search2;
    using beam_search::BeamSearchResult;
    using utils::iter_to_str;
    typedef double REAL_t;
    typedef string state_t;
    typedef BeamSearchResult<REAL_t,state_t> result_t;
    typedef vector<result_t> results_t;

    const int MAX_LENGTH = 2;
    map<state_t, REAL_t> choices = {
        // initial_choices
        {"a", 0.6},
        {"b", 0.4},
        // after chosing a
        {"aa", 0.55},  // (total worth 0.33)
        {"ab", 0.45},  // (total worth 0.18)
        // after choosing b
        {"ba", 0.99},  // (total worth 0.495)
        {"bb", 0.11},  // (total worth 0.044)
    };

    // Above example is designed to demonstrate greedy solution,
    // as well as better optimal solution:
    // GREEDY:    (beam_width == 1) => "aa" worth 0.33
    // OPTIMAL:   (beam_width == 2) => "ba" worth 0.495
    auto res_aa = result_t("aa", {0,0}, std::log(0.6 * 0.55));
    auto res_ab = result_t("ab", {0,1}, std::log(0.6 * 0.45));
    auto res_ba = result_t("ba", {1,0}, std::log(0.4 * 0.99));
    auto res_bb = result_t("bb", {1,1}, std::log(0.4 * 0.11));

    auto initial_state = "";
    auto candidate_scores = [&](state_t state) -> Mat<REAL_t> {
        Mat<REAL_t> ret(2,1);
        (*ret.w())(0,0) = std::log(choices.at(state + "a"));
        (*ret.w())(1,0) = std::log(choices.at(state + "b"));

        return ret;
    };
    auto make_choice =
            [&](state_t prev_state, uint choice) -> state_t {
        return prev_state + (choice == 0 ? "a" : "b");
    };

    const uint FAKE_END_SYMBOL = 999;

    auto my_beam_search = [&](int beam_width) -> results_t {
        return beam_search2<REAL_t,state_t>(initial_state,
                                            beam_width,
                                            candidate_scores,
                                            make_choice,
                                            FAKE_END_SYMBOL,
                                            MAX_LENGTH);
    };

    auto beam_search_results_equal =
            [](results_t as, results_t bs) -> AssertionResult {
        if (as.size() != bs.size()) {
            return AssertionFailure()
                    << "Results of different sizes: "
                    << as.size() << " != " << bs.size();
        }
        for (int ridx = 0; ridx < as.size(); ++ridx) {
            auto a = as[ridx];
            auto b = bs[ridx];
            if (a.state != b.state)
                return AssertionFailure()
                    << "Result " << ridx + 1 << " has different states: <"
                    << a.state  << "> != <" << b.state << ">";
            if (a.solution != b.solution) {
                auto a_str = iter_to_str(a.solution.begin(),
                                         a.solution.end());
                auto b_str = iter_to_str(b.solution.begin(),
                                         b.solution.end());
                return AssertionFailure()
                    << "Result " << ridx + 1 << " has different solution: <"
                    << a_str << "> != <" << b_str << ">";
            }
            if (std::abs(a.score - b.score) > 1e-9)
                return AssertionFailure()
                    << "Result " << ridx + 1 << " has different score: "
                    << a.score  << " != " << b.score;
        }
        return AssertionSuccess();
    };

    EXPECT_THROW(my_beam_search(0),std::runtime_error);
    EXPECT_TRUE(beam_search_results_equal(
        my_beam_search(1),
        {res_aa}
    ));
    EXPECT_TRUE(beam_search_results_equal(
        my_beam_search(2),
        {res_ba, res_aa}
    ));
    EXPECT_TRUE(beam_search_results_equal(
        my_beam_search(4),
        {res_ba, res_aa, res_ab, res_bb}
    ));
    EXPECT_TRUE(beam_search_results_equal(
        my_beam_search(10),
        {res_ba, res_aa, res_ab, res_bb}
    ));
}

*/

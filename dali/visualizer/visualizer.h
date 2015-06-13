#ifndef DALI_VISUALIZER_VISUALIZER_H
#define DALI_VISUALIZER_VISUALIZER_H

#include <gflags/gflags.h>
#include <json11.hpp>
#ifdef DALI_USE_VISUALIZER
    #include <redox.hpp>
#endif
#include <memory>
#include <chrono>
#include <functional>
#include <string>

#include "dali/visualizer/EventQueue.h"
#include "dali/utils/core_utils.h"
#include "dali/mat/Mat.h"

// to import Throttled
#include "dali/utils/Reporting.h"

DECLARE_string(visualizer_hostname);
DECLARE_int32(visualizer_port);
DECLARE_string(visualizer);

// TODO: Szymon explain how this works
namespace visualizable {

    struct Visualizable {
        virtual json11::Json to_json() = 0;
    };

    template<typename R>
    struct Sentence : public Visualizable {
        std::vector<std::string> tokens;
        std::vector<R> weights;
        bool spaces = true;

        Sentence(std::vector<std::string> tokens);

        void set_weights(const std::vector<R>& _weights);
        void set_weights(const Mat<R>& _weights);

        virtual json11::Json to_json() override;
    };

    template<typename R>
    std::vector<std::shared_ptr<Sentence<R>>> sentence_vector(const std::vector<std::vector<std::string>>& vec);

    template<typename R>
    struct Sentences : public Visualizable {
        typedef std::shared_ptr<Sentence<R>> sentence_ptr;
        std::vector<sentence_ptr> sentences;
        std::vector<R> weights;

        Sentences(std::vector<sentence_ptr> sentences);
        Sentences(std::vector<std::vector<std::string>> vec);

        void set_weights(const std::vector<R>& _weights);
        void set_weights(const Mat<R>& _weights);

        virtual json11::Json to_json() override;
    };

    template<typename R>
    struct ParallelSentence : public Visualizable {
        typedef std::shared_ptr<Sentence<R>> sentence_ptr;
        sentence_ptr sentence1;
        sentence_ptr sentence2;
        ParallelSentence(sentence_ptr sentence1, sentence_ptr sentence2);
        virtual json11::Json to_json() override;
    };

    template<typename R>
    struct QA : public Visualizable {
        typedef std::shared_ptr<Sentence<R>> sentence_ptr;
        typedef std::shared_ptr<Visualizable> visualizable_ptr;
        visualizable_ptr context;
        sentence_ptr question;
        sentence_ptr answer;

        QA(visualizable_ptr context, sentence_ptr question, sentence_ptr answer);

        virtual json11::Json to_json() override;
    };

    struct GridLayout : public Visualizable {
        typedef std::shared_ptr<Visualizable> visualizable_ptr;

        std::vector<std::vector<visualizable_ptr>> grid;
        visualizable_ptr output;

        // adds a card in <column>-th column
        void add_in_column(int column, visualizable_ptr);

        virtual json11::Json to_json() override;
    };

    template<typename R>
    struct FiniteDistribution : public Visualizable {
        static const std::vector<R> empty_vec;
        std::vector<R> distribution;
        std::vector<R> scores;
        std::vector<std::string> labels;
        int top_picks;

        FiniteDistribution(const std::vector<R>& distribution,
                           const std::vector<R>& scores,
                           const std::vector<std::string>& labels,
                           int max_top_picks = -1);

        FiniteDistribution(const std::vector<R>& distribution,
               const std::vector<std::string>& labels,
               int max_top_picks = -1);

        virtual json11::Json to_json() override;
    };

    template<typename T>
    struct Probability: public Visualizable {
        T probability;

        Probability(T probability);

        virtual json11::Json to_json() override;
    };

    struct Message: public Visualizable {
        std::string content;

        Message(std::string content);

        virtual json11::Json to_json() override;
    };

    struct Tree: public Visualizable {
        std::string label;
        std::vector<std::shared_ptr<Tree>> children;

        Tree(std::string label);

        Tree(std::initializer_list<std::shared_ptr<Tree>> children);

        Tree(std::vector<std::shared_ptr<Tree>> children);

        Tree(std::string label, std::initializer_list<std::shared_ptr<Tree>> children);

        Tree(std::string label, std::vector<std::shared_ptr<Tree>> children);

        virtual json11::Json to_json() override;
    };
}

// TODO: explain what this does
class Visualizer {
    private:
        std::string my_namespace;
#ifdef DALI_USE_VISUALIZER
        std::shared_ptr<redox::Redox> rdx;
#endif
        EventQueue eq;
        EventQueue::repeating_t pinging;
        Throttled throttle;

        std::mutex connection_mutex;
        std::atomic<int> rdx_state;

        bool name_initialized = false;

        const bool rename_if_needed;

        bool update_name();

        void connected_callback(int status);
        bool ensure_connection();
    public:
        class duplicate_name_error : public std::runtime_error {
            public:
                explicit duplicate_name_error(const std::string& what_arg);
                explicit duplicate_name_error(const char* what_arg);
        };
        Visualizer(std::string my_namespace, bool rename_if_needed=false);

        void feed(const json11::Json& obj);
        void feed(const std::string& str);
        void throttled_feed(Throttled::Clock::duration time_between_feeds, std::function<json11::Json()> f);
};

#endif

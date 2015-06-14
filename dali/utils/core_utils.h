#ifndef CORE_UTILS_DALI_H
#define CORE_UTILS_DALI_H

#include <algorithm>
#include <atomic>
#include <cctype>
#include <chrono>
#include <cstring>
#include <dirent.h>
#include <errno.h>
#include <fstream>
#include <functional>
#include <gflags/gflags.h>
#include <iomanip>
#include <iostream>
#include <locale>
#include <map>
#include <memory>
#include <mutex>
#include <regex>
#include <ostream>
#include <random>
#include <set>
#include <sstream>
#include <string>
#include <sys/stat.h>
#include <sys/stat.h>
#include <unordered_map>
#include <vector>
#include "SQLiteCpp/Database.h"

// need to include Index for typedef eigen_index_segment

#include "dali/utils/gzstream.h"
#include "protobuf/corpus.pb.h"
#include "dali/utils/ThreadPool.h"

namespace Indexing {
    class Index;
}

// MACRO DEFINITIONS
#define ELOG(EXP) std::cout << #EXP "\t=\t" << (EXP) << std::endl
#define SELOG(STR,EXP) std::cout << #STR "\t=\t" << (EXP) << std::endl

// Useful for expanding macros. Obviously two levels of macro
// are needed....
#define STR(x) __THIS_IS_VERY_ABNOXIOUS(x)
#define __THIS_IS_VERY_ABNOXIOUS(tok) #tok

// Default writing mode useful for default argument to
// makedirs
#define DEFAULT_MODE S_IRWXU | S_IRGRP |  S_IXGRP | S_IROTH | S_IXOTH

std::ostream& operator<<(std::ostream&, const std::vector<std::string>&);
std::ostream& operator<<(std::ostream&, const std::map<std::string, uint>&);
std::ostream& operator<<(std::ostream&, const std::map<std::string, float>&);
std::ostream& operator<<(std::ostream&, const std::map<std::string, double>&);
std::ostream& operator<<(std::ostream&, const std::map<std::string, std::string>&);
std::ostream& operator<<(std::ostream&, const std::unordered_map<std::string, uint>&);
std::ostream& operator<<(std::ostream&, const std::unordered_map<std::string, float>&);
std::ostream& operator<<(std::ostream&, const std::unordered_map<std::string, double>&);
std::ostream& operator<<(std::ostream&, const std::unordered_map<std::string, std::string>&);

template<typename T>
std::ostream& operator<<(std::ostream&, const std::vector<T>&);

typedef std::vector<std::string> VS;

namespace utils {
        #ifndef NDEBUG
            std::string explain_mat_bug(const std::string&, const char*, const int&);
            template<typename T>
            bool contains_NaN(T);
        #endif

        /** Utility function to create directory tree */
        bool makedirs(const char* path, mode_t mode = DEFAULT_MODE);
        typedef std::vector<std::string> str_sequence;

        /**
        type tokenized_labeled_dataset
        a vector of vectors of string sequences (e.g. loading a tsv with many rows,
        and each row has whitespace separated words for each column)
        **/
        typedef std::vector<std::vector<str_sequence>> tokenized_labeled_dataset;
        typedef std::vector<std::pair<str_sequence, uint>> tokenized_uint_labeled_dataset;

        extern const char* end_symbol;
        extern const char* unknown_word_symbol;

        class Vocab {
                private:
                        void construct_word2index();
                        void add_unknown_word();
                public:
                    typedef uint ind_t;
                    ind_t unknown_word;
                    std::map<std::string, ind_t> word2index;
                    str_sequence index2word;

                    std::vector<ind_t> encode(const str_sequence& words, bool with_end_symbol = false) const;
                    std::vector<std::string> decode(Indexing::Index, bool remove_end_symbol = false) const;

                    // create vocabulary from many vectors of words. Vectors do not
                    // need to contain unique words. They need not be sorted.
                    // Standard use case is combining train/validate/test sets.
                    static Vocab from_many_nonunique(std::initializer_list<str_sequence>,
                                                     bool add_unknown_word=true);
                    Vocab();
                    Vocab(str_sequence&);
                    Vocab(str_sequence&, bool);
                    Vocab(str_sequence&&, bool);
                    Vocab(str_sequence&&);
                    size_t size() const;
        };

        class CharacterVocab {
                public:
                    typedef uint ind_t;
                    ind_t min_char;
                    ind_t max_char;

                    std::vector<ind_t>       encode(const str_sequence& words) const;
                    std::vector<std::string> decode(Indexing::Index) const;
                    std::vector<std::string> decode_characters(Indexing::Index) const;

                    CharacterVocab(int min_char, int max_char);
                    size_t size() const;
        };

        template<typename T>
        void tuple_sum(std::tuple<T, T>&, std::tuple<T,T>);
        /**
        Ends With
        ---------

        Check whether a string ends with the same contents as another.

        Inputs
        ------

        std::string const& full: where to look
        std::string const& ending: what to look for

        Outputs
        -------

        bool endswith : whether the second string ends the first.
        */
        bool endswith(const std::string&, const std::string&);
        /**
        Starts With
        -----------

        Check whether a string ends with the same contents as another.

        Inputs
        ------

        std::string const& full: where to look
        std::string const& beginning: what to look for

        Outputs
        -------

        bool startswith : whether the second string starts the first.

        */
        bool startswith(const std::string&, const std::string&);

        template<typename T>
        bool add_to_set(std::vector<T>&, T&);

        template<typename T>
        bool in_vector(const std::vector<T>&, const T&);

        template<typename T>
        std::vector<T> concatenate(std::initializer_list<std::vector<T>>);

        template<typename IN, typename OUT>
        std::vector<OUT> fmap(std::vector<IN> in_list,
                              std::function<OUT(IN)> f);


        /**
        Load Labeled Corpus
        -------------------

        Read text file line by line and extract pairs of labeled text
        by splitting on the first space character. Whatever is before
        the first space is the label and is stored second in the returned
        pairs, and after the space is the example, and this is returned
        first in the pairs.

        Inputs
        ------

        const std::string& fname : labeled corpus file

        Outputs
        -------

        std::vector<std::pair<std::string, std::string>> pairs : labeled corpus pairs

        **/
        std::vector<std::pair<std::string, std::string>> load_labeled_corpus(const std::string&);


        std::vector<str_sequence> load_tokenized_unlabeled_corpus(const std::string&);
        str_sequence tokenize(const std::string&);
        str_sequence get_vocabulary(const tokenized_labeled_dataset&, int);
        str_sequence get_vocabulary(const std::vector<str_sequence>&, int);
        str_sequence get_vocabulary(const tokenized_uint_labeled_dataset&, int);
        str_sequence get_label_vocabulary(const tokenized_labeled_dataset&);

        template<typename T>
        void load_corpus_from_stream(Corpus& corpus, T& stream);
        Corpus load_corpus_protobuff(const std::string&);
        /**
        Load Protobuff Dataset
        ----------------------

        Load a set of protocol buffer serialized files from ordinary
        or gzipped files, and conver their labels from an index
        to their string representation using an index2label mapping.

        Inputs
        ------

        std::string directory : where the protocol buffer files are stored
        const std::vector<std::string>& index2label : mapping from numericals to
                                                      string labels

        Outputs
        -------

        utils::tokenized_multilabeled_dataset dataset : pairs of tokenized strings
                                                        and vector of string labels

        **/
        tokenized_labeled_dataset load_protobuff_dataset(std::string, const std::vector<std::string>&);
        tokenized_labeled_dataset load_protobuff_dataset(SQLite::Statement& query, const std::vector<std::string>&, int max_elements = 100, int column = 0);

        std::string& trim(std::string&);
        std::string& ltrim(std::string&);
        std::string& rtrim(std::string&);

        void map_to_file(const std::map<std::string, str_sequence>&, const std::string&);

        void ensure_directory(std::string&);

        std::vector<std::string> split_str(const std::string&, const std::string&);

        /**
        Text To Map
        -----------
        Read a text file, extract all key value pairs and
        ignore markdown decoration characters such as =, -,
        and #

        Inputs
        ------
        std::string fname : the file to read

        Outputs
        -------
        std::map<string, std::vector<string> > map : the extracted key value pairs.

        **/
        std::map<std::string, str_sequence> text_to_map(const std::string&);
        template<typename T, typename K>
        void stream_to_hashmap(T&, std::map<std::string, K>&);
        template<typename T>
        std::map<std::string, T> text_to_hashmap(const std::string&);

        template<typename T>
        void stream_to_list(T&, str_sequence&);

        str_sequence load_list(const std::string&);
        void save_list(const std::vector<std::string>& list, std::string fname, std::ios_base::openmode = std::ios::out);

        template<typename T>
        void save_list_to_stream(const std::vector<std::string>& list, T&);

        template<typename T>
        void stream_to_redirection_list(T&, std::map<std::string, std::string>&);

        std::map<std::string, std::string> load_redirection_list(const std::string&);

        template<typename T>
        void stream_to_redirection_list(T&, std::map<std::string, std::string>&, std::function<std::string(std::string&&)>&, int num_threads = 1);

        std::map<std::string, std::string> load_redirection_list(const std::string&, std::function<std::string(std::string&&)>&&, int num_threads = 1);

        /**
        randint
        -------
        Sample integers from a uniform distribution between (and including)
        lower and upper int values.

        Inputs
        ------
        int lower
        int upper

        Outputs
        -------
        int sample
        **/
        int randint(int lower, int upper);


        double randdouble(double lower=0.0, double upper=1.0);


        /**
        Is Gzip ?
        ---------
        Check whether a file has the header information for a GZIP
        file or not.

        Inputs
        ------
        fname : potential gzip file's path

        Outputs
        -------
        whether header matches gzip header
        **/
        bool is_gzip(const std::string& fname);

        /* Used for Eigen CwiseExpr - wraps a lambda expression */
        template<typename T>
        struct LambdaOperator {
            std::function<T(T)> lambda_expr;
            LambdaOperator(std::function<T(T)> lambda_expr);
            T operator() (T) const;
        };

        template<typename T>
        struct sigmoid_operator {
            T operator() (T) const;
        };

        template<typename T>
        struct log_or_zero {
            T operator() (T) const;
        };

        template<typename T>
        struct steep_sigmoid_operator {
            // Sourced from Indico's Passage library's activations
            // (Theano Python module)
            // https://github.com/IndicoDataSolutions/Passage
            // motivation in this Youtube video:
            // https://www.youtube.com/watch?v=VINCQghQRuM
            const T aggressiveness;
            steep_sigmoid_operator(T aggressiveness);
            T operator() (T) const;
        };
        template<typename T>
        struct tanh_operator {
                T operator() (T) const;
        };
        template<typename T>
        struct relu_operator {
                T operator() (T) const;
        };
        template<typename T>
        struct sign_operator {
                T operator() (T) const;
        };
        template<typename T>
        struct max_operator {
                T operator() (T) const;
        };
        template<typename T>
        struct dtanh_operator {
                T operator() (T) const;
        };

        template<typename T>
    T from_string(const std::string&);

    bool is_number(const std::string&);

    template<typename T>
    void assert_map_has_key(std::map<std::string, T>&, const std::string&);

    /**
    Split
    -----

    Split a string at a character into a vector of strings.
    Optionally choose to keep empty strings, e.g.:

        > utils::split("//hello//world", '/');

        => vector<string>({"hello", "world"})

    vs.

        > utils::split("//hello//world", '/', true);

        => vector<string>({"", "", "hello", "", "world", "", ""})

    Inputs
    ------
      std::string& text : text to split using char
                 char c : character used to cut up text
bool keep_empty_strings : keep empty strings [see above], defaults to false.

    Outputs
    -------

    std::vector<std::string> split : split string sequence (tokenized)

    **/
    str_sequence split(const std::string &, char, bool keep_empty_strings = false);

    std::string join(const std::vector<std::string>& vs,
                     const std::string& in_between="");

    /**
    Triggers To Strings
    -------------------

    Convert triggers from an example to their
    string representation using an index2label
    string vector.

    Inputs
    ------

    const google::protobuf::RepeatedPtrField<Example::Trigger>& triggers : list of Trigger protobuff objects
    const vector<string>& index2target : mapping from numerical to string representation

    Outputs
    -------

    std::vector<std::string> data : the strings corresponding to the trigger targets
    **/
    str_sequence triggers_to_strings(const google::protobuf::RepeatedPtrField<Example::Trigger>&, const str_sequence&);

    template <typename T>
    std::vector<size_t> argsort(const std::vector<T> &);

    str_sequence listdir(const std::string&);

    std::vector<size_t> random_arange(size_t);

    std::vector<std::vector<size_t>> random_minibatches(size_t total_elements, size_t minibatch_size);

    std::vector<uint> arange(uint, uint);

    template <class T> inline void hash_combine(std::size_t &, const T &);

    bool file_exists(const std::string&);

    std::string dir_parent(const std::string& path, int levels_up = 1);

    std::string dir_join(const std::vector<std::string>&);

    template<typename T>
    std::vector<T> normalize_weights(const std::vector<T>& weights);


    // returns candidate from candidates with longest common prefix with input.
    // if longest prefix is ambiguous, returns the first one from the list.
    std::string prefix_match(std::vector<std::string> candidates, std::string input);

    /**
    Exit With Message
    -----------------
    Exit the program with a message printed to std::cerr

    Inputs
    ------
    error message
    error_code : code for the error, defaults to 1

    **/
    void exit_with_message(const std::string&, int error_code = 1);

    bool validate_flag_nonempty(const char* flagname, const std::string& value);

    template<typename T>
    T vsum(const std::vector<T>& vec);

    template<typename T>
    std::vector<T> reversed(const std::vector<T>& v);

    class ThreadError {
        /* Small utility class used to safely average error contributions
           from different threads. */
        public:
            const int num_threads;
            std::vector<double> thread_error;
            std::vector<int>    thread_error_updates;

            ThreadError(int num_threads);

            // should be called from a thread (this internally uses thread pool's number.)
            void update(double error);
            double this_thread_average();
            double average();
            void reset();

    };


    class Timer {
        typedef std::chrono::system_clock clock_t;

        static std::unordered_map<std::string, std::atomic<int>> timers;
        static std::mutex timers_mutex;

        std::string name;
        bool stopped;
        bool started;
        std::chrono::time_point<clock_t> start_time;

        public:
            // creates timer and starts measuring time.
            Timer(std::string name, bool autostart=true);
            // destroys timer and stops counting if the timer was not previously stopped.
            ~Timer();

            // explicitly start the timer
            void start();
            // explicitly stop the timer
            void stop();

            static void report();
    };

    class MS {
        public:
            std::stringstream stream;
            operator std::string() const { return stream.str(); }

            template<class T>
            MS& operator<<(T const& VAR) { stream << VAR; return *this; }
    };

    template<typename T>
    std::string iter_to_str(T begin, T end) {
        std::stringstream ss;
        bool first = true;
        for (; begin != end; begin++)
        {
            if (!first)
                ss << ", ";
            ss << *begin;
            first = false;
        }
        return ss.str();
    }

    std::string capitalize(const std::string& s);

    void assert2(bool condition);
    void assert2(bool condition, std::string message);

    extern std::string green;
    extern std::string red;
    extern std::string blue;
    extern std::string yellow;
    extern std::string cyan;
    extern std::string black;
    extern std::string reset_color;
    extern std::string bold;
}

#endif

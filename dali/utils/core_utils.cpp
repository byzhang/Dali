#include "core_utils.h"

#include "dali/mat/Mat.h"
#include "dali/mat/Index.h"


using std::vector;
using std::string;
using std::ifstream;
using std::stringstream;
using std::ofstream;
using std::set;
using std::make_shared;
using std::function;
using std::initializer_list;

const char* utils::end_symbol          = "**END**";
const char* utils::unknown_word_symbol = "███████";

std::ostream &operator <<(std::ostream &os, const vector<string> &v) {
   if (v.size() == 0) return os << "[]";
   os << "[\"";
   std::copy(v.begin(), v.end() - 1, std::ostream_iterator<string>(os, "\", \""));
   return os << v.back() << "\"]";
}

std::ostream &operator <<(std::ostream &os, const std::map<string, uint> &v) {
   if (v.size() == 0) return os << "{}";
   os << "{\n";
   for (auto& kv : v) {
       os << "\"" << kv.first << "\" => " << kv.second << ",\n";
   }
   return os << "}";
}

std::ostream &operator <<(std::ostream &os, const std::unordered_map<string, uint> &v) {
   if (v.size() == 0) return os << "{}";
   os << "{\n";
   for (auto& kv : v) {
       os << "\"" << kv.first << "\" => " << kv.second << ",\n";
   }
   return os << "}";
}
std::ostream &operator <<(std::ostream &os, const std::unordered_map<string, float> &v) {
   if (v.size() == 0) return os << "{}";
   os << "{\n";
   for (auto& kv : v) {
       os << "\"" << kv.first << "\" => " << kv.second << ",\n";
   }
   return os << "}";
}
std::ostream &operator <<(std::ostream &os, const std::unordered_map<string, double> &v) {
   if (v.size() == 0) return os << "{}";
   os << "{\n";
   for (auto& kv : v) {
       os << "\"" << kv.first << "\" => " << kv.second << ",\n";
   }
   return os << "}";
}
std::ostream &operator <<(std::ostream &os, const std::map<string, float> &v) {
   if (v.size() == 0) return os << "{}";
   os << "{\n";
   for (auto& kv : v) {
       os << "\"" << kv.first << "\" => " << kv.second << ",\n";
   }
   return os << "}";
}
std::ostream &operator <<(std::ostream &os, const std::map<string, double> &v) {
   if (v.size() == 0) return os << "{}";
   os << "{\n";
   for (auto& kv : v) {
       os << "\"" << kv.first << "\" => " << kv.second << ",\n";
   }
   return os << "}";
}

std::ostream &operator <<(std::ostream &os, const std::map<string, string> &v) {
   if (v.size() == 0) return os << "{}";
   os << "{\n";
   for (auto& kv : v) {
       os << "\"" << kv.first << "\" => \"" << kv.second << "\",\n";
   }
   return os << "}";
}
std::ostream &operator <<(std::ostream &os, const std::unordered_map<string, string> &v) {
   if (v.size() == 0) return os << "{}";
   os << "{\n";
   for (auto& kv : v) {
       os << "\"" << kv.first << "\" => \"" << kv.second << "\",\n";
   }
   return os << "}";
}

template<typename T>
std::ostream& operator<<(std::ostream& os, const vector<T>& v) {
        if (v.size() == 0) return os << "[]";
        os << "[";
        for (auto& f : v)
                os << std::fixed
                   << std::setw( 7 ) // keep 7 digits
                   << std::setprecision( 3 ) // use 3 decimals
                   << std::setfill( ' ' ) // pad values with blanks this->w(i,j)
                   << f << " ";
        return os << "]";
}

template std::ostream& operator<< <double>(std::ostream& strm, const vector<double>& a);
template std::ostream& operator<< <float>(std::ostream& strm, const vector<float>& a);
template std::ostream& operator<< <uint>(std::ostream& strm, const vector<uint>& a);
template std::ostream& operator<< <int>(std::ostream& strm, const vector<int>& a);
template std::ostream& operator<< <size_t>(std::ostream& strm, const vector<size_t>& a);

namespace utils {

    #ifndef NDEBUG
    string explain_mat_bug(const string& mat_name, const char* file, const int& line) {
        stringstream ss;
        ss << "Matrix \"" << mat_name << "\" has NaNs in file:\"" << file << "\" and line: " << line;
        return ss.str();
    }
    template<typename T>
    bool contains_NaN(T val) {return !(val == val);}

    template bool contains_NaN(float);
    template bool contains_NaN(double);
    #endif

    vector<size_t> random_arange(size_t size) {
        vector<size_t> indices(size);
        for (size_t i=0; i < size;i++) indices[i] = i;
        std::random_shuffle( indices.begin(), indices.end() );
        return indices;
    }

    vector<vector<size_t>> random_minibatches(size_t total_elements, size_t minibatch_size) {
        vector<size_t> training_order = utils::random_arange(total_elements);
        int num_minibatches = training_order.size() / minibatch_size;
        vector<vector<size_t>> minibatches(num_minibatches);
        for (int tidx = 0; tidx < total_elements; ++tidx) {
            minibatches[tidx%num_minibatches].push_back(training_order[tidx]);
        }
        return minibatches;
    }

    vector<uint> arange(uint start, uint end) {
        vector<uint> indices(end - start);
        for (uint i=0; i < indices.size();i++) indices[i] = i;
        return indices;
    }

    void ensure_directory (std::string& dirname) {
        if (dirname.back() != '/') dirname += "/";
    }

    vector<string> split(const std::string &s, char delim, bool keep_empty_strings) {
        std::vector<std::string> elems;
        std::stringstream ss(s);
        string item;
        while (std::getline(ss, item, delim))
            if (!item.empty() || keep_empty_strings)
                elems.push_back(item);
        return elems;
    }

    string join(const vector<string>& vs, const string& in_between) {
        std::stringstream ss;
        for (int i = 0; i < vs.size(); ++i) {
            ss << vs[i];
            if (i + 1 != vs.size())
                ss << in_between;
        }
        return ss.str();
    }

    template<typename T>
    void load_corpus_from_stream(Corpus& corpus, T& stream) {
        corpus.ParseFromIstream(&stream);
    }

    template void load_corpus_from_stream(Corpus&, igzstream&);
    template void load_corpus_from_stream(Corpus&, std::fstream&);
    template void load_corpus_from_stream(Corpus&, std::stringstream&);
    template void load_corpus_from_stream(Corpus&, std::istream&);

    Corpus load_corpus_protobuff(const std::string& path) {
        Corpus corpus;
        if (is_gzip(path)) {
            igzstream fpgz(path.c_str(), std::ios::in | std::ios::binary);
            load_corpus_from_stream(corpus, fpgz);
        } else {
            std::fstream fp(path, std::ios::in | std::ios::binary);
            load_corpus_from_stream(corpus, fp);
        }
        return corpus;
    }

    template<typename T>
    bool add_to_set(vector<T>& set, T& el) {
        if(std::find(set.begin(), set.end(), el) != set.end()) {
                return false;
        } else {
                set.emplace_back(el);
                return true;
        }
    }

    template bool add_to_set(vector<int>&,    int&);
    template bool add_to_set(vector<uint>&,   uint&);
    template bool add_to_set(vector<string>&, string&);


    template<typename T>
    bool in_vector(const std::vector<T>& set, const T& el) {
        return std::find(set.begin(), set.end(), el) != set.end();
    }

    template bool in_vector(const vector<int>&,    const int&);
    template bool in_vector(const vector<uint>&,   const uint&);
    template bool in_vector(const vector<string>&, const string&);

    template<typename T>
    vector<T> concatenate(initializer_list<vector<T>> lists) {
        vector<T> concatenated_list;
        for (auto& list: lists) {
            for (auto& el: list) {
                concatenated_list.emplace_back(el);
            }
        }
        return concatenated_list;
    }

    template vector<int> concatenate(initializer_list<vector<int>>);
    template vector<Mat<float>> concatenate(std::initializer_list<vector<Mat<float>>>);
    template vector<Mat<double>> concatenate(std::initializer_list<vector<Mat<double>>>);

    template<typename IN, typename OUT>
    vector<OUT> fmap(vector<IN> in_list, function<OUT(IN)> f) {
        vector<OUT> out_list;
        for (IN& in_element: in_list)
            out_list.push_back(f(in_element));
        return out_list;
    }

    template vector<string> fmap(vector<int>, function<string(int)>);

    template<typename T>
    void tuple_sum(std::tuple<T, T>& A, std::tuple<T, T> B) {
            std::get<0>(A) += std::get<0>(B);
            std::get<1>(A) += std::get<1>(B);
    }

    template void tuple_sum(std::tuple<float, float>&, std::tuple<float, float>);
    template void tuple_sum(std::tuple<double, double>&, std::tuple<double, double>);
    template void tuple_sum(std::tuple<int, int>&, std::tuple<int, int>);
    template void tuple_sum(std::tuple<uint, uint>&, std::tuple<uint, uint>);

    template<typename T>
    void assert_map_has_key(std::map<string, T>& map, const string& key) {
            if (map.count(key) < 1) {
                    stringstream error_msg;
                    error_msg << "Map is missing the following key : \"" << key << "\".";
                    throw std::runtime_error(error_msg.str());
            }
    }

    template void assert_map_has_key(std::map<string, string>&, const string&);
    template void assert_map_has_key(std::map<string, vector<string>>&, const string&);

    vector<string> listdir(const string& folder) {
            vector<string> filenames;
            DIR *dp;
        struct dirent *dirp;
        if ((dp  = opendir(folder.c_str())) == nullptr) {
            stringstream error_msg;
                    error_msg << "Error: could not open directory \"" << folder << "\"";
                    throw std::runtime_error(error_msg.str());
        }
        // list all contents of directory:
        while ((dirp = readdir(dp)) != nullptr)
            // exclude "current directory" (.) and "parent directory" (..)
            if (std::strcmp(dirp->d_name, ".") != 0 && std::strcmp(dirp->d_name, "..") != 0)
                    filenames.emplace_back(dirp->d_name);
        closedir(dp);
        return filenames;
    }

    vector<string> split_str(const string& original, const string& delimiter) {
            std::vector<std::string> tokens;
            auto delimiter_ptr = delimiter.begin();
            stringstream ss(original);
            int inside = 0;
            std::vector<char> token;
            char ch;
            while (ss) {
                    ch = ss.get();
                    if (ch == *delimiter_ptr) {
                            delimiter_ptr++;
                            inside++;
                            if (delimiter_ptr == delimiter.end()) {
                                    tokens.emplace_back(token.begin(), token.end());
                                    token.clear();
                                    inside = 0;
                                    delimiter_ptr = delimiter.begin();
                            }
                    } else {
                            if (inside > 0) {
                                    token.insert(token.end(), delimiter.begin(), delimiter_ptr);
                                    delimiter_ptr = delimiter.begin();
                                    inside = 0;
                            } else {
                                    token.push_back(ch);
                            }
                    }
            }
            if (inside > 0) {
                    token.insert(token.end(), delimiter.begin(), delimiter_ptr);
                    tokens.emplace_back(token.begin(), token.end());
                    token.clear();
            } else {
                    if (token.size() > 0)
                            tokens.emplace_back(token.begin(), token.end()-1);
            }
            return tokens;
    }
    std::map<string, std::vector<string>> text_to_map(const string& fname) {
            ifstream infile(fname);
            string line;
            const char space = ' ';
            std::map<string, std::vector<string>> map;
            while (std::getline(infile, line)) {
                    if (*line.begin() != '=' && *line.begin() != '-' && *line.begin() != '#') {
                            const auto tokens = utils::split(line, space);
                            if (tokens.size() > 1) {
                                    auto ptr = tokens.begin() + 1;
                                    while( ptr != tokens.end()) {
                                            map[tokens[0]].emplace_back(*(ptr++));
                                    }
                            }
                    }
            }
            return map;
    }

    template<typename T, typename K>
    void stream_to_hashmap(T& infile, std::map<string, K>& map) {
            string line;
            const char space = ' ';
            while (std::getline(infile, line)) {
                    const auto tokens = utils::split(line, space);
                    if (tokens.size() > 1)
                            map[tokens[0]] = from_string<K>(tokens[1]);
            }
    }

    template<typename T>
    std::map<string, T> text_to_hashmap(const string& fname) {
            std::map<string, T> map;
            if (is_gzip(fname)) {
                    igzstream fpgz(fname.c_str(), std::ios::in | std::ios::binary);
                    stream_to_hashmap(fpgz, map);
            } else {
                    std::fstream fp(fname, std::ios::in | std::ios::binary);
                    stream_to_hashmap(fp, map);
            }
            return map;
    }

    template<typename T>
    void stream_to_list(T& fp, vector<string>& list) {
            string line;
            while (std::getline(fp, line))
                    list.emplace_back(line);
    }
    template void stream_to_list(stringstream&, vector<string>&);

    vector<string> load_list(const string& fname) {
        vector<string> list;
        if (is_gzip(fname)) {
            igzstream fpgz(fname.c_str(), std::ios::in | std::ios::binary);
            stream_to_list(fpgz, list);
        } else {
            std::fstream fp(fname, std::ios::in | std::ios::binary);
            stream_to_list(fp, list);
        }
        return list;
    }

    template<typename T>
    void save_list_to_stream(const vector<string>& list, T& fp) {
        for (auto& el : list) {
            fp << el << "\n";
        }
    }

    void save_list(const vector<string>& list, string fname, std::ios_base::openmode mode) {
        if (endswith(fname, ".gz")) {
            ogzstream fpgz(fname.c_str(), mode);
            save_list_to_stream(list, fpgz);
        } else {
            ofstream fp(fname, mode);
            save_list_to_stream(list, fp);
        }
    }

    template<typename T>
    void stream_to_redirection_list(T& fp, std::map<string, string>& mapping, std::function<std::string(std::string&&)>& preprocessor, int num_threads) {
        string line;
        const char dash = '-';
        const char arrow = '>';
        bool saw_dash = false;
        auto checker = [&saw_dash, &arrow, &dash](const char& ch) {
            if (saw_dash) {
                if (ch == arrow) {
                    return true;
                } else {
                    saw_dash = (ch == dash);
                    return false;
                }
            } else {
                saw_dash = (ch == dash);
                return false;
            }
        };
        if (num_threads > 1) {
            ThreadPool pool(num_threads);
            while (std::getline(fp, line)) {
                pool.run([&mapping, &preprocessor, &checker, line]() {
                    auto pos_end_arrow = std::find_if(line.begin(), line.end(), checker);
                    if (pos_end_arrow != line.end()) {
                        mapping.emplace(
                            std::piecewise_construct,
                            std::forward_as_tuple(
                                preprocessor(
                                    std::string(
                                        line.begin(),
                                        pos_end_arrow-1
                                    )
                                )
                            ),
                            std::forward_as_tuple(
                                preprocessor(
                                    std::string(
                                        pos_end_arrow+1,
                                        line.end()
                                    )
                                )
                            )
                        );
                    }
                });
            }
        } else {
            while (std::getline(fp, line)) {
                auto pos_end_arrow = std::find_if(line.begin(), line.end(), checker);
                if (pos_end_arrow != line.end()) {
                    mapping.emplace(
                        std::piecewise_construct,
                        std::forward_as_tuple( preprocessor(std::string(line.begin(), pos_end_arrow-1))),
                        std::forward_as_tuple( preprocessor(std::string(pos_end_arrow+1, line.end())))
                    );
                }
            }
        }
    }

    template<typename T>
    void stream_to_redirection_list(T& fp, std::map<string, string>& mapping) {
        string line;
        const char dash = '-';
        const char arrow = '>';
        bool saw_dash = false;
        auto checker = [&saw_dash, &arrow, &dash](const char& ch) {
            if (saw_dash) {
                if (ch == arrow) {
                    return true;
                } else {
                    saw_dash = (ch == dash);
                    return false;
                }
            } else {
                saw_dash = (ch == dash);
                return false;
            }
        };
        while (std::getline(fp, line)) {
            auto pos_end_arrow = std::find_if(line.begin(), line.end(), checker);
            if (pos_end_arrow != line.end()) {
                mapping.emplace(
                    std::piecewise_construct,
                    std::forward_as_tuple( line.begin(), pos_end_arrow-1),
                    std::forward_as_tuple( pos_end_arrow+1, line.end())
                );
            }
        }
    }

    template void stream_to_redirection_list(stringstream&, std::map<string, string>&, std::function<std::string(std::string&&)>&, int);
    template void stream_to_redirection_list(stringstream&, std::map<string, string>&);

    std::map<string, string> load_redirection_list(const string& fname, std::function<std::string(std::string&&)>&& preprocessor, int num_threads) {
        std::map<string, string> mapping;
        if (is_gzip(fname)) {
            igzstream fpgz(fname.c_str(), std::ios::in | std::ios::binary);
            stream_to_redirection_list(fpgz, mapping, preprocessor, num_threads);
        } else {
            std::fstream fp(fname, std::ios::in | std::ios::binary);
            stream_to_redirection_list(fp, mapping, preprocessor, num_threads);
        }
        return mapping;
    }

    std::map<string, string> load_redirection_list(const string& fname) {
        std::map<string, string> mapping;
        if (is_gzip(fname)) {
            igzstream fpgz(fname.c_str(), std::ios::in | std::ios::binary);
            stream_to_redirection_list(fpgz, mapping);
        } else {
            std::fstream fp(fname, std::ios::in | std::ios::binary);
            stream_to_redirection_list(fp, mapping);
        }
        return mapping;
    }

    template std::map<string, string> text_to_hashmap(const string&);
    template std::map<string, int>    text_to_hashmap(const string&);
    template std::map<string, float>  text_to_hashmap(const string&);
    template std::map<string, double> text_to_hashmap(const string&);
    template std::map<string, uint>   text_to_hashmap(const string&);

    void map_to_file(const std::map<string, std::vector<string>>& map, const string& fname) {
        ofstream fp;
        fp.open(fname.c_str(), std::ios::out);
        for (auto& kv : map) {
            fp << kv.first;
            for (auto& v: kv.second)
                fp << " " << v;
            fp << "\n";
        }
    }

    vector<std::pair<string, string>> load_labeled_corpus(const string& fname) {
        ifstream fp(fname.c_str());
        string l;
        const char space = ' ';
        vector<std::pair<string, string>> pairs;
        string::size_type n;
        while (std::getline(fp, l)) {
            n = l.find(space);
            pairs.emplace_back(std::piecewise_construct,
                std::forward_as_tuple(l.begin() + n + 1, l.end()),
                std::forward_as_tuple(l.begin(), l.begin() + n)
            );
        }
        return pairs;
    }

    vector<string> triggers_to_strings(const google::protobuf::RepeatedPtrField<Example::Trigger>& triggers, const vector<string>& index2target) {
        vector<string> data;
        data.reserve(triggers.size());
        for (auto& trig : triggers)
            if (trig.id() < index2target.size())
                data.emplace_back(index2target[trig.id()]);
        return data;
    }

    tokenized_labeled_dataset load_protobuff_dataset(string directory, const vector<string>& index2label) {
        ensure_directory(directory);
        auto files = listdir(directory);
        tokenized_labeled_dataset dataset;
        for (auto& file : files) {
            auto corpus = load_corpus_protobuff(directory + file);
            for (auto& example : corpus.example()) {
                dataset.emplace_back(std::initializer_list<vector<string>>({
                    vector<string>(example.words().begin(), example.words().end()),
                    triggers_to_strings(example.trigger(), index2label)
                }));
            }
        }
        return dataset;
    }

    tokenized_labeled_dataset load_protobuff_dataset(
        SQLite::Statement& query,
        const vector<string>& index2label,
        int num_elements,
        int column) {
        int els_seen = 0;
        tokenized_labeled_dataset dataset;
        while (query.executeStep()) {
            const char* protobuff_serialized = query.getColumn(column);
            stringstream ss(protobuff_serialized);
            Corpus corpus;
            load_corpus_from_stream(corpus, ss);
            for (auto& example : corpus.example()) {
                dataset.emplace_back(std::initializer_list<vector<string>>{
                    vector<string>(example.words().begin(), example.words().end()),
                    triggers_to_strings(example.trigger(), index2label)
                });
                ++els_seen;
            }
            if (els_seen >= num_elements) {
                break;
            }
        }
        return dataset;
    }

    vector<string> tokenize(const string& s) {
        stringstream ss(s);
        std::istream_iterator<string> begin(ss);
        std::istream_iterator<string> end;
        return vector<string>(begin, end);
    }

    vector<vector<string>> load_tokenized_unlabeled_corpus(const string& fname) {
        ifstream fp(fname.c_str());
        string l;
        vector<vector<string>> list;
        while (std::getline(fp, l))
            list.emplace_back(tokenize(string(l.begin(), l.end())));
        return list;
    }

    vector<string> get_vocabulary(const tokenized_labeled_dataset& examples, int min_occurence) {
        std::map<string, uint> word_occurences;
        string word;
        for (auto& example : examples)
            for (auto& word : example[0]) word_occurences[word] += 1;
        vector<string> list;
        for (auto& key_val : word_occurences)
            if (key_val.second >= min_occurence)
                list.emplace_back(key_val.first);
        list.emplace_back(utils::end_symbol);
        return list;
    }

    vector<string> get_vocabulary(const vector<vector<string>>& examples, int min_occurence) {
        std::map<string, uint> word_occurences;
        string word;
        for (auto& example : examples) {
            for (auto& word : example) {
                word_occurences[word] += 1;
            }
        }
        vector<string> list;
        for (auto& key_val : word_occurences) {
            if (key_val.second >= min_occurence) {
                list.emplace_back(key_val.first);
            }
        }
        list.emplace_back(utils::end_symbol);
        return list;
    }

    vector<string> get_vocabulary(const tokenized_uint_labeled_dataset& examples, int min_occurence) {
        std::map<string, uint> word_occurences;
        string word;
        for (auto& example : examples) {
            for (auto& word : example.first) {
                word_occurences[word] += 1;
            }
        }
        vector<string> list;
        for (auto& key_val : word_occurences) {
            if (key_val.second >= min_occurence) {
                list.emplace_back(key_val.first);
            }
        }
        list.emplace_back(utils::end_symbol);
        return list;
    }

    vector<string> get_label_vocabulary(const tokenized_labeled_dataset& examples) {
        std::set<string> labels;
        string word;
        for (auto& example : examples) {
            assert2(example.size() > 1, "Examples must have at least 2 columns.");
            labels.insert(example[1].begin(), example[1].end());
        }
        return vector<string>(labels.begin(), labels.end());
    }

    // Trimming text from StackOverflow:
    // http://stackoverflow.com/questions/216823/whats-the-best-way-to-trim-stdstring

    // trim from start
    std::string &ltrim(std::string &s) {
        s.erase(s.begin(), std::find_if(s.begin(), s.end(), std::not1(std::ptr_fun<int, int>(std::isspace))));
        return s;
    }

    // trim from end
    std::string &rtrim(std::string &s) {
        s.erase(std::find_if(s.rbegin(), s.rend(), std::not1(std::ptr_fun<int, int>(std::isspace))).base(), s.end());
        return s;
    }

    // trim from both ends
    std::string &trim(std::string &s) {
        return ltrim(rtrim(s));
    }

    // From this StackOverflow:
    // http://stackoverflow.com/questions/675039/how-can-i-create-directory-tree-in-c-linux
    bool makedirs(const char* path, mode_t mode) {
        // const cast for hack
        char* p = const_cast<char*>(path);

        // Do mkdir for each slash until end of string or error
        while (*p != '\0') {
            // Skip first character
            p++;

            // Find first slash or end
            while(*p != '\0' && *p != '/') p++;

            // Remember value from p
            char v = *p;

            // Write end of string at p
            *p = '\0';

            // Create folder from path to '\0' inserted at p
            if(mkdir(path, mode) == -1 && errno != EEXIST) {
                *p = v;
                return false;
            }

            // Restore path to it's former glory
            *p = v;
        }
        return true;
    }

    int randint(int lower, int upper) {
        assert2 (lower <= upper, "Lower bound must be smaller than upper bound.");
        std::default_random_engine generator;
        std::uniform_int_distribution<int> distribution(lower, upper);
        std::random_device rd;
        generator.seed(rd());
        return distribution(generator);
    }

    double randdouble(double lower, double upper) {
        assert2 (lower <= upper, "Lower bound must be smaller than upper bound.");
        std::default_random_engine generator;
        std::uniform_real_distribution<double> distribution(lower, upper);
        std::random_device rd;
        generator.seed(rd());
        return distribution(generator);
    }

    void Vocab::construct_word2index() {
        uint i = 0;
        for (auto& s : index2word)
            word2index[s] = i++;
    }
    void Vocab::add_unknown_word() {
        index2word.emplace_back(unknown_word_symbol);
        word2index[unknown_word_symbol] = index2word.size() - 1;
        unknown_word = index2word.size() - 1;
    }

    size_t Vocab::size() const {
        return index2word.size();
    }

    vector<typename Vocab::ind_t> Vocab::encode(const str_sequence& words, bool with_end_symbol) const {
        vector<ind_t> result;
        result.reserve(words.size() + (with_end_symbol ? 1 : 0));
        std::transform(words.begin(), words.end(),
                       std::back_inserter(result), [this](const string& word) {
            if (word2index.find(word) == word2index.end()) {
                return unknown_word;
            } else {
                return word2index.at(word);
            }
        });
        if (with_end_symbol) {
            result.emplace_back( word2index.at(utils::end_symbol) );
        }
        return result;
    }

    vector<string> Vocab::decode(Indexing::Index indices, bool remove_end_symbol) const {
        vector<string> result;
        result.reserve(indices.size());
        // either the decoding must remove the end symbol
        // if there is one, or we assume there is none to remove.
        bool has_end_symbol = remove_end_symbol ?
            indices[indices.size() - 1] == word2index.at(utils::end_symbol) :
            false;
        auto index_end = indices.data() + indices.size();
        if (has_end_symbol) index_end--;
        std::transform(indices.data(), index_end,
                       std::back_inserter(result), [this](const ind_t& idx) {
            if (idx < index2word.size()) {
                return index2word[idx];
            } else {
                return index2word[unknown_word];
            }
        });
        return result;
    }

    Vocab Vocab::from_many_nonunique(std::initializer_list<str_sequence> sequences,
                              bool add_unknown_word) {
        vector<string> words;
        for (auto& sequence: sequences) {
            for (auto& word: sequence) {
                words.push_back(word);
            }
        }
        // make them unique
        std::sort(words.begin(), words.end());
        words.erase(std::unique(words.begin(), words.end()), words.end());

        return Vocab(words, add_unknown_word);
    }

    Vocab::Vocab() : unknown_word(-1) {add_unknown_word();};

    Vocab::Vocab(vector<string>& _index2word) : index2word(_index2word), unknown_word(-1) {
        construct_word2index();
        add_unknown_word();
    }
    Vocab::Vocab(vector<string>& _index2word, bool _unknown_word) : index2word(_index2word), unknown_word(-1) {
        construct_word2index();
        if (_unknown_word) add_unknown_word();
    }

    Vocab::Vocab(vector<string>&& _index2word) : Vocab(_index2word) {
    }
    Vocab::Vocab(vector<string>&& _index2word, bool _unknown_word) : Vocab(_index2word, _unknown_word) {
    }

    CharacterVocab::CharacterVocab(int min_char, int max_char)
        : min_char(min_char), max_char(max_char) {
        assert2(max_char > min_char, MS() << "Maximum character (" << max_char << ") must be larger than minimum character (" << min_char << ").");
        assert2(max_char >= 0 && min_char >= 0, "Cannot have negative characters in mapping");
    }

    size_t CharacterVocab::size() const {
        return (size_t) ((max_char - min_char) + 1);
    }

    vector<typename Vocab::ind_t> CharacterVocab::encode(const str_sequence& words) const {
        vector<ind_t> result;
        int char_size = 0;
        // add all characters:
        for (auto& w : words) char_size += w.size();
        // for spaces:
        if (words.size() > 0) char_size += words.size() - 1;

        result.reserve(char_size);
        int unknown_char = max_char - min_char;
        int space_char = ' ' - min_char;

        if (space_char < 0) space_char = unknown_char;

        int word_idx = 0;
        for (auto& w : words) {
            for (auto& c : w) {
                if ((int)c >= min_char && (int)c < max_char) {
                    result.emplace_back(c - min_char);
                } else {
                    // all unknown get replaced by max_char
                    result.emplace_back(unknown_char);
                }
            }
            word_idx++;
            if (word_idx < words.size()) result.emplace_back(space_char);
        }
        return result;
    }

    vector<string> CharacterVocab::decode(Indexing::Index indices) const {
        vector<string> result;
        result.reserve(indices.size());

        stringstream stream;
        for (auto& index : indices) {
            if (index == max_char) {
                stream << "█";
            } else {
                stream << ((char) (index + min_char));
            }
        }
        return tokenize(stream.str());
    }

    vector<string> CharacterVocab::decode_characters(Indexing::Index indices) const {
        vector<string> result(indices.size());
        int char_idx = 0;
        for (auto& index : indices) {
            if (index == max_char) {
                result[char_idx++] = "█";
            } else {
                result[char_idx++] = (char) (index + min_char);
            }
        }
        return result;
    }

    template<typename T>
    T from_string(const std::string& s) {
        std::istringstream stream (s);
        T t;
        stream >> t;
        return t;
    }

    bool is_number(const std::string& s) {
        bool is_negative = *s.begin() == '-';
        bool is_decimal  = false;
        if (is_negative && s.size() == 1) {
            return false;
        }
        return !s.empty() && std::find_if(s.begin() + (is_negative ? 1 : 0),
            s.end(), [&is_decimal](char c) {
                if (is_decimal) {
                    if (c == '.') {
                        return true;
                    } else {
                        return !std::isdigit(c);
                    }
                } else {
                    if (c == '.') {
                        is_decimal = true;
                        return false;
                    } else {
                        return !std::isdigit(c);
                    }
                }
            }) == s.end();
    }

    template float from_string<float>(const std::string& s);
    template int from_string<int>(const std::string& s);
    template uint from_string<uint>(const std::string& s);
    template double from_string<double>(const std::string& s);
    template long from_string<long>(const std::string& s);

    bool is_gzip(const std::string& fname) {
        const unsigned char gzip_code = 0x1f;
        const unsigned char gzip_code2 = 0x8b;
        unsigned char ch;
        std::ifstream file;
        file.open(fname);
        if (!file) return false;
        file.read(reinterpret_cast<char*>(&ch), 1);
        if (ch != gzip_code)
                return false;
        if (!file) return false;
        file.read(reinterpret_cast<char*>(&ch), 1);
        if (ch != gzip_code2)
                return false;
        return true;
    }

    template <typename T>
    vector<size_t> argsort(const vector<T> &v) {
            // initialize original index locations
            vector<size_t> idx(v.size());
            for (size_t i = 0; i != idx.size(); ++i) idx[i] = i;

            // sort indexes based on comparing values in v
            sort(idx.begin(), idx.end(),
               [&v](size_t i1, size_t i2) {return v[i1] < v[i2];});

            return idx;
    }
    template vector<size_t> argsort(const vector<size_t>&v);
    template vector<size_t> argsort(const vector<float>&v);
    template vector<size_t> argsort(const vector<double>&v);
    template vector<size_t> argsort(const vector<int>&v);
    template vector<size_t> argsort(const vector<uint>&v);
    template vector<size_t> argsort(const vector<std::string>&v);

    template<typename T>
    LambdaOperator<T>::LambdaOperator(std::function<T(T)> lambda_expr) : lambda_expr(lambda_expr) {
    }

    template<typename T>
    T LambdaOperator<T>::operator () (T x) const { return lambda_expr(x); }

    template struct LambdaOperator<float>;
    template struct LambdaOperator<double>;

    template<typename T>
    T sigmoid_operator<T>::operator () (T x) const { return 1.0 / (1.0 + exp(-x)); }


    template<typename T>
    T log_or_zero<T>::operator () (T x) const { return x > 0 ? log(x) : 0; }

    template struct log_or_zero<float>;
    template struct log_or_zero<double>;

    template<typename T>
    T steep_sigmoid_operator<T>::operator () (T x) const {return 1.0 / (1.0 + exp( - aggressiveness * x));}

    template<typename T>
    steep_sigmoid_operator<T>::steep_sigmoid_operator(T _aggressiveness) : aggressiveness(_aggressiveness) {};

    template<typename T>
    T tanh_operator<T>::operator() (T x) const { return std::tanh(x); }

    template<typename T>
    T relu_operator<T>::operator() (T x) const { return std::max(x, (T) 0.0); }

    template<typename T>
    T max_operator<T>::operator() (T x) const { return x > 0.0 ? 1.0 : 0.0; }

    template<typename T>
    T sign_operator<T>::operator() (T x) const { return x > 0.0 ? 1.0 : -1.0; }

    template<typename T>
    T dtanh_operator<T>::operator() (T x) const { return 1.0 - x*x; }

    template <class T> inline void hash_combine(std::size_t & seed, const T & v) {
      std::hash<T> hasher;
      seed ^= hasher(v) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
    }

    template struct sigmoid_operator<float>;
    template struct steep_sigmoid_operator<float>;
    template struct tanh_operator<float>;
    template struct relu_operator<float>;
    template struct sign_operator<float>;
    template struct max_operator<float>;
    template struct dtanh_operator<float>;
    // template struct clip_operator<float>;

    template struct sigmoid_operator<double>;
    template struct steep_sigmoid_operator<double>;
    template struct tanh_operator<double>;
    template struct relu_operator<double>;
    template struct sign_operator<double>;
    template struct max_operator<double>;
    template struct dtanh_operator<double>;

    void exit_with_message(const std::string& message, int error_code) {
            std::cerr << message << std::endl;
            exit(error_code);
    }

    bool endswith(std::string const & full, std::string const & ending) {
        if (full.length() >= ending.length()) {
            return (0 == full.compare(full.length() - ending.length(), ending.length(), ending));
        } else {
            return false;
        }
    }

    bool startswith(std::string const & full, std::string const & beginning) {
        if (full.length() >= beginning.length()) {
            return (0 == full.compare(0, beginning.length(), beginning));
        } else {
            return false;
        }
    }

    bool file_exists (const std::string& fname) {
        struct stat buffer;
        return (stat (fname.c_str(), &buffer) == 0);
    }

    std::string dir_parent(const std::string& path, int levels_up) {
        auto file_path_split = split(__FILE__, '/');
        assert(levels_up < file_path_split.size());
        stringstream ss;
        if (path[0] == '/')
            ss << '/';
        for (int i = 0; i < file_path_split.size() - levels_up; ++i) {
            ss << file_path_split[i];
            if (i + 1 != file_path_split.size() - levels_up)
                ss << '/';
        }
        return ss.str();
    }

    std::string dir_join(const vector<std::string>& paths) {
        stringstream ss;
        for (int i = 0; i < paths.size(); ++i) {
            ss << paths[i];
            if (i + 1 != paths.size())
                ss << '/';
        }
        return ss.str();
    }

    template<typename T>
    std::vector<T> normalize_weights(const std::vector<T>& weights) {
        T minimum = weights[0];
        T sum = 0;
        for (int i=0; i<weights.size(); ++i) {
            minimum = std::min(minimum, weights[i]);
            sum += weights[i];
        }
        vector<T> res;
        T normalized_sum = sum - minimum * weights.size();
        for (int i=0; i<weights.size(); ++i) {
            res.push_back((weights[i] - minimum) / (normalized_sum));
        }
        return res;
    }

    template vector<float> normalize_weights(const std::vector<float>&);
    template vector<double> normalize_weights(const std::vector<double>&);

    string prefix_match(vector<string> candidates, string input) {
        assert2(!candidates.empty(), "Empty set of candidates for prefix matching.");
        int best_match_idx = -1;
        for (auto& candidate: candidates) {
            if (candidate.size() < input.size())
                continue;
            if (startswith(candidate, input))
                return candidate;
        }
        assert2(false, MS() << "Could not find match for " << input << " in " << candidates <<".");
        return "";
    }

    bool validate_flag_nonempty(const char* flagname, const std::string& value) {
        if (value.empty()) {
            std::cout << "Invalid value for --" << flagname << " (can't be empty)" << std::endl;
        }
        return not value.empty();
    }

    template<typename T>
    T vsum(const vector<T>& vec) {
        T res = 0;
        for(T item: vec) res += item;
        return res;
    }

    template float vsum(const vector<float>& vec);
    template double vsum(const vector<double>& vec);
    template int vsum(const vector<int>& vec);
    template uint vsum(const vector<uint>& vec);



    template<typename T>
    vector<T> reversed(const vector<T>& v) {
        vector<T> ret(v.rbegin(), v.rend());
        return ret;
    }

    std::string capitalize(const std::string& s) {
        std::string capitalized = s;
        // capitalize
        if (capitalized[0] >= 'a' && capitalized[0] <= 'z') {
            capitalized[0] += ('A' - 'a');
        }
        return capitalized;
    }

    template vector<float> reversed(const vector<float>& vec);
    template vector<double> reversed(const vector<double>& vec);
    template vector<int> reversed(const vector<int>& vec);
    template vector<uint> reversed(const vector<uint>& vec);
    template vector<size_t> reversed(const vector<size_t>& vec);
    template vector<string> reversed(const vector<string>& vec);
    template vector<vector<string>> reversed(const vector<vector<string>>& vec);


    std::unordered_map<std::string, std::atomic<int>> Timer::timers;
    std::mutex Timer::timers_mutex;


    ThreadError::ThreadError(int num_threads) :
            num_threads(num_threads),
            thread_error(num_threads),
            thread_error_updates(num_threads) {
        reset();
    }

    void ThreadError::update(double error) {
        thread_error[ThreadPool::get_thread_number()] += error;
        thread_error_updates[ThreadPool::get_thread_number()] += 1;
    }

    double ThreadError::this_thread_average() {
        int updates = thread_error_updates[ThreadPool::get_thread_number()];
        double error = thread_error[ThreadPool::get_thread_number()];
        return (updates == 0) ? 1.0/0.0 : error/updates;
    }

    double ThreadError::average() {
        return vsum(thread_error) / vsum(thread_error_updates);
    }

    void ThreadError::reset() {
        for (int tidx = 0; tidx < num_threads; ++tidx) {
            thread_error[tidx] = 0;
            thread_error_updates[tidx] = 0;
        }
    }

    Timer::Timer(std::string name, bool autostart) : name(name),
                                                     stopped(false),
                                                     started(false) {
        if (timers.find(name) == timers.end()) {
            std::lock_guard<decltype(timers_mutex)> guard(timers_mutex);
            if (timers.find(name) == timers.end())
                timers[name] = 0;
        }
        if (autostart)
            start();
    }

    void Timer::start() {
        assert(!started);
        start_time = clock_t::now();
        started = true;
    }

    void Timer::stop() {
        assert(!stopped);
        timers[name] += std::chrono::duration_cast< std::chrono::milliseconds >
                        (clock_t::now() - start_time).count();
        stopped = true;
    }

    Timer::~Timer() {
        if(!stopped)
            stop();
    }

    void Timer::report() {
        std::lock_guard<decltype(timers_mutex)> guard(timers_mutex);

        for (auto& kv : timers) {
            std::cout << "\"" << kv.first << "\" => "
                      << std::fixed << std::setw(5) << std::setprecision(4) << std::setfill(' ')
                      << (double) kv.second / 1000  << "s" << std::endl;
        }

        timers.clear();
    }

    void assert2(bool condition) {
        assert2(condition, "");
    }

    void assert2(bool condition, std::string message) {
        if (!condition) {
            throw std::runtime_error(message);
        }
    }
    // color codes: http://www.codebuilder.me/2014/01/color-terminal-text-in-c/
    std::string green       = "\033[32m";
    std::string red         = "\033[31m";
    std::string blue        = "\033[34m";
    std::string yellow      = "\033[33m";
    std::string cyan        = "\033[36m";
    std::string black       = "\033[30m";
    std::string reset_color = "\033[0m";
    std::string bold        = "\033[1m";
}


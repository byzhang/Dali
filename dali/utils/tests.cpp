/*#include <chrono>
#include <vector>
#include <memory>
#include <gtest/gtest.h>
#include <sstream>
#include <cstdio>
#include <string>
#include "dali/utils.h"

using std::chrono::milliseconds;
using std::make_shared;
using std::string;
using std::stringstream;
using std::vector;
using utils::OntologyBranch;
using utils::CharacterVocab;

TEST(ThreadPool, wait_until_idle) {
    const int NUM_THREADS = 10;
    const int NUM_JOBS = 100;
    ThreadPool t(NUM_THREADS);

    bool barrier = true;

    for(int j=0; j<NUM_JOBS; ++j) {
        t.run([&barrier]() {
            while(barrier) {
                std::this_thread::yield();
            };
        });
    }

    // wait for threads to pick up work.
    while(t.active_workers() < NUM_THREADS);

    // Threads are currently waiting for barrier.
    // Ensure that wait until idle returns false..
    ASSERT_FALSE(t.wait_until_idle(milliseconds(1)));
    // Remove barrier and hope they come back.
    barrier = false;

    // Assert all threads will be done exentually.
    ASSERT_TRUE(t.wait_until_idle());
}

TEST(ThreadPool, thread_number) {
    const int NUM_THREADS = 4;
    const int JOBS_PER_ATTEMPT = 10;
    ThreadPool pool(NUM_THREADS);


    for(int t = 0; t < NUM_THREADS; ++t) {
        // Try to get thread t to manifest itself by checking
        // it's thread_number.
        bool thread_t_seen = false;
        while (!thread_t_seen) {
            for(int job = 0; job < JOBS_PER_ATTEMPT; ++job) {
                pool.run([&t, &thread_t_seen]() {
                    for (int i=0; i<10000; ++i) {
                        if(t == ThreadPool::get_thread_number()) {
                            thread_t_seen = true;
                        }
                    }
                });
            }
            pool.wait_until_idle();
        }
    }
}

TEST(utils, stream_to_redirection_list) {
    stringstream ss(
        "hello->world\n"
        "what->happened\n"
        "who->is this?\n");

    std::map<string, string> mapping;
    utils::stream_to_redirection_list(ss, mapping);
    ASSERT_TRUE(mapping.find("hello") != mapping.end());
    ASSERT_TRUE(mapping.find("what") != mapping.end());
    ASSERT_TRUE(mapping.find("who") != mapping.end());
    ASSERT_EQ(mapping.at("hello"), "world");
    ASSERT_EQ(mapping.at("what"), "happened");
    ASSERT_EQ(mapping.at("who"), "is this?");
}

TEST(utils, stream_to_list) {
    stringstream ss(
        "hello\n"
        "what\n"
        "who\n");

    std::vector<string> list;
    utils::stream_to_list(ss, list);
    ASSERT_EQ(list.size(), 3);
    ASSERT_EQ(list[0], "hello");
    ASSERT_EQ(list[1], "what");
    ASSERT_EQ(list[2], "who");
}

TEST(utils, split_str) {
    string input = "hellobobmeisterhowareyou?";

    auto tokens = utils::split_str(input, "bobmeister");

    ASSERT_EQ(tokens.size(), 2);
    ASSERT_EQ(tokens[0], "hello");
    ASSERT_EQ(tokens[1], "howareyou?");
}

TEST(utils, trim) {
    string input = "     hello_world thus ?     ";
    auto trimmed = utils::trim(input);
    ASSERT_EQ(trimmed, "hello_world thus ?");
}

TEST(xml_cleaner, preprocess) {
    // TODO add fail conditions for this test
    string input = "{{Other uses}}\n"
    "{{Redirect|Foo|FOO (Forward Observation Officer)|Artillery observer}}\n"
    "\n"
    "The terms '''foobar''' ({{IPAc-en|ˈ|f|uː|b|ɑr}}), '''fubar''', or "
    "'''foo''', '''bar''', '''baz''' and '''qux''' (alternatively, '''"
    "quux''') are sometimes used as [[placeholder name]]s (also referred"
    " to as [[metasyntactic variable]]s) in [[computer programming]] or "
    "computer-related documentation.<ref name=\"rfc3092\" /> They have been "
    "used to name entities such as [[Variable (computer science)|variable]]s,"
    " [[Function (computer science)|functions]], and [[command (computing)|command]]s"
    " whose purpose is unimportant and serve only to demonstrate a concept."
    "  The words themselves have no meaning in this usage.  ''Foobar'' is "
    "sometimes used alone; ''foo'', ''bar'', and ''baz'' are sometimes used"
    " in that order, when multiple entities are needed.\n"
    "\n"
    "The usage in [[computer programming]] examples and [[pseudocode]] "
    "varies; in certain circles, it is used extensively, but many prefer "
    "descriptive names, while others prefer to use single letters.  "
    "[[Eric S. Raymond]] has called it an \"important hackerism\" "
    "alongside [[kludge]] and [[cruft]].<ref name=\"Raymond\">{{cite "
    "book|url=http://books.google.com/books?id=g80P_4v4QbIC&amp;lpg=PP1"
    "&amp;pg=PA5|title=The New Hacker's Dictionary|author=[[Eric S. "
    "Raymond]]|publisher=[[MIT Press]]|year=1996|isbn=0-262-68092-0}}</ref>Okay\n";

    auto cleaned = utils::xml_cleaner::process_text_keeping_brackets(input);
}

TEST(utils, construct_lattice) {
    auto root = make_shared<OntologyBranch>("root");
    for (auto& v : {"Joe", "Bob", "Max", "Mary", "Jane", "Goodwin"})
        make_shared<OntologyBranch>(v)->add_parent(root);
    auto root2 = make_shared<OntologyBranch>("root 2");

    for (auto& child : root->children) {
        child->add_parent(root2);
    }

    stringstream ss;
    // Output the name of both parents (verify they are indeed 2)
    for (auto& par : root->children[2]->parents)
        ss << par.lock()->name << " ";

    ASSERT_EQ(ss.str(), "root root 2 ");
    ASSERT_EQ(root->children[2]->parents.size(), 2);
}

TEST(utils, load_save_lattice) {
    auto root = make_shared<OntologyBranch>("root");
    for (auto& v : {"Joe", "Bob", "Max", "Mary", "Jane", "Goodwin"})
        make_shared<OntologyBranch>(v)->add_parent(root);
    auto root2 = make_shared<OntologyBranch>("root 2");

    for (auto& child : root->children)
        child->add_parent(root2);

    string data_folder = STR(DALI_DATA_DIR) "/";
    string fname       = data_folder + "/lattice2.txt";
    string fname_gz    = fname + ".gz";

    if (utils::file_exists(fname))
        std::remove(fname.c_str());
    // Test saving a lattice file
    ASSERT_FALSE(utils::file_exists(fname));
    root->save(fname);
    ASSERT_TRUE(utils::file_exists(fname));
    ASSERT_FALSE(utils::is_gzip(fname));

    if (utils::file_exists(fname_gz))
        std::remove(fname_gz.c_str());
    // Test saving a gzipped file:
    ASSERT_FALSE(utils::file_exists(fname_gz));
    root->save(fname_gz);
    ASSERT_TRUE(utils::file_exists(fname_gz));
    ASSERT_TRUE(utils::is_gzip(fname_gz));
}

TEST(utils, load_tsv) {
    string tsv_file = STR(DALI_DATA_DIR) "/CoNLL_NER/NER_dummy_dataset.tsv";
    ASSERT_THROW(
        utils::load_tsv(tsv_file, 3, '\t'), std::runtime_error
    );
    auto dataset = utils::load_tsv(tsv_file, 4, '\t');
    ASSERT_EQ(dataset.size(), 7);
    ASSERT_EQ(dataset.back().front().front(), ".");
}

TEST(utils, load_lattice) {
    string data_folder = STR(DALI_DATA_DIR) "/";
    auto loaded_tree = OntologyBranch::load(data_folder + "lattice.txt");

    // find 2 roots
    ASSERT_EQ(loaded_tree.size(), 2);

    stringstream ss;
    for (auto& r : loaded_tree)
        ss << r->name << " ";
    ASSERT_EQ(ss.str(), "root 3 root 4 ");
    // find root 2
    ASSERT_TRUE(loaded_tree[0]->lookup_table->find("root 2") != loaded_tree[0]->lookup_table->end());

    // compare with the original tree
    auto root = make_shared<OntologyBranch>("root");
    for (auto& v : {"Joe", "Bob", "Max", "Mary", "Jane", "Goodwin"})
        make_shared<OntologyBranch>(v)->add_parent(root);

    // iterate through children of root 2
    auto root2_loaded = loaded_tree[0]->lookup_table->at("root 2");

    int found = 0;
    for (auto& child : root->children) {
        for (auto& subchild : root2_loaded->children) {
            if (subchild->name == child->name) {
                found += 1;
                break;
            }
        }
    }
    ASSERT_EQ(found, root->children.size());
}

TEST(utils, smart_parser) {
    std::shared_ptr<std::stringstream> ss = std::make_shared<std::stringstream>();
    *ss << "siema 12 123\n"
        << "555\n"
        << "lol lone 123\n"
        << " \n"
        << "155\n";
    SmartParser sp(ss);
    assert(sp.next_string() == "siema");
    assert(sp.next_int() == 12);
    assert(sp.next_int() == 123);
    assert(sp.next_int() == 555);
    assert(sp.next_line() == "lol lone 123");
    assert(sp.next_int() == 155);
}

TEST(utils, pearson_correlation) {
    vector<double> x = {43, 21, 25, 42, 57, 59};
    vector<double> y = {99, 65, 79, 75, 87, 81};

    auto corr = utils::pearson_correlation(x,y);

    ASSERT_NEAR(corr, 0.5298, 1e-5);
}

TEST(utils, CharacterVocab) {
    auto seq      = vector<string>{"bob", "ate", "an", "apple"};
    auto vocab    = CharacterVocab(0, 255);
    auto chars    = vocab.encode(seq);
    auto seq_size = utils::join(seq, " ").size();
    ASSERT_EQ(chars.size(), seq_size);
    ASSERT_EQ(seq, vocab.decode(chars));


    auto char_decoded_seq = vocab.decode_characters(chars);
    ASSERT_EQ(char_decoded_seq.size(), seq_size);
    ASSERT_EQ(utils::join(seq, " "), utils::join(char_decoded_seq));


    // space char is char 32, if we start at 33 we lose it, and
    // spaces get replaced by "█":
    auto spaceless_vocab = CharacterVocab(33, 255);
    auto spaceless_chars = spaceless_vocab.encode(seq);
    ASSERT_NE(seq, spaceless_vocab.decode(spaceless_chars));
    auto special_seq = utils::join(seq, "\xFF");
    ASSERT_EQ(special_seq, utils::join(spaceless_vocab.decode(spaceless_chars)));
}

TEST(utils, prefix_match) {
    using utils::prefix_match;
    vector<string> candidates = {
        "siema",
        "lol",
        "we_hit_a_wall",
    };
    // candidates match with themselvers
    for (auto& candidate : candidates) {
        ASSERT_EQ(candidate, prefix_match(candidates, candidate));
    }

    EXPECT_THROW(prefix_match(candidates, "low"), std::runtime_error);
    EXPECT_THROW(prefix_match(candidates, "lol2"), std::runtime_error);

    EXPECT_EQ(prefix_match(candidates, ""), "siema");
    EXPECT_EQ(prefix_match(candidates, "lo"), "lol");
    EXPECT_EQ(prefix_match(candidates, "we_hit"), "we_hit_a_wall");
}

struct Range : utils::GeneratorHeart<int> {
    void run(int start, int end, int interval=1) {
        for (int i=start; i<end; i+=interval) {
            yield(i);
        }
    }
};

TEST(utils, generator_test) {
    auto vals = vector<int>();
    for (int i : utils::Gen<Range>(2,9,2)) vals.emplace_back(i);
    ASSERT_EQ(vals, vector<int>({2, 4, 6, 8}));
}

TEST(utils, lambda_generator_test) {
    auto gen = utils::Generator<int>([](utils::yield_t<int> yield) {
        for (int i=2; i<9; i+=2) yield(i);
    });
    auto vals = vector<int>();
    for (int i : gen)
        vals.emplace_back(i);
    ASSERT_EQ(vals, vector<int>({2, 4, 6, 8}));
}


TEST(utils, test_initialize_gen) {
    // This test illustrates that generator_constructor can be sometimes
    // dangerous if we do not think about initialization

    // TEST GOAL: generate {1,2,3,4,5,  1,2,3,4,5} using shared_resource.

    int shared_resource = 1;

    auto advance_noinitialization = [&shared_resource](utils::yield_t<int> yield) {
        int repeats = 5;
        while(repeats--) {
            yield(shared_resource++);
        }
    };

    auto advance_correct = [&shared_resource](utils::yield_t<int> yield) {
        shared_resource = 1;
        int repeats = 5;
        while(repeats--) {
            yield(shared_resource++);
        }
    };

    auto noinitialization = utils::Generator<int>(advance_noinitialization);
    auto correct = utils::Generator<int>(advance_correct);

    auto vals = vector<int>();
    for (int i : noinitialization)
        vals.emplace_back(i);
    noinitialization.reset();
    for (int i : noinitialization)
        vals.emplace_back(i);
    ASSERT_EQ(vector<int>({1,2,3,4,5,6,7,8,9,10}), vals);

    vals.clear();
    for (int i : correct)
        vals.emplace_back(i);
    correct.reset();
    for (int i : correct)
        vals.emplace_back(i);
    ASSERT_EQ(vector<int>({1,2,3,4,5,1,2,3,4,5}), vals);
}


TEST(utils, recursive_generator_test) {
    // here we are using Generator rather than make generator,
    // so that we can use it multiple times. For example each time we call
    // gen_12345() new generator is constructed.

    // TEST GOAL: generate {1,2,3,4,5} five times.
    auto gen_12345 = utils::Generator<int>([](utils::yield_t<int> yield) {
        for (int i=1; i<=5; i+=1) yield(i);
    });
    auto gen_5x_12345 = utils::Generator<int>([&gen_12345](utils::yield_t<int> yield) {
        int repeats = 5;
        while(repeats--) {
            gen_12345.reset();
            for (auto num: gen_12345)
                yield(num);
        }
    });

    auto vals = vector<int>();
    for (int i : gen_5x_12345)
        vals.emplace_back(i);
    ASSERT_EQ(vector<int>({1,2,3,4,5, 1,2,3,4,5, 1,2,3,4,5, 1,2,3,4,5, 1,2,3,4,5}), vals);
}

TEST(utils, combine_generators) {
    // here we take two short generators and
    // create a longer one out of the pair:
    auto comb_gen = (
        utils::Generator<int>([](utils::yield_t<int> yield) {
            for (int i=1; i<=5; i+=1) yield(i);
        })
        +
        utils::Generator<int>([](utils::yield_t<int> yield) {
            for (int i=6; i<=10; i+=1) yield(i);
        })
    );

    auto vals = vector<int>();
    for (int i : comb_gen)
        vals.emplace_back(i);

    ASSERT_EQ(vals, vector<int>({1,2,3,4,5,6,7,8,9,10}));

}
*/

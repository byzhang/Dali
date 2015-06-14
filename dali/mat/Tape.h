#ifndef CORE_NEW_GRAPH_H
#define CORE_NEW_GRAPH_H

#include <functional>
#include <vector>

namespace graph {
    void emplace_back(std::function<void()>&& f);

    void backward();

    bool backprop_enabled();

    class Tape {
        public:
            std::vector<std::function<void()>>  backprop;

            void backward ();
    };

    class NoBackprop {
        private:
            // value of backprop before object go activated.
            const bool old_value;
            // whether the object actually does something (used for condition).
            const bool enabled;
            NoBackprop(const NoBackprop&) = delete;
            NoBackprop& operator =(NoBackprop const &) = delete;

        public:
            explicit NoBackprop();
            // Disable backprop only if condition is true
            explicit NoBackprop(bool condition);
            ~NoBackprop();
    };
}

#endif

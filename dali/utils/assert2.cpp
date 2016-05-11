#include "dali/utils/assert2.h"

namespace utils {
    void assert2(bool condition, const std::string& message) {
        if (!condition) {
            throw std::runtime_error(message);
        }
    }

    void assert2(bool condition) {
        assert2(condition, "");
    }
}

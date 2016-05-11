#ifndef DALI_UTILS_ASSERT2_H
#define DALI_UTILS_ASSERT2_H

#include <string>
#include <stdexcept>

// throw assertion errors with messages
namespace utils {
    void assert2(bool condition);
    void assert2(bool condition, const std::string& message);
}
#define ASSERT2(condition, message) if (!(condition)) utils::assert2(false, (message))

#endif

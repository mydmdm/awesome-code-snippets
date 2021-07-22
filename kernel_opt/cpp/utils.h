#ifndef __UTILS_H__
#define __UTILS_H__

#include <chrono>
#include <iostream>
#include <stdio.h>

/**
 * essential 
 */
#define COMMA ,

#define VA_ARGS(...) , ##__VA_ARGS__

#define do_fprintf(PREFIX, OUT, FMT, ...)                                         \
    do                                                                            \
    {                                                                             \
        fprintf(OUT, "%s:%d <%s()> [%s] ", __FILE__, __LINE__, __func__, PREFIX); \
        fprintf(OUT, FMT, __VA_ARGS__);                                           \
    } while (0)

#define say_goodbye(PREFIX)                                  \
    do                                                       \
    {                                                        \
        do_fprintf(PREFIX, stderr, "%s!!\n", "Fatal Error"); \
        exit(-1);                                            \
    } while (0)

/**
 * syntax sugar
 */
#define range(VAR, STOP) for (auto VAR = 0u; VAR != STOP; ++VAR)
#define foreach(VAR, CONTAINER) for (auto VAR : CONTAINER)

// integer ceil
#define iceil(A, B) ((A + B - 1) / (B))

/**
 * Assertion and Expection
 */

#define assert_eq(A, B, PREFIX)                                                                                     \
    do                                                                                                              \
    {                                                                                                               \
        if ((A) != (B))                                                                                             \
        {                                                                                                           \
            std::cerr << "AssertEqual fails[" << PREFIX << "]: " << (A) << " is not equal to " << (B) << std::endl; \
            say_goodbye("AssertExcpetion");                                                                         \
        }                                                                                                           \
    } while (0)

// A must be greater than or equal to B
#define assert_ge(A, B, PREFIX)                                                                                         \
    do                                                                                                                  \
    {                                                                                                                   \
        if ((A) < (B))                                                                                                  \
        {                                                                                                               \
            std::cerr << "AssertGreaterEqual fails[" << PREFIX << "]: " << (A) << " is less than " << (B) << std::endl; \
            say_goodbye("AssertExcpetion");                                                                             \
        }                                                                                                               \
    } while (0)

// A must be true
#define assert_true(A, PREFIX)                                                                     \
    do                                                                                             \
    {                                                                                              \
        if (!(A))                                                                                  \
        {                                                                                          \
            std::cerr << "Assert fails[" << PREFIX << "]: " << (A) << " is not true" << std::endl; \
            say_goodbye("AssertExcpetion");                                                        \
        }                                                                                          \
    } while (0)

/* dealing with timer
*/
inline std::chrono::high_resolution_clock::time_point get_now()
{
    return std::chrono::high_resolution_clock::now();
}

/* calculate the time eclipse from a previous time point (start)
 */
inline time_t time_difference_ns(std::chrono::high_resolution_clock::time_point start)
{
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(get_now() - start);
    return duration.count();
}

#endif /* __UTILS_H__ */

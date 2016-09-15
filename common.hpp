#pragma once

#include <fstream>
#include <cstdint>
#include <sys/time.h>
#include <mutex>
#include <string>
#include <omp.h>
#include <algorithm>

#ifdef USE_TBB
#include <tbb/parallel_sort.h>
#endif

#define likely(x)       __builtin_expect((x),1)
#define unlikely(x)     __builtin_expect((x),0)

#define INLINE inline __attribute__((always_inline))

namespace hygraph {

typedef uint32_t vid_t;
typedef uint32_t eid_t;

enum EdgeDir {
    EDGE_NONE = 0,
    EDGE_OUT = 1,
    EDGE_IN = 2,
    EDGE_BOTH = 3
};

enum ActivityType {
    ACTIVITY_ALWAYS,
    ACTIVITY_SELECTED
};


typedef uint32_t bitvec_t;

#define BITVEC_GET(v, i) \
    (((v))[(i) / 32] & (((uint32_t)1) << ((i) % 32)))

#define BITVEC_SET(v, i) \
    (((v))[(i) / 32] |= (((uint32_t)1) << ((i) % 32)))

#define BITVEC_UNSET(v, i) \
    (((v))[(i) / 32] &= ~(((uint32_t)1) << ((i) % 32)))

#define BITVEC_TOGGLE(v, i, f) \
    ((f) ? (BITVEC_SET(v, i)) : (BITVEC_UNSET(v, i)))

#define BITVEC_SIZE(s) \
    (((s) / 32) + ((s) % 32 != 0 ? 1 : 0))

struct empty_t {
    //
};

template <typename T>
struct is_empty_type {
    static const bool value = false;
};

template <>
struct is_empty_type<empty_t> {
    static const bool value = true;
};

template <typename T>
__host__ __device__ size_t ceil_div(T a, T b) {
    return (a / b) + (a % b != 0);
}

static double timer() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return (tv.tv_sec + tv.tv_usec / 1000000.0);
}


static std::mutex log_mutex;
static double log_start_time = 0;

template <typename ...A>
static void log(const char *fmt, A... args) {
    log_mutex.lock();

    if (log_start_time == 0) {
        log_start_time = timer();
    }

    printf("[% 10.5f] ", timer() - log_start_time);
    printf(fmt, args...);
    printf("\n");
    fflush(stderr);
    fflush(stdout);


    log_mutex.unlock();
}

static void log(const char *str) {
    log("%s", str);
}

size_t get_file_size(const std::string &filename) {
    std::ifstream f(filename.c_str(), std::ifstream::ate | std::ifstream::binary);
    return f.tellg();
}


template <typename RandomIt, typename Compare>
INLINE void par_sort(RandomIt first, RandomIt last, Compare comp) {
#ifdef USE_TBB
    tbb::parallel_sort(first, last, comp);
#else
    sort(first, last, comp);
#endif
}


};

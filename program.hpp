#pragma once

#include <cfloat>

#include "common.hpp"
#include "cuda_common.cuh"

namespace hygraph {

template <typename P, typename V, typename E, typename S, typename M>
class Program {
    public:
    //static const ActivityType activity = ACTIVITY_SELECTED;

    typedef V vertex_value_t;
    typedef E edge_value_t;
    typedef M message_t;
    typedef S vertex_state_t;

    INLINE CUDA_HOST_DEVICE bool init_state(const V &val, S &state) {
        return false;
    }

    INLINE CUDA_HOST_DEVICE void init_message(M &m) const {
        m = M();
    }

    INLINE CUDA_HOST_DEVICE bool generate_message(int superstep, const V &val, const S &state, M &msg) const {
        return false;
    }

    INLINE CUDA_HOST_DEVICE bool process_edge(int superstep, const E &edge, M &msg) const {
        return true;
    }

    INLINE CUDA_HOST_DEVICE bool process_vertex(int superstep, const V &val, const M &msg, S &state) const {
        return false;
    }

    INLINE CUDA_HOST_DEVICE void aggregate(const M &msg, M &result) const {
        if (!is_empty_type<M>::value) {
            printf("missing implementation of aggregate");
        }
    }

    INLINE CUDA_DEVICE void cuda_atomic_aggregate(const M msg, M *result) const {
        if (!is_empty_type<M>::value) {
            M old_val = *result;
            M new_val;

            do {
                new_val = old_val;
                ((P*) this)->aggregate(msg, new_val);
            } while(!cuda_atomic_cas(result, &old_val, new_val));
        }
    }
};



class BFSProgram: public Program<BFSProgram, vid_t, empty_t, uint16_t, empty_t> {
    public:
    static const ActivityType activity = ACTIVITY_SELECTED;

    vid_t root;

    BFSProgram(vid_t r): root(r) {
        //
    }

    INLINE CUDA_HOST_DEVICE bool init_state(const vertex_value_t &id, vertex_state_t &state) const {
        state = id == root ? 0 : (vertex_state_t)~0;
        return id == root;
    }

    INLINE CUDA_HOST_DEVICE bool generate_message(int superstep, const vertex_value_t &id, const vertex_state_t &state, message_t &msg) const {
        return true;
    }

    INLINE CUDA_HOST_DEVICE bool process_vertex(int superstep, const vertex_value_t &val, const message_t &msg, vertex_state_t &state) const {
        if (state == (vertex_state_t)~0) {
            state = superstep + 1;
            return true;
        } else {
            return false;
        }
    }
};


class PRProgram: public Program<PRProgram, uint32_t, empty_t, float, float> {
    public:
    static const ActivityType activity = ACTIVITY_ALWAYS;

    INLINE CUDA_HOST_DEVICE bool init_state(const vertex_value_t &val, vertex_state_t &state) const {
        state = 1.0;
        return true;
    }

    INLINE CUDA_HOST_DEVICE bool generate_message(int superstep, const vertex_value_t &val, const vertex_state_t &state, message_t &msg) const {
        msg = state / val;
        return true;
    }

    INLINE CUDA_HOST_DEVICE void init_message(message_t &m) const {
        m = 0.0;
    }

    INLINE CUDA_HOST_DEVICE bool process_vertex(int superstep, const vertex_value_t &val, const message_t &msg, vertex_state_t &state) const {
        float damping_factor = 0.85;
        state = damping_factor * msg + (1 - damping_factor);
        return true;
    }

    INLINE CUDA_HOST_DEVICE void aggregate(const message_t &msg, message_t &result) const {
        result += msg;
    }

    INLINE CUDA_DEVICE void cuda_atomic_aggregate(const message_t msg, message_t *result) const {
        atomicAdd(result, msg);
    }
};


class ConnProgram: public Program<ConnProgram, vid_t, empty_t, vid_t, vid_t> {
    public:
    static const ActivityType activity = ACTIVITY_SELECTED;

    INLINE CUDA_HOST_DEVICE bool init_state(const vertex_value_t &id, vertex_state_t &state) const {
        state = id;
        return true;
    }

    INLINE CUDA_HOST_DEVICE bool generate_message(int superstep, const vertex_value_t &val, const vertex_state_t &state, message_t &msg) const {
        msg = state;
        return true;
    }

    INLINE CUDA_HOST_DEVICE void init_message(message_t &m) const {
        m = ~(vid_t)0;
    }

    INLINE CUDA_HOST_DEVICE bool process_vertex(int superstep, const vertex_value_t &val, const message_t &msg, vertex_state_t &state) const {
        if (msg < state) {
            state = msg;
            return true;
        }

        return false;
    }

    INLINE CUDA_HOST_DEVICE void aggregate(const message_t &msg, message_t &result) const {
        if (msg < result) {
            result = msg;
        }
    }

    INLINE CUDA_DEVICE void cuda_atomic_aggregate(const message_t msg, message_t *result) const {
        atomicMin(result, msg);
    }
};


class SSSPProgram: public Program<SSSPProgram, vid_t, float, float, float> {
    public:
    static const ActivityType activity = ACTIVITY_SELECTED;

    vid_t root;

    SSSPProgram(vid_t r): root(r) {
        //
    }

    INLINE CUDA_HOST_DEVICE bool init_state(const vertex_value_t &id, vertex_state_t &state) const {
        state = id == root ? 0.0 : FLT_MAX;
        return (id == root);
    }

    INLINE CUDA_HOST_DEVICE bool generate_message(int superstep, const vertex_value_t &val, const vertex_state_t &state, message_t &msg) const {
        msg = state;
        return true;
    }

    INLINE CUDA_HOST_DEVICE void init_message(message_t &m) const {
        m = FLT_MAX;
    }

    INLINE CUDA_HOST_DEVICE bool process_edge(int superstep, const edge_value_t &edge, message_t &msg) const {
        msg += edge;
        return true;
    }

    INLINE CUDA_HOST_DEVICE bool process_vertex(int superstep, const vertex_value_t &val, const message_t &msg, vertex_state_t &state) const {
        if (msg < state) {
            state = msg;
            return true;
        }

        return false;
    }

    INLINE CUDA_HOST_DEVICE void aggregate(const message_t &msg, message_t &result) const {
        if (msg < result) {
            result = msg;
        }
    }

    INLINE CUDA_DEVICE void cuda_atomic_aggregate(const message_t msg, message_t *result) const {
        float curr = *result;

        while (msg < curr) {
            if (cuda_atomic_cas(result, &curr, msg)) {
                break;
            }
        }
    }
};

}

#pragma once

#include <string>
#include <vector>
#include <utility>
#include <algorithm>
#include <memory>

#include <cstddef>
#include "common.hpp"

#include <iostream>

#define CUDA_KERNEL __global__
#define CUDA_HOST_DEVICE __host__ __device__
#define CUDA_DEVICE __device__
#define CUDA_WARP_SIZE (32)

namespace hygraph {
using namespace std;

class cuda_exception: public exception {
    string msg;
    cudaError_t err;

    public:
    cuda_exception(const char *msg) :
            msg(msg),
            err(cudaSuccess) {
        //
    }

    cuda_exception(cudaError_t code) :
            msg(cudaGetErrorString(code)),
            err(code) {
        //
    }

    cuda_exception(const char *msg, cudaError_t code) :
            msg(string(msg) + ": " + string(cudaGetErrorString(code))),
            err(code) {
        //
    }

    const char* what() const noexcept {
        return msg.c_str();
    }

    cudaError_t code() {
        return err;
    }
};

#define CUDA_CHECK(fun, ...) (cuda_check(#fun, fun(__VA_ARGS__)))

void cuda_check(const char *msg, cudaError_t err) {
    if (err != cudaSuccess) {
        throw cuda_exception(msg, err);
    }
}

void cuda_check_last(const char *msg) {
    cuda_check(msg, cudaGetLastError());
}

template <typename T, bool OnHost>
class cuda_mem {
    private:
    T *ptr;
    size_t capacity;

    public:
    cuda_mem() {
        ptr = NULL;
        capacity = 0;
    }

    cuda_mem(size_t size) {
        ptr = NULL;
        capacity = 0;

        allocate(size);
    }

    cuda_mem(const cuda_mem &rhs) {
        ptr = NULL;
        capacity = 0;

        *this = rhs;
    }

    cuda_mem(cuda_mem &&rhs) {
        ptr = NULL;
        capacity = 0;

        *this = move(rhs);
    }

    ~cuda_mem() {
        free();
    }

    void allocate(size_t new_capacity) {
        if (new_capacity == capacity) {
            return; // noop
        }

        free();

        if (new_capacity > 0) {
            if (OnHost) {
                CUDA_CHECK(cudaMallocHost, &ptr, new_capacity * sizeof(T));
            } else {
                CUDA_CHECK(cudaMalloc, &ptr, new_capacity * sizeof(T));
            }
        } else {
            ptr = NULL;
        }

        capacity = new_capacity;
    }

    void allocate(const T *data, size_t count) {
        allocate(count);
        from_host(data);
    }

    void allocate(const vector<T> &vec) {
        allocate(vec.data(), vec.size());
    }

    void allocate(const cuda_mem &m) {
        allocate(m.size());
        copy_from(m.get());
    }

    void free() {
        if (ptr) {
            if (OnHost) {
                CUDA_CHECK(cudaFreeHost, ptr);
            } else {
                CUDA_CHECK(cudaFree, ptr);
            }
        }

        ptr = NULL;
        capacity = 0;
    }

    void reallocate(size_t new_capacity) {
        if (new_capacity != capacity) {
            cuda_mem clone(new_capacity);
            clone.copy_from(*this, 0, 0, min(capacity, new_capacity));
            swap(clone);
        }
    }

    void fill(const T &val) {
        if (ptr) {
            if (OnHost) {
                std::fill(ptr, ptr + capacity, val);
            } else {
                char *val_ptr = (char*) &val;
                bool all_match = true;

                for (size_t index = 0; index < sizeof(T); index++) {
                    all_match |= val_ptr[index] == val_ptr[0];
                }

                if (all_match) {
                    CUDA_CHECK(cudaMemset, ptr, val_ptr[0], sizeof(T) * capacity);
                } else {
                    throw cuda_exception("Fill unsupported for general values");
                }
            }
        }
    }

    void clear() {
        fill(T());
    }

    private:
    template <bool IsAsync, bool IsSource, bool IsPeerHost>
    void transfer(cudaStream_t stream, void *data, size_t offset, size_t count) const {
        if (count == 0) {
            return;
        }

        if (offset >= capacity || offset + count > capacity) {
            throw cuda_exception("copy out of bounds");
        }

        cudaMemcpyKind kind;
        const void *src;
        void *dst;

        if (IsPeerHost && OnHost) {
            kind = cudaMemcpyHostToHost;
        } else if (!IsPeerHost && !OnHost) {
            kind = cudaMemcpyDeviceToDevice;
        } else if (IsSource ? OnHost : IsPeerHost) {
            kind = cudaMemcpyHostToDevice;
        } else {
            kind = cudaMemcpyDeviceToHost;
        }

        if (IsSource) {
            src = const_cast<T*>(ptr) + offset;
            dst = data;
        } else {
            src = data;
            dst = ptr + offset;
        }

        if (IsAsync) {
            CUDA_CHECK(cudaMemcpyAsync, dst, src, count * sizeof(T), kind, stream);
        } else {
            CUDA_CHECK(cudaMemcpy, dst, src, count * sizeof(T), kind);
        }
    }

    public:
    void to_host(T *data, size_t offset, size_t count) const {
        transfer<false, true, true>(0, (void*) data, offset, count);
    }

    void from_host(const T *data, size_t offset, size_t count) {
        transfer<false, false, true>(0, const_cast<T*>(data), offset, count);
    }

    void to_host_async(cudaStream_t stream, T *data, size_t offset, size_t count) const {
        transfer<true, true, true>(stream, (void*) data, offset, count);
    }

    void from_host_async(cudaStream_t stream, const T *data, size_t offset, size_t count) {
        transfer<true, false, true>(stream, const_cast<T*>(data), offset, count);
    }

    void to_device(T *data, size_t offset, size_t count) const {
        transfer<false, true, false>(0, (void*) data, offset, count);
    }

    void from_device(const T *data, size_t offset, size_t count) {
        transfer<false, false, false>(0, const_cast<T*>(data), offset, count);
    }

    void to_device_async(cudaStream_t stream, T *data, size_t offset, size_t count) const {
        transfer<true, true, false>(stream, (void*) data, offset, count);
    }

    void from_device_async(cudaStream_t stream, const T *data, size_t offset, size_t count) {
        transfer<true, false, false>(stream, const_cast<T*>(data), offset, count);
    }


    void to_host(T *data) const {
        to_host(data, 0, capacity);
    }

    void from_host(const T *data) {
        from_host(data, 0, capacity);
    }

    void to_host_async(cudaStream_t stream, T *data) const {
        to_host_async(stream, data, 0, capacity);
    }

    void from_host_async(cudaStream_t stream, const T *data) {
        from_host_async(stream, data, 0, capacity);
    }

    void to_device(T *data) const {
        to_device(data, 0, capacity);
    }

    void from_device(const T *data) {
        from_device(data, 0, capacity);
    }

    void to_device_async(cudaStream_t stream, T *data) const {
        to_device_async(stream, data, 0, capacity);
    }

    void from_device_async(cudaStream_t stream, const T *data) {
        from_device_async(stream, data, 0, capacity);
    }

    void to_host(vector<T> &vec) const {
        vec.resize(size());
        to_host(vec.data());
    }

    void from_host(const vector<T> &vec) {
        from_host(vec.data(), 0, vec.size());
    }

    template <bool PeerOnHost>
    void copy_to(cuda_mem<T, PeerOnHost> &rhs, size_t lhs_offset, size_t rhs_offset, size_t count) const {
        if (rhs_offset + count > rhs.size()) {
            throw cuda_exception("Copy out of bounds");
        }

        if (PeerOnHost) {
            to_host(rhs.get() + rhs_offset, lhs_offset, count);
        } else {
            to_device(rhs.get() + rhs_offset, lhs_offset, count);
        }
    }

    template <bool PeerOnHost>
    void copy_to(cuda_mem<T, PeerOnHost> &rhs) const {
        copy_to(rhs, 0, 0, capacity);
    }

    template <bool PeerOnHost>
    void copy_to_async(cudaStream_t stream, cuda_mem<T, PeerOnHost> &rhs, size_t lhs_offset, size_t rhs_offset, size_t count) const {
        if (rhs_offset + count > rhs.size()) {
            throw cuda_exception("Copy out of bounds");
        }

        if (PeerOnHost) {
            to_host_async(stream, rhs.get() + rhs_offset, lhs_offset, count);
        } else {
            to_device_async(stream, rhs.get() + rhs_offset, lhs_offset, count);
        }
    }

    template <bool PeerOnHost>
    void copy_to_async(cudaStream_t stream, cuda_mem<T, PeerOnHost> &rhs) const {
        copy_to_async(stream, rhs, 0, 0, capacity);
    }

    template <bool PeerOnHost>
    void copy_from(cuda_mem<T, PeerOnHost> &rhs, size_t lhs_offset, size_t rhs_offset, size_t count) {
        rhs.copy_to(*this, rhs_offset, lhs_offset, count);
    }

    template <bool PeerOnHost>
    void copy_from(cuda_mem<T, PeerOnHost> &rhs) {
        rhs.copy_to(*this);
    }

    template <bool PeerOnHost>
    void copy_from_async(cudaStream_t stream, cuda_mem<T, PeerOnHost> &rhs, size_t lhs_offset, size_t rhs_offset, size_t count) {
        rhs.copy_to_async(stream, *this, rhs_offset, lhs_offset, count);
    }

    template <bool PeerOnHost>
    void copy_from_async(cudaStream_t stream, cuda_mem<T, PeerOnHost> &rhs) {
        rhs.copy_to_async(stream, *this);
    }

    T at(size_t index) {
        if (OnHost) {
            return ptr[index];
        } else {
            T val;
            to_host(&val, index, 1);
            return val;
        }
    }

    void swap(cuda_mem &rhs) {
        std::swap(ptr, rhs.ptr);
        std::swap(capacity, rhs.capacity);
    }

    void operator=(cuda_mem &&rhs) {
        swap(rhs);
    }

    void operator=(const cuda_mem &rhs) {
        allocate(rhs.size());
        copy_from(rhs);
    }

    cuda_mem clone() const {
        return cuda_mem(*this);
    }

    INLINE T *get() {
        return ptr;
    }

    INLINE const T *get() const {
        return ptr;
    }

    INLINE size_t size() const {
        return capacity;
    }

    INLINE size_t size_in_bytes() const {
        return size() * sizeof(T);
    }
};


template <typename T>
using cuda_pinned_mem = cuda_mem<T, true>;

template <typename T>
using cuda_device_mem = cuda_mem<T, false>;

CUDA_DEVICE bool cuda_get_bit(bitvec_t *v, size_t index) {
    uint32_t word_index = index / 32;
    uint32_t bit_index = index % 32;
    uint32_t val = ((uint32_t*)v)[word_index];

    int ret;
    asm("bfe.u32 %0, %1, %2, 1;" : "=r"(ret) : "r"(val), "r"(bit_index));
    return ret;
}

CUDA_DEVICE void cuda_atomic_set_bit(bitvec_t *v, size_t index) {
    uint32_t word_index = index / 32;
    uint32_t bit_index = index % 32;
    bitvec_t mask = ((bitvec_t) 1) << bit_index;

    if ((v[word_index] & mask) == 0) {
        atomicOr(((uint32_t*) v) + word_index, mask);
    }
}

template <typename T>
CUDA_DEVICE bool cuda_atomic_cas(T *ptr, T *oldVal, T newVal) {
    bool success = false;

    if (sizeof(T) == sizeof(unsigned int)) {
        unsigned int a = * (unsigned int*) oldVal;
        unsigned int b = * (unsigned int*) &newVal;
        unsigned int c = atomicCAS((unsigned int*) ptr, a, b);
        success = a == c;
        *((unsigned int*) oldVal) = c;
    }

    else if (sizeof(T) == sizeof(unsigned long long)) {
        unsigned long long a = * (unsigned long long*) oldVal;
        unsigned long long b = * (unsigned long long*) &newVal;
        unsigned long long c =  atomicCAS((unsigned long long*) ptr, a, b);
        success = a == c;
        *((unsigned long long*) oldVal) = c;
    }

    else {
        printf("ERROR: unsupported message size\n");
        asm("trap;");
    }

    return success;
}

template <typename F, typename ...Args>
static CUDA_KERNEL void cuda_launch_kernel_wrapper(Args ...args) {
    F()(args...);
}

template <typename F, typename N, typename ...Args>
static void cuda_launch_kernel(cudaStream_t stream, size_t cta_size, F fun, N size, Args ...args) {

    cuda_launch_kernel_wrapper<F, N, Args...><<<ceil_div(size, (N) cta_size), cta_size, 0, stream>>>(size, args...);
    cuda_check_last(typeid(F).name());
}

};


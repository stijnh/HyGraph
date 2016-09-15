#pragma once

#include <thread>
#include <omp.h>
#include <vector>
#include <cstddef>
#include <atomic>

#include "common.hpp"
#include "cuda_common.cuh"
#include "local_graph.hpp"
#include "program.hpp"

namespace hygraph {
using namespace std;

template <typename P, bool SharedMemInbox=true, bool SharedMemActivity=SharedMemInbox>
class HybridBlockGraph {
    static const size_t BLOCK_ALIGNMENT = CUDA_WARP_SIZE;
    static const size_t cta_size = 1024;

    public:
    typedef typename P::vertex_value_t V;
    typedef typename P::edge_value_t E;
    typedef typename P::message_t M;
    typedef typename P::vertex_state_t S;

    int device;
    size_t num_vertices;
    size_t num_edges;
    size_t num_blocks;
    size_t num_threads;

    size_t block_loaded_gpu_offset;
    size_t block_size;

    size_t shared_mem_per_block;

    vector<V> vertex_vals;
    cuda_pinned_mem<S> vertex_state;
    cuda_pinned_mem<bitvec_t> vertex_active;
    cuda_pinned_mem<S> vertex_new_state;
    cuda_pinned_mem<bitvec_t> vertex_new_active;

    cuda_device_mem<V> d_vertex_vals;
    cuda_device_mem<S> d_vertex_state;
    cuda_device_mem<bitvec_t> d_vertex_active;
    cuda_device_mem<S> d_vertex_new_state;
    cuda_device_mem<bitvec_t> d_vertex_new_active;
    cuda_device_mem<M> d_vertex_inbox;

    vector<vector<eid_t> > block_indices;
    vector<vector<vid_t> > block_unique_src;
    vector<vector<vid_t> > block_edge_dst;
    vector<vector<E> > block_edge_vals;

    cuda_device_mem<eid_t> d_block_edge_begin;
    cuda_device_mem<eid_t> d_block_edge_end;
    cuda_device_mem<vid_t> d_block_edge_src;
    cuda_device_mem<vid_t> d_block_edge_dst;
    cuda_device_mem<E> d_block_edge_vals;

    public:
    HybridBlockGraph(int d, size_t t): device(d), num_threads(t) {
        //
    }

    ~HybridBlockGraph() {
        clear();
    }

    void clear() {
        vertex_vals.clear();
        vertex_state.clear();
        vertex_active.clear();

        vertex_new_state.clear();
        vertex_new_active.clear();

        block_indices.clear();
        block_edge_dst.clear();
        block_edge_vals.clear();

        d_block_edge_begin.clear();
        d_block_edge_end.clear();
        d_block_edge_src.clear();
        d_block_edge_dst.clear();
        d_block_edge_vals.clear();
    }

    private:
    void sort_edges_for_blocks(vector<tuple<uint32_t, vid_t, vid_t, E> > &edges) {
        par_sort(edges.begin(), edges.end(), [&](
                    // tuple:   block id  src    dst    value
                    const tuple<uint32_t, vid_t, vid_t, E> &a,
                    const tuple<uint32_t, vid_t, vid_t, E> &b) {
            uint32_t k = get<0>(a);
            uint32_t l = get<0>(b);

            if (k != l) {
                return k < l;
            }

            vid_t x = get<1>(a);
            vid_t y = get<1>(b);

            if (x != y) {
                return x < y;
            }

            vid_t z = get<2>(a);
            vid_t w = get<2>(b);

            return z < w;
        });
    }

    public:
    bool load(LocalGraph<V, E> &g, size_t vertices_per_block) {
        size_t n = g.num_vertices;
        size_t m = g.num_edges;

        block_size = vertices_per_block;

        if (block_size % BLOCK_ALIGNMENT != 0) {
            log("error: block size should be multiple of %d",
                    int(BLOCK_ALIGNMENT));
            return false;
        }

        num_vertices = n;
        num_edges = m;
        num_blocks = ceil_div(n, block_size);

        // Find number of blocks
        log("initializing block-format graph, found %d vertices, %d edges, %d blocks",
                int(num_vertices), int(num_edges), int(num_blocks));

        // Copy vertex values
        log ("copying vertex values");
        size_t s = num_blocks * block_size;
        vertex_vals.resize(s);
        vertex_state.allocate(s);
        vertex_new_state.allocate(s);
        vertex_active.allocate(BITVEC_SIZE(s));
        vertex_new_active.allocate(BITVEC_SIZE(s));
        vertex_vals = g.vertex_vals;

        // Copy edges and sort
        log("copying edges");
        vector<tuple<uint32_t, vid_t, vid_t, E> > edges(m);

#pragma omp parallel for
        for (size_t i = 0; i < m; i++) {
            edges[i] = make_tuple(
                g.edge_dst[i] / block_size,
                g.edge_src[i],
                g.edge_dst[i],
                g.edge_vals[i]);
        }

        log("sorting edges...");
        sort_edges_for_blocks(edges);
        log("finished sorting edges");


        // create blocks
        log("create blocks...");

        block_indices.resize(num_blocks);
        block_unique_src.resize(num_blocks);
        block_edge_dst.resize(num_blocks);
        block_edge_vals.resize(num_blocks);

        size_t offset = 0;
        for (size_t i = 0; i < num_blocks; i++) {
            size_t size = 0;

            while (offset + size < m && get<0>(edges[offset + size]) == i) {
                size++;
            }

            vid_t prev_src = (vid_t)-1;
            vector<vid_t> unq_src;
            vector<eid_t> indices;
            vector<vid_t> dst(size);
            vector<E> vals(size);

            for (size_t j = 0; j < size; j++) {
                vid_t src = get<1>(edges[offset + j]);
                dst[j] = get<2>(edges[offset + j]);
                vals[j] = get<3>(edges[offset + j]);

                if (j == 0 || src != prev_src) {
                    unq_src.push_back(src);
                    indices.push_back(j);
                    prev_src = src;
                }
            }

            indices.push_back(size);

            block_indices[i] = move(indices);
            block_unique_src[i] = move(unq_src);
            block_edge_dst[i] = move(dst);
            block_edge_vals[i] = move(vals);

            offset += size;
        }

        cudaDeviceProp dev_prop;
        CUDA_CHECK(cudaGetDeviceProperties, &dev_prop, device);
        shared_mem_per_block = dev_prop.sharedMemPerBlock;

        log("found device %s (compute capability %d.%d, %d SMs, %.2f GB)",
                dev_prop.name,
                dev_prop.major,
                dev_prop.minor,
                dev_prop.multiProcessorCount,
                dev_prop.totalGlobalMem / 1024.0 / 1024.0 / 1024.0);

        log("initializing device");
        CUDA_CHECK(cudaSetDevice, device);
        CUDA_CHECK(cudaFree, NULL); // force CUDA initialization.

        log("allocating device memory");
        s = num_blocks * block_size;
        d_vertex_vals.allocate(vertex_vals);
        d_vertex_state.allocate(s);
        d_vertex_new_state.allocate(s);
        d_vertex_active.allocate(BITVEC_SIZE(s));
        d_vertex_new_active.allocate(BITVEC_SIZE(s));
        if (!SharedMemInbox) d_vertex_inbox.allocate(s);
        d_block_edge_begin.allocate(num_blocks);
        d_block_edge_end.allocate(num_blocks);


        size_t mem_per_edge = 2 * sizeof(vid_t) + (is_empty_type<E>::value ? 0 : sizeof(E));
        size_t mem_free, mem_total;
        size_t mem_margin = 1024 * 1024; // Keep 1MB global mem free
        CUDA_CHECK(cudaMemGetInfo, &mem_free, &mem_total);

        size_t num_edges_total = 0;
        size_t num_edges_loaded = 0;

        block_loaded_gpu_offset = 0;
        offset = 0;
        for (int i = num_blocks - 1; i >= 0; i--) {
            size_t num = 0;

            while (offset < m && get<0>(edges[m - offset - 1]) == i) {
                offset++;
                num++;
            }

            // Align to CUDA warp size.
            num = ceil_div(num, (size_t)CUDA_WARP_SIZE) * CUDA_WARP_SIZE;

            num_edges_total += num;
            if (num_edges_total * mem_per_edge + mem_margin < mem_free) {
                num_edges_loaded = num_edges_total;
                block_loaded_gpu_offset = i;
            }
        }

        // Not all blocks fit into global memory.
        if (block_loaded_gpu_offset != 0) {
            double conv = 1024.0 * 1024.0 * 1024.0;

            log("entire graph (%.2f GB) cannot be loaded into device memory (%.2f GB free)",
                    (num_edges_total * mem_per_edge) / conv,
                    mem_free / conv);
            log("only last %d blocks were loaded into device memory (%.2f GB)",
                    num_blocks - block_loaded_gpu_offset,
                    (num_edges_loaded * mem_per_edge) / conv);
        }

        d_block_edge_src.allocate(num_edges_loaded);
        d_block_edge_dst.allocate(num_edges_loaded);
        if (!is_empty_type<E>::value) {
            d_block_edge_vals.allocate(num_edges_loaded);
        }

        d_block_edge_begin.fill(0);
        d_block_edge_end.fill(0);

        size_t host_offset = 0;
        size_t dev_offset = 0;
        for (size_t i = 0; i < num_blocks; i++) {
            size_t size = 0;

            while (host_offset + size < m && get<0>(edges[host_offset + size]) == i) {
                size++;
            }

            if (i >= block_loaded_gpu_offset) {
                vector<vid_t> srcs(size);
                vector<vid_t> dsts(size);
                vector<E> vals(size);

#pragma omp parallel for
                for (size_t j = 0; j < size; j++) {
                    srcs[j] = get<1>(edges[host_offset + j]);
                    dsts[j] = get<2>(edges[host_offset + j]);
                    vals[j] = get<3>(edges[host_offset + j]);
                }

                eid_t dev_begin = dev_offset;
                eid_t dev_end = dev_begin + size;

                d_block_edge_begin.from_host(&dev_begin, i, 1);
                d_block_edge_end.from_host(&dev_end, i, 1);

                d_block_edge_src.from_host(srcs.data(), dev_begin, size);
                d_block_edge_dst.from_host(dsts.data(), dev_begin, size);
                if (!is_empty_type<E>::value) {
                    d_block_edge_vals.from_host(vals.data(), dev_begin, size);
                }

                // Align dev_offset to CUDA warp size, this increases memory coalescing
                // since the next block will start at the right memory alignment.
                dev_offset += size;
                dev_offset = ceil_div(dev_offset, (size_t) CUDA_WARP_SIZE) * CUDA_WARP_SIZE;
            }

            host_offset += size;
        }

        log("finished initialization");
        return true;
    }

    size_t process_block_cpu(const P &program, int superstep, size_t bid) {
        size_t size = block_size;
        size_t offset = bid * size;

        if (offset > num_vertices) {
            return 0;
        }

        if (offset + size > num_vertices) {
            size = num_vertices - offset;
        }

        vector<M> inbox(size);
        vector<bitvec_t> inbox_active(BITVEC_SIZE(size));
        fill(inbox_active.begin(), inbox_active.end(), 0);

        if (P::activity == ACTIVITY_ALWAYS) {
            for (size_t i = 0; i < size; i++) {
                program.init_message(inbox[i]);
            }
        }

        size_t num_indices = block_indices[bid].size() - 1;
        eid_t *indices = block_indices[bid].data();
        vid_t *unique_src = block_unique_src[bid].data();
        vid_t *edge_dst = block_edge_dst[bid].data();
        E *edge_vals = block_edge_vals[bid].data();

        for (size_t i = 0; i < num_indices; i++) {
            vid_t a = unique_src[i];

            if (P::activity == ACTIVITY_SELECTED && !BITVEC_GET(vertex_active.get(), a)) {
                continue;
            }

            size_t begin = indices[i];
            size_t end = indices[i + 1];
            M msg;

            if (!program.generate_message(superstep, vertex_vals[a], vertex_state.at(a), msg)) {
                continue;
            }

            size_t j = begin;

            do {
                vid_t b = edge_dst[j];
                M m = msg;

                if (!program.process_edge(superstep, edge_vals[j], m)) {
                    continue;
                }

                vid_t b_rel = b - offset;

                if (P::activity == ACTIVITY_ALWAYS || BITVEC_GET(inbox_active, b_rel)) {
                    program.aggregate(m, inbox[b_rel]);
                } else {
                    BITVEC_SET(inbox_active, b_rel);
                    inbox[b_rel] = m;
                }

                j++;
            } while (unlikely(j != end));
        }

        size_t num_active = 0;

        for (size_t i = 0; i < size; i++) {
            S state = vertex_state.at(i + offset);
            bool act = false;

            if (P::activity == ACTIVITY_ALWAYS || BITVEC_GET(inbox_active, i)) {
                act = program.process_vertex(
                        superstep,
                        vertex_vals[i + offset],
                        inbox[i],
                        state);
            }

            vertex_new_state.get()[i + offset] = state;

            if (P::activity == ACTIVITY_SELECTED) {
                BITVEC_TOGGLE(vertex_new_active.get(), i + offset, act);
                if (act) num_active++;
            }
        }

        return num_active;
    }

    struct kernel_process_block {
        CUDA_DEVICE void operator()(
                const int superstep,
                const P program,
                const size_t start_gblock_id,
                const size_t gblock_size,
                const V *__restrict__ vertex_vals,
                const S *__restrict__ vertex_state,
                const bitvec_t *__restrict__ vertex_active,
                S *__restrict__ vertex_new_state,
                bitvec_t *__restrict__ vertex_new_active,
                M *__restrict__ global_inbox,
                const eid_t *__restrict__ edge_gblock_begin,
                const eid_t *__restrict__ edge_gblock_end,
                const vid_t *__restrict__ edge_src,
                const vid_t *__restrict__ edge_dst,
                const E *__restrict__ edge_vals,
                vid_t *__restrict__ num_active) {
            extern __shared__ char smem[];

            int bid = blockIdx.x + start_gblock_id;
            int tid = threadIdx.x;

            int offset = gblock_size * bid;
            int size = gblock_size;

            M *shared_inbox =
                !SharedMemInbox || is_empty_type<M>::value
                ? NULL
                : (M*) smem;

            bitvec_t *shared_inbox_active =
                P::activity == ACTIVITY_ALWAYS || !SharedMemActivity
                ? NULL
                : (shared_inbox != NULL
                        ? (bitvec_t*) &shared_inbox[size]
                        : (bitvec_t*) smem);

            if (!is_empty_type<M>::value) {
                for (vid_t i = tid; i < size; i+= cta_size) {
                    if (SharedMemInbox) {
                        program.init_message(shared_inbox[i]);
                    } else {
                        program.init_message(global_inbox[i + offset]);
                    }
                }
            }

            if (P::activity == ACTIVITY_SELECTED) {
                for (vid_t i = tid; i < BITVEC_SIZE(size); i += cta_size) {
                    if (SharedMemActivity) {
                        shared_inbox_active[i] = 0;
                    } else {
                        ((uint32_t*)vertex_new_active)[offset / CUDA_WARP_SIZE + i] = 0;
                    }
                }
            }

            __syncthreads();

            eid_t begin = edge_gblock_begin[bid];
            eid_t end = edge_gblock_end[bid];

            for (eid_t i = begin + tid; i < end; i += cta_size) {
                vid_t src = edge_src[i];

                if (P::activity == ACTIVITY_SELECTED && !BITVEC_GET(vertex_active, src)) {
                    continue;
                }

                vid_t dst = edge_dst[i];
                M msg;

                if (!program.generate_message(superstep, vertex_vals[src], vertex_state[src], msg)) {
                    continue;
                }

                if (!program.process_edge(superstep, edge_vals[i], msg)) {
                    continue;
                }

                if (P::activity == ACTIVITY_SELECTED) {
                    if (SharedMemActivity) {
                        cuda_atomic_set_bit(shared_inbox_active, dst - offset);
                    } else {
                        cuda_atomic_set_bit(vertex_new_active, dst);
                    }
                }

                if (!is_empty_type<M>::value) {
                    if (SharedMemInbox) {
                        program.cuda_atomic_aggregate(msg, shared_inbox + dst - offset);
                    } else {
                        program.cuda_atomic_aggregate(msg, global_inbox + dst);
                    }
                }
            }

            __syncthreads();

            for (vid_t i = tid; i < size; i += cta_size) {
                bool act;
                S state = vertex_state[i + offset];

                if (P::activity == ACTIVITY_ALWAYS) {
                    act = true;
                } else if (SharedMemActivity) {
                    act = cuda_get_bit(shared_inbox_active, i);
                } else {
                    act = cuda_get_bit(vertex_new_active, i + offset);
                }

                if (act) {
                    M msg;

                    if (!is_empty_type<M>::value) {
                        if (SharedMemInbox) {
                            msg = shared_inbox[i];
                        } else {
                            msg = global_inbox[i + offset];
                        }
                    }

                    act = program.process_vertex(
                            superstep,
                            vertex_vals[i + offset],
                            msg,
                            state);
                }

                vertex_new_state[i + offset] = state;

                if (P::activity == ACTIVITY_SELECTED) {
                    uint32_t ballot_result = __ballot(act);

                    if (tid % CUDA_WARP_SIZE == 0) {
                        ((uint32_t*) vertex_new_active)[(i + offset) / CUDA_WARP_SIZE] = ballot_result;
                    }

                }
            }

            if (P::activity == ACTIVITY_SELECTED) {
                __syncthreads();

                int *shared_num_active = (int*)smem;
                int warp_num_active = 0;

                for (vid_t i = tid; i < BITVEC_SIZE(size); i += cta_size) {
                    warp_num_active += __popc(((uint32_t*)vertex_new_active)[offset / CUDA_WARP_SIZE + i]);
                }

                shared_num_active[tid] = warp_num_active;

                int delta = 1;

#pragma unroll 20
                while (delta * 2 < cta_size) {
                    delta *= 2;
                }

#pragma unroll 20
                for (; delta > 0; delta /= 2) {
                    if (delta >= 32) {
                        __syncthreads();
                    }

                    if (tid > delta) {
                        if (delta >= 32) {
                            continue;
                        } else {
                            break;
                        }
                    }

                    if (tid + delta < cta_size) {
                        shared_num_active[tid] += shared_num_active[tid + delta];
                    }
                }


                if (tid == 0) {
                    int block_num_active = shared_num_active[0];

                    if (block_num_active > 0) {
                        atomicAdd(num_active, block_num_active);
                        __threadfence_system();
                    }
                }
            }
        }
    };


    void process_blocks_gpu(cudaStream_t stream, const P &program, int superstep,
            size_t begin_bid, size_t end_bid, vid_t *d_num_active) {
        if (begin_bid >= end_bid) {
            return;
        }

        CUDA_CHECK(cudaSetDevice, device);
        if (sizeof(M) == 4) {
            CUDA_CHECK(cudaDeviceSetSharedMemConfig, cudaSharedMemBankSizeFourByte);
        } else if (sizeof(M) == 8) {
            CUDA_CHECK(cudaDeviceSetSharedMemConfig, cudaSharedMemBankSizeEightByte);
        } else {
            CUDA_CHECK(cudaDeviceSetSharedMemConfig, cudaSharedMemBankSizeDefault);
        }

        size_t smem_size = 0;

        if (P::activity == ACTIVITY_SELECTED && SharedMemActivity) {
            smem_size += sizeof(bitvec_t) * BITVEC_SIZE(block_size);
        }

        if (!is_empty_type<M>::value && SharedMemInbox) {
            smem_size += sizeof(M) * block_size;
        }

        if (P::activity == ACTIVITY_SELECTED) {
            smem_size = max(smem_size, cta_size * sizeof(int));
        }

        if (smem_size <= 16 * 1024) {
            CUDA_CHECK(cudaDeviceSetCacheConfig, cudaFuncCachePreferL1);
        } else if (smem_size <= 32 * 1024) {
            CUDA_CHECK(cudaDeviceSetCacheConfig, cudaFuncCachePreferEqual);
        } else {
            CUDA_CHECK(cudaDeviceSetCacheConfig, cudaFuncCachePreferShared);
        }

        // Not enough shared memory. Do not crash, but just print an error since
        // otherwise benchmarks could fail mid-execution.
        if (smem_size > shared_mem_per_block) {
            log("required amount of shared memory exceeds maximum");
            return;
        }

        size_t num_ctas = end_bid - begin_bid;

        cuda_launch_kernel_wrapper<
                kernel_process_block
            ><<<
                num_ctas,
                cta_size,
                smem_size,
                stream
            >>>(
                superstep,
                program,
                begin_bid,
                block_size,
                d_vertex_vals.get(),
                d_vertex_state.get(),
                d_vertex_active.get(),
                d_vertex_new_state.get(),
                d_vertex_new_active.get(),
                d_vertex_inbox.get(),
                d_block_edge_begin.get(),
                d_block_edge_end.get(),
                d_block_edge_src.get(),
                d_block_edge_dst.get(),
                d_block_edge_vals.get(),
                d_num_active);

        cuda_check_last("kernel_process_block");
    }

    template <bool ToHost>
    void sync_vertices(cudaStream_t stream, size_t offset, size_t len) {
        if (offset % (sizeof(bitvec_t) * 8) != 0 || len % (sizeof(bitvec_t) * 8) != 0) {
            throw runtime_error("fatal copy error");
        }

        if (!is_empty_type<S>::value) {
            // To host
            if (ToHost) {
                d_vertex_new_state.copy_to_async(
                        stream,
                        vertex_new_state,
                        offset,
                        offset,
                        len);
            }

            // To device
            else {
                vertex_new_state.copy_to_async(
                        stream,
                        d_vertex_new_state,
                        offset,
                        offset,
                        len);
            }
        }

        if (P::activity == ACTIVITY_SELECTED) {
            if (ToHost) {
                d_vertex_new_active.copy_to_async(
                        stream,
                        vertex_new_active,
                        BITVEC_SIZE(offset),
                        BITVEC_SIZE(offset),
                        BITVEC_SIZE(len));
            } else {
                vertex_new_active.copy_to_async(
                        stream,
                        d_vertex_new_active,
                        BITVEC_SIZE(offset),
                        BITVEC_SIZE(offset),
                        BITVEC_SIZE(len));
            }
        }
    }

    void sync_blocks_gpu_to_cpu(cudaStream_t stream, size_t begin_bid, size_t end_bid) {
        sync_vertices<true>(stream, begin_bid * block_size, (end_bid - begin_bid) * block_size);
    }

    void sync_blocks_cpu_to_gpu(cudaStream_t stream, size_t bid) {
        sync_vertices<false>(stream, bid * block_size, block_size);
    }

    void initialize_state(const P &program) {
#pragma omp parallel for schedule(static, 1)
        for (size_t i = 0; i < num_blocks; i++) {
            size_t offset = i * block_size;
            size_t size = block_size;

            for (vid_t v = offset; v < offset + size; v++) {
                bool act = program.init_state(
                        vertex_vals[v],
                        vertex_state.get()[v]);

                BITVEC_TOGGLE(vertex_active.get(), v, act);
            }
        }

        log("copying vertices state to device");
        vertex_state.copy_to(d_vertex_state);
        vertex_active.copy_to(d_vertex_active);
    }


    void run_gpu(const P &program, size_t max_iter) {

        // If loaded_offset > 0 then some blocks are not loaded into
        // memory, only hybrid execution is possible.
        if (block_loaded_gpu_offset != 0) {
            log("processing failed since not all blocks loaded into GPU memory");
            return;
        }

        CUDA_CHECK(cudaSetDevice, device);
        cudaStream_t stream;
        CUDA_CHECK(cudaStreamCreate, &stream);

        initialize_state(program);

        cuda_device_mem<vid_t> d_nactive(1);
        d_nactive.fill(0);

        log("starting algorithm");
        size_t it = 0;

        double before = timer();
        while (it < max_iter) {
            process_blocks_gpu(
                    stream,
                    program,
                    it,
                    0,
                    num_blocks,
                    d_nactive.get());

            it++;
            d_vertex_state.swap(d_vertex_new_state);
            d_vertex_active.swap(d_vertex_new_active);

            if (P::activity == ACTIVITY_SELECTED) {
                vid_t zero = 0;
                vid_t nactive;

                d_nactive.to_host_async(stream, &nactive);
                d_nactive.from_host_async(stream, &zero);
                CUDA_CHECK(cudaStreamSynchronize, stream);
                log("nactive=%d", nactive);

                if (nactive == 0) {
                    break;
                }
            }
        }
        CUDA_CHECK(cudaStreamSynchronize, stream);
        double after = timer();

        log("performed %d iterations in %.5f seconds on GPU", int(it), after - before);

        CUDA_CHECK(cudaStreamDestroy, stream);
    }


    void run_cpu(const P &program, size_t max_iter) {
        initialize_state(program);

        atomic<vid_t> nactive;
        atomic<size_t> counter;
        size_t it = 0;

        double before = timer();
        while (it < max_iter) {
            nactive.store(0);
            counter.store(0);

#pragma omp parallel num_threads(num_threads)
            {
                while (true) {
                    size_t bid = counter.fetch_add(1);

                    if (bid >= num_blocks) {
                        break;
                    }

                    size_t a = process_block_cpu(
                            program,
                            it,
                            bid);

                    nactive.fetch_add(a);
                }
            }

            it++;
            vertex_state.swap(vertex_new_state);
            vertex_active.swap(vertex_new_active);
            log("nactive=%d", int(nactive.load()));

            if (P::activity == ACTIVITY_SELECTED && nactive.load() == 0) {
                break;
            }
        }
        double after = timer();

        log("performed %d iterations in %.5f seconds on CPU", int(it), after - before);
    }

    template <bool Verbose=false>
    void run_hybrid_static(const P &program, size_t max_iter, double alpha, size_t gpu_num_streams=8, size_t gpu_batch_size=20) {
        return run_hybrid<false, Verbose>(program, max_iter, gpu_num_streams, gpu_batch_size, alpha);
    }

    template <bool Verbose=false>
    void run_hybrid_dynamic(const P &program, size_t max_iter, size_t gpu_num_streams=8, size_t gpu_batch_size=20) {
        return run_hybrid<true, Verbose>(program, max_iter, gpu_num_streams, gpu_batch_size);
    }

    template <bool Dynamic, bool Verbose>
    void run_hybrid(const P &program, size_t max_iter, size_t gpu_num_streams, size_t gpu_batch_size, double alpha=-1) {
        CUDA_CHECK(cudaSetDevice, device);

        cudaStream_t d2h_stream, h2d_stream;
        CUDA_CHECK(cudaStreamCreate, &d2h_stream);
        CUDA_CHECK(cudaStreamCreate, &h2d_stream);

        cudaEvent_t d2h_event, h2d_event;
        CUDA_CHECK(cudaEventCreate, &d2h_event, cudaEventDisableTiming | cudaEventBlockingSync);
        CUDA_CHECK(cudaEventCreate, &h2d_event, cudaEventDisableTiming | cudaEventBlockingSync);

        size_t num_streams = gpu_num_streams;
        vector<cudaStream_t> exe_streams(num_streams);
        vector<cudaEvent_t> exe_events(num_streams);

        for (size_t i = 0; i < num_streams; i++) {
            CUDA_CHECK(cudaStreamCreate, &exe_streams[i]);
            CUDA_CHECK(cudaEventCreate, &exe_events[i], cudaEventDisableTiming | cudaEventBlockingSync);
        }

        vid_t *h_mapped_nactive, *d_mapped_nactive;
        CUDA_CHECK(cudaHostAlloc, &h_mapped_nactive, sizeof(vid_t), cudaHostAllocMapped);
        CUDA_CHECK(cudaHostGetDevicePointer, &d_mapped_nactive, h_mapped_nactive, 0);

        initialize_state(program);
        mutex lock;
        atomic<vid_t> nactive;
        size_t cpu_counter;
        size_t gpu_counter;
        size_t it = 0;

        size_t max_cpu_counter;
        size_t max_gpu_counter;

        if (Dynamic) {
            max_cpu_counter = num_blocks;
            max_gpu_counter = num_blocks - block_loaded_gpu_offset; // blocks 0...loaded_offset not in GPU mem.
        } else {
            max_cpu_counter = (size_t)(alpha * num_blocks + 0.5);
            if (max_cpu_counter > num_blocks)              max_cpu_counter = num_blocks;
            if (max_cpu_counter < block_loaded_gpu_offset) max_cpu_counter = block_loaded_gpu_offset;

            max_gpu_counter = num_blocks - max_cpu_counter;
        }

        double cpu_waiting = 0, gpu_waiting = 0;
        double cpu_proc = 0, gpu_proc = 0;
        double cpu_idle = 0, gpu_idle = 0;

        double before = timer();
        while (it < max_iter) {
            nactive.store(0);
            cpu_counter = 0;
            gpu_counter = 0;
            *h_mapped_nactive = 0;

            double before_iter = timer();
            double cpu_wait_done, gpu_wait_done;
            double cpu_proc_done, gpu_proc_done;

#pragma omp parallel num_threads(num_threads + 1)
            {
                CUDA_CHECK(cudaSetDevice, device);

                // GPU thread
                if (omp_get_thread_num() == 0) {
                    CUDA_CHECK(cudaEventSynchronize, h2d_event);
                    gpu_wait_done = timer();

                    for (size_t i = 0;; i = (i + 1) % num_streams) {
                        cudaError_t code = cudaStreamQuery(exe_streams[i]);

                        // Stream not ready, check next stream
                        if (code == cudaErrorNotReady) {
                            continue;
                        }

                        // Stream error, throw error
                        if (code != cudaSuccess) {
                            throw cuda_exception("cudaStreamQuery", code);
                        }

                        size_t begin_bid;
                        size_t end_bid;

                        {
                            lock_guard<mutex> guard(lock);

                            size_t remaining = num_blocks - (cpu_counter + gpu_counter);
                            size_t batch = gpu_batch_size;
                            batch = min(batch, remaining);
                            batch = min(batch, max_gpu_counter - gpu_counter);

                            if (batch == 0) {
                                break;
                            }

                            end_bid = num_blocks - gpu_counter;
                            begin_bid = end_bid - batch;
                            gpu_counter += batch;
                        }

                        process_blocks_gpu(
                                exe_streams[i],
                                program,
                                it,
                                begin_bid,
                                end_bid,
                                d_mapped_nactive);

                        CUDA_CHECK(cudaEventRecord,
                                exe_events[i],
                                exe_streams[i]);

                        CUDA_CHECK(cudaStreamWaitEvent,
                            d2h_stream,
                            exe_events[i],
                            0);

                        sync_blocks_gpu_to_cpu(d2h_stream,
                                begin_bid,
                                end_bid);

                    }

                    for (size_t i = 0; i < num_streams; i++) {
                        CUDA_CHECK(cudaEventSynchronize, exe_events[i]);
                    }

                    if (P::activity == ACTIVITY_SELECTED) {
                        nactive.fetch_add(*h_mapped_nactive);
                    }

                    gpu_proc_done = timer();
                }

                // CPU thread
                else {
                    CUDA_CHECK(cudaEventSynchronize, d2h_event);
                    cpu_wait_done = timer();

                    while (true) {
                        size_t bid;

                        {
                            lock_guard<mutex> guard(lock);

                            if (cpu_counter + gpu_counter >= num_blocks) {
                                break;
                            }

                            if (cpu_counter >= max_cpu_counter) {
                                break;
                            }

                            bid = cpu_counter++;
                        }

                        size_t x = process_block_cpu(
                                program,
                                it,
                                bid);

                        sync_blocks_cpu_to_gpu(h2d_stream, bid);

                        if (P::activity == ACTIVITY_SELECTED) {
                            nactive.fetch_add(x);
                        }
                    }

                    cpu_proc_done = timer();
                }
            }

            double after_iter = timer();

            if (Verbose) {
                log("timing for iteration %d:", it + 1);
                log("       | CPU      | GPU ");
                log("-------+----------+----------");
                log("blocks | %8d | %8d",
                        cpu_counter,
                        gpu_counter);
                log("wait   | %8.5f | %8.5f ",
                        cpu_wait_done - before_iter,
                        gpu_wait_done - before_iter);
                log("proc.  | %8.5f | %8.5f ",
                        cpu_proc_done - cpu_wait_done,
                        gpu_proc_done - gpu_wait_done);
                log("idle   | %8.5f | %8.5f ",
                        after_iter - cpu_proc_done,
                        after_iter - gpu_proc_done);

                cpu_waiting += cpu_wait_done - before_iter;
                cpu_proc += cpu_proc_done - cpu_wait_done;
                cpu_idle += after_iter - cpu_proc_done;

                gpu_waiting += gpu_wait_done - before_iter;
                gpu_proc += gpu_proc_done - gpu_wait_done;
                gpu_idle += after_iter - gpu_proc_done;

            }

            CUDA_CHECK(cudaEventRecord, h2d_event, h2d_stream);
            CUDA_CHECK(cudaEventRecord, d2h_event, d2h_stream);

            it++;
            vertex_state.swap(vertex_new_state);
            vertex_active.swap(vertex_new_active);
            d_vertex_state.swap(d_vertex_new_state);
            d_vertex_active.swap(d_vertex_new_active);

            if (P::activity == ACTIVITY_SELECTED && nactive.load() == 0) {
                break;
            }
        }
        double after = timer();

        if (Verbose) {
            log("       | CPU      | GPU ");
            log("-------+----------+----------");
            log("wait   | %8.5f | %8.5f ",
                    cpu_waiting, gpu_waiting);
            log("proc.  | %8.5f | %8.5f ",
                    cpu_proc, gpu_proc);
            log("idle   | %8.5f | %8.5f ",
                    cpu_idle, gpu_idle);
        }

        log("performed %d iterations in %.5f seconds on CPU+GPU", int(it), after - before);

        CUDA_CHECK(cudaStreamSynchronize, d2h_stream);
        CUDA_CHECK(cudaStreamSynchronize, h2d_stream);

        CUDA_CHECK(cudaStreamDestroy, d2h_stream);
        CUDA_CHECK(cudaStreamDestroy, h2d_stream);

        CUDA_CHECK(cudaEventDestroy, d2h_event);
        CUDA_CHECK(cudaEventDestroy, h2d_event);

        for (size_t i = 0; i < num_streams; i++) {
            CUDA_CHECK(cudaStreamDestroy, exe_streams[i]);
            CUDA_CHECK(cudaEventDestroy, exe_events[i]);
        }

        CUDA_CHECK(cudaFreeHost, h_mapped_nactive);
    }
};

}

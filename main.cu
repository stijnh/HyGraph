#include <climits>

#include "local_graph.hpp"
#include "hybrid_block_graph.hpp"
#include "program.hpp"

using namespace std;
using namespace hygraph;


vid_t find_good_traversal_root(const LocalGraph<vid_t, empty_t> &lg) {
    vid_t best_root = 0;
    size_t best_root_count = 0;

#pragma omp parallel for schedule(dynamic,1)
    for (size_t i = 0; i < 100; i++) {
        vid_t root = rand() % lg.num_vertices;

        vector<bool> active(lg.num_vertices);
        fill(active.begin(), active.end(), false);
        active[root] = true;

        size_t iter = 0;
        bool updated = true;
        do {
            updated = false;

            for (size_t j = 0; j < lg.num_edges; j++) {
                if (active[lg.edge_src[j]] && !active[lg.edge_dst[j]]) {
                    active[lg.edge_dst[j]] = true;
                    updated = true;
                }
            }
        } while(updated && iter++ < 20);

        size_t count = 0;
        for (size_t v = 0; v < lg.num_vertices; v++) {
            if (active[v]) {
                count++;
            }
        }

        log("attempt %d: root %d covers %.2f%%", i, root,
                count / double(lg.num_vertices) * 100);

#pragma omp critical
        {
            if (count > best_root_count) {
                best_root = root;
                best_root_count = count;
            }
        }
    }

    log("root %d covers %.2f%% of the vertices", best_root,
            best_root_count / double(lg.num_vertices) * 100.0);

    return best_root;
}

template <typename P, typename V, typename E>
void run(LocalGraph<V,E> &lg, P prog, size_t max_iter) {
    size_t block_size = (1 << 13);
    size_t num_streams = 8;
    size_t batch_size = 20;

    // Shuffle order of vertices to ensure they are evenly spread out
    lg.shuffle_vertices();

    // Sort vertices within each block by out-degree, this increases
    // the read locality of the vertices since most frequently
    // accessed vertices are at the front of each block.
    lg.sort_vertices_degree(EDGE_OUT, true, block_size);

    HybridBlockGraph<P> g(0, 16);
    g.load(lg, block_size);
    g.run_hybrid_dynamic(prog, max_iter, num_streams, batch_size);
}

void run_bfs(LocalGraph<vid_t,empty_t> &lg) {
    vid_t root = lg.vertex_vals[find_good_traversal_root(lg)];
    run<BFSProgram>(lg, BFSProgram(root), 100);
}

void run_sssp(LocalGraph<vid_t,empty_t> &lg) {
    vid_t root = lg.vertex_vals[find_good_traversal_root(lg)];

    LocalGraph<vid_t, float> lgp;
    swap(lgp.num_vertices, lg.num_vertices);
    swap(lgp.num_edges, lg.num_edges);
    swap(lgp.vertex_vals, lg.vertex_vals);
    swap(lgp.edge_src, lg.edge_src);
    swap(lgp.edge_dst, lg.edge_dst);

    // Generate random edge values
    lgp.edge_vals.resize(lgp.num_edges);

    for (size_t i = 0; i < lgp.num_edges; i++) {
        lgp.edge_vals[i] = rand() / float(RAND_MAX);
    }

    run<SSSPProgram>(lgp, SSSPProgram(root), 100);
}

void run_pr(LocalGraph<vid_t,empty_t> &lg) {
    LocalGraph<uint32_t, empty_t> lgp;
    swap(lgp.num_vertices, lg.num_vertices);
    swap(lgp.num_edges, lg.num_edges);
    swap(lgp.vertex_vals, lg.vertex_vals);
    swap(lgp.edge_src, lg.edge_src);
    swap(lgp.edge_dst, lg.edge_dst);


    // Set the out-degree of each vertex as its value.
    lgp.vertex_vals.resize(lgp.num_vertices);
    fill(lgp.vertex_vals.begin(), lgp.vertex_vals.end(), 0);

    for (size_t i = 0; i < lg.num_edges; i++) {
        lgp.vertex_vals[lgp.edge_src[i]]++;
    }

    run<PRProgram>(lgp, PRProgram(), 10);
}

void run_cc(LocalGraph<vid_t,empty_t> &lg) {
    lg.to_undirected();
    run<ConnProgram>(lg, ConnProgram(), 100);
}

int main(int argc, char *argv[]) {
    LocalGraph<vid_t,empty_t> lg;

    string file = "test.txt";
    string alg = "bfs";

    if (argc > 1) file = string(argv[1]);
    if (argc > 2) alg = string(argv[2]);

    log("reading from file: %s", file.c_str());

    if (file.find(".bin") != string::npos) {
        load_from_binary_file(lg, file);
    } else {
        load_from_file(lg, file);
    }

    log("executing algorithm: %s", alg.c_str());

    if (alg == "bfs") {
        run_bfs(lg);
    } else if (alg == "pr") {
        run_pr(lg);
    } else if (alg == "cc") {
        run_cc(lg);
    } else if (alg == "sssp") {
        run_sssp(lg);
    } else {
        log("unknown algorithm");
    }
}

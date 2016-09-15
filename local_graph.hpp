#pragma once

#include <algorithm>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>

#include "common.hpp"

namespace hygraph {
using namespace std;

template <typename V, typename E>
class LocalGraph {
    public:
    size_t num_vertices;
    vector<V> vertex_vals;

    size_t num_edges;
    vector<vid_t> edge_src;
    vector<vid_t> edge_dst;
    vector<E> edge_vals;

    LocalGraph() {
        num_vertices = 0;
        num_edges = 0;
    }

    void clear() {
        num_vertices = 0;
        vertex_vals.clear();

        num_edges = 0;
        edge_src.clear();
        edge_dst.clear();
        edge_vals.clear();
    }

    vid_t add_vertex(const V value) {
        vertex_vals.push_back(value);
        return num_vertices++;
    }

    eid_t add_edge(const vid_t &src, const vid_t &dst, const E &value=E()) {
        edge_src.push_back(src);
        edge_dst.push_back(dst);
        edge_vals.push_back(value);
        return num_edges++;
    }

    void invert() {
#pragma omp parallel for
        for (size_t i = 0; i < num_edges; i++) {
            swap(edge_src[i], edge_dst[i]);
        }
    }

    void to_canonical() {
#pragma omp parallel for
        for (size_t i = 0; i < num_edges; i++) {
            vid_t a = edge_src[i];
            vid_t b = edge_dst[i];

            edge_src[i] = min(a, b);
            edge_dst[i] = max(a, b);
        }
    }

    void sort_edges() {
        vector<tuple<vid_t, vid_t, E> > edges(num_edges);

#pragma omp parallel for
        for (size_t i = 0; i < num_edges; i++) {
            edges[i] = make_tuple(
                    edge_src[i],
                    edge_dst[i],
                    edge_vals[i]);
        }

        par_sort(edges.begin(), edges.end(), [](const tuple<vid_t, vid_t, E> a, const tuple<vid_t, vid_t, E> b) {
                return get<0>(a) != get<0>(b) ?
                    get<0>(a) < get<0>(b) :
                    get<1>(a) < get<1>(b);
        });

#pragma omp parallel for
        for (size_t i = 0; i < num_edges; i++) {
            edge_src[i] = get<0>(edges[i]);
            edge_dst[i] = get<1>(edges[i]);
            edge_vals[i] = get<2>(edges[i]);
        }
    }

    void remove_duplicate_edges() {
        sort_edges();

        size_t index = 0;

        for (size_t i = 0; i < num_edges; i++) {
            if (i == 0 || edge_src[i] != edge_src[i - 1] || edge_dst[i] != edge_dst[i - 1]) {
                edge_src[index] = edge_src[i];
                edge_dst[index] = edge_dst[i];
                edge_vals[index] = edge_vals[i];
                index++;
            }
        }

        size_t diff = num_edges - index;

        num_edges = index;
        edge_src.resize(num_edges);
        edge_dst.resize(num_edges);
        edge_vals.resize(num_edges);

        log("removed %d edges", diff);
    }


    void to_undirected() {
        to_canonical();
        remove_duplicate_edges();

        edge_src.resize(num_edges * 2);
        edge_dst.resize(num_edges * 2);
        edge_vals.resize(num_edges * 2);

#pragma omp parallel for
        for (size_t i = 0; i < num_edges; i++){
            size_t j = i + num_edges;

            // Copy edge i to edge j, but swap src and dst.
            edge_src[j] = edge_dst[i];
            edge_dst[j] = edge_src[i];
            edge_vals[j] = edge_vals[i];
        }

        num_edges *= 2;
    }

    void remap_vertices(const vector<vid_t> &vmap) {
        const vector<V> old_vertex_vals = vertex_vals;

        for (size_t i = 0; i < num_vertices; i++) {
            vertex_vals[vmap[i]] = old_vertex_vals[i];
        }

#pragma omp parallel for
        for (size_t i = 0; i < num_edges; i++) {
            edge_src[i] = vmap[edge_src[i]];
            edge_dst[i] = vmap[edge_dst[i]];
        }
    }

    void shuffle_vertices() {
        vector<vid_t> vmap(num_vertices);

        for (size_t i = 0; i < num_vertices; i++) {
            vmap[i] = i;
        }

        random_shuffle(vmap.begin(), vmap.end());
        remap_vertices(vmap);
    }

    void sort_vertices_degree(const EdgeDir dir, const bool reversed, size_t block_size) {
        vector<size_t> degree(num_vertices);

        for (size_t i = 0; i < num_edges; i++) {
            if (dir & EDGE_IN) {
                degree[edge_dst[i]]++;
            }

            if (dir & EDGE_OUT) {
                degree[edge_src[i]]++;
            }
        }

        vector<vid_t> order(num_vertices);

        for (size_t i = 0; i < num_vertices; i++) {
            order[i] = i;
        }

        par_sort(order.begin(), order.end(), [&](const vid_t a, const vid_t b) {
                int x = a / block_size; // block id for vertex a
                int y = b / block_size; // block id for vertex b

                if (x != y) return x < y; // different blocks, keep ordering.

                return reversed
                    ? degree[a] > degree[b]
                    : degree[a] < degree[b];
        });

        vector<vid_t> vmap(num_vertices);

        for (size_t i = 0; i < num_vertices; i++) {
            vmap[order[i]] = i;
        }

        remap_vertices(vmap);
    }
};

#define MAGIC_WORD ("HET_GRAPH_BINARY_FORMAT")

template <typename V, typename E>
bool write_to_binary_file(LocalGraph<V,E> &g, const string &filename) {
    std::ofstream f(filename.c_str(), ios_base::binary | ios_base::out);

    size_t n = g.num_vertices;
    size_t m = g.num_edges;

    f.write((char*) MAGIC_WORD, string(MAGIC_WORD).size());

    f.write((char*) &n, sizeof(size_t));
    f.write((char*) &m, sizeof(size_t));

    f.write((char*) g.vertex_vals.data(), sizeof(V) * n);
    f.write((char*) g.edge_src.data(), sizeof(vid_t) * m);
    f.write((char*) g.edge_dst.data(), sizeof(vid_t) * m);
    f.write((char*) g.edge_vals.data(), sizeof(E) * m);

    f.flush();

    bool success = !!f;
    f.close();

    if (!success) {
        std::cerr << "failed to write to file: " << filename << std::endl;
        return false;
    }

    log("wrote %d vertices and %d edges to %s", n, m, filename.c_str());
    return true;
}

template <typename V, typename E>
bool load_from_binary_file(LocalGraph<V,E> &g, const string &filename) {
    ifstream f(filename.c_str(), ios_base::binary | ios_base::in);

    if (!f) {
        std::cerr << "failed to open file: " << filename << endl;
        return false;
    }

    char buffer[32];
    size_t n = g.num_vertices;
    size_t m = g.num_edges;

    f.read((char*) buffer, string(MAGIC_WORD).size());
    f.read((char*) &n, sizeof(size_t));
    f.read((char*) &m, sizeof(size_t));

    if (!std::equal(buffer, buffer + string(MAGIC_WORD).size(), MAGIC_WORD)) {
        std::cerr << "invalid format for binary file: " << filename << std::endl;
        return false;
    }

    if (!f) {
        std::cerr << "error while reading file: " << filename << std::endl;
        return false;
    }

    g.num_vertices = n;
    g.num_edges = m;
    g.vertex_vals.resize(n);
    g.edge_src.resize(m);
    g.edge_dst.resize(m);
    g.edge_vals.resize(m);

    f.read((char*) g.vertex_vals.data(), sizeof(V) * n);
    f.read((char*) g.edge_src.data(), sizeof(vid_t) * m);
    f.read((char*) g.edge_dst.data(), sizeof(vid_t) * m);
    f.read((char*) g.edge_vals.data(), sizeof(E) * m);

    if (!f) {
        std::cerr << "error while reading file: " << filename << std::endl;
        return false;
    }

    log("read %d vertices and %d edges from %s", n, m, filename.c_str());
    return true;
}

template <typename V, typename E>
bool load_from_file(LocalGraph<V,E> &g, const string &filename) {
    ifstream f(filename.c_str());

    size_t total_bytes = get_file_size(filename);
    size_t read_bytes = 0;

    log("reading graph from file '%s'", filename.c_str());

    if (!f.is_open()) {
        std::cerr << "failed to open file: " << filename << std::endl;
        return false;
    }

    size_t lineno = 0;

    auto isdigit = [](char c) {
        return c >= '0' && c <= '9';
    };

    auto isspace = [](char c) {
        return c == ' ' || c == '\t' || c == '\r';
    };

    auto iscomment = [](char c) {
        return c == '#' || c == '*' || c == '-';
    };

    unordered_map<uint64_t, vid_t> vmap;
    string line;

    while (getline(f, line)) {
        const char *c = line.c_str();

        lineno++;
        read_bytes += line.size() + 1;

        if (lineno % 1000000 == 0) {
            log("read %d lines (%.2f%%)", lineno, double(read_bytes) / total_bytes * 100);
        }

        while (isspace(*c)) {
            c++;
        }

        if (*c == '\0' || iscomment(*c)) {
            continue;
        }

        if (!isdigit(*c)) {
            std::cerr << "invalid source vertex on line " << lineno << " of file " << filename << std::endl;
            return false;
        }

        uint64_t src = 0;
        while (isdigit(*c)) {
            src = src * 10 + (*c - '0');
            c++;
        }

        while (isspace(*c)) {
            c++;
        }

        if (!isdigit(*c)) {
            std::cerr << "invalid target vertex on line " << lineno << " of file " << filename << std::endl;
            return false;
        }

        uint64_t dst = 0;
        while (isdigit(*c)) {
            dst = dst * 10 + (*c - '0');
            c++;
        }

        while (isspace(*c)) {
            c++;
        }

        // TODO: check if EOL?
        //
        auto src_iter = vmap.find(src);
        auto dst_iter = vmap.find(dst);

        if (src_iter == vmap.end()) {
            src_iter = vmap.insert(make_pair(src, g.add_vertex((V) src))).first;
        }

        if (dst_iter == vmap.end()) {
            dst_iter = vmap.insert(make_pair(dst, g.add_vertex((V) dst))).first;
        }

        g.add_edge(src_iter->second, dst_iter->second);
    }

    log("done reading file, found %d vertices and %d edges", g.num_vertices, g.num_edges);

    return true;
}


}

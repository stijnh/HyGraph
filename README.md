# HyGraph: High-Performance Graph Processing on Hybrid CPU-GPU Platforms by Dynamic Load-Balancing

HyGraph [1] is a graph-processing system designed for hybrid platforms consisting of a multi-core CPU and a NVIDIA GPU. It solves the issue of workload imbalance in hybrid graph processing by replicating vertex state into both CPU and GPU memory and scheduling task onto both CPU and GPU in a dynamic fashion. This balances the workload of both devices, resulting in high performance. By overlapping communication and computation, the overhead of data transfers between CPU and GPU is hidden.

# Compile 
HyGraph requires the following packages (in parentheses are the recommended versions):
* NVIDIA CUDA Compiler (at least 7.0)
* Any C compiler (tested with gcc 4.8.2)
* GNU Make
* CMake

To compile, create a directory named build and run cmake in this directory.

```
mkdir build
cd build
cmake ..
```

# Usage
To run HyGraph, run `main`.

```
./main [graph file] [algorithm]
```

Where algorithm should be one of the following: `bfs` (Breadth-first search), `pr` (PageRank), `sssp` (Single-Source Shortest Path), `cc` (Connected components). The graph file should be in human-readable format containing one edge per line where each edge consists of a pair of two number. Duplicate edges, empty lines or comments (start with `#` or `*` or `-`) are ignored. Example of valid graph file:

```
1 2
2 3
3 5
1 5
```

To decrease graph loading time, it is possible to convert the graph into a binary format using the `convert` program.

```
./convert [text graph file] [binary graph file]
```

Where the binary graph file should be have `.bin` appended. For example, `./convert test.txt test.txt.bin`.



# License
This software is licensed under the GNU GPL v3.0.


# Bibliography
[1] Heldens S., Varbanescu A. L., Iosup A., HyGraph: Fast Graph Processing on Hybrid CPU-GPU Platforms by Dynamic Load-Balancing, 2016. Manuscript submitted for publication.


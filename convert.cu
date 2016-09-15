#include "local_graph.hpp"

using namespace std;
using namespace hygraph;

int main(int argc, char *argv[]) {
    LocalGraph<vid_t, empty_t> lg;

    if (argc < 3) {
        cerr << "usage: " << argv[0] << " <input file> <output file>" << endl;
        return EXIT_FAILURE;
    }

    string in_file = string(argv[1]);
    string out_file = string(argv[2]);

    if (!load_from_file(lg, in_file)) {
        cerr << "error: failed to read input file" << endl;
        return EXIT_FAILURE;
    }

    lg.sort_edges();
    lg.remove_duplicate_edges();

    if (!write_to_binary_file(lg, out_file)) {
        cerr << "error: failed to write output file" << endl;
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}

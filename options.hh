#include <getopt.h>
#include <cstddef>
#include <ctime>
#include <iostream>
#include <string>

struct TracerArgs {
    std::string outfile;
    bool output;
    size_t iterations;
};

TracerArgs parse_args(int argc, char *argv[]) {
    using namespace std;
    const static struct option longopts[] = {
        {"outfile", optional_argument, NULL, 'o'},
        {"iterations", optional_argument, NULL, 'n'},
        {"help", no_argument, NULL, 'h'}};

    TracerArgs opts = {"", false, 0};

    int longindex = 0;
    char flag = 0;
    while ((flag = getopt_long(argc, argv, "o:n:h", longopts, &longindex)) !=
           -1) {
        switch (flag) {
            case 'o':
                opts.output = true;
                opts.outfile = optarg;
                break;
            case 'n':
                opts.iterations = stoi(optarg);
                cout << "WARNING: iterations flag not yet implemented." << endl;
                break;
            case 'h':
            default:
                cout << "Usage: tracer [options...]" << endl;
                cout << " -o, --outfile    output the image to a bitmap file"
                     << endl;
                cout << " -n, --iterations run for a given number of iterations"
                     << endl;
                exit(-1);
        }
    }

    return opts;
}
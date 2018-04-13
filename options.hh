#include <getopt.h>
#include <cstddef>
#include <ctime>
#include <iostream>
#include <string>

struct TracerArgs {
    std::string outfile;
    bool output;
    int width;
    int height;
    size_t iterations;
};

TracerArgs parse_args(int argc, char *argv[]) {
    using namespace std;
    const static struct option longopts[] = {
        {"outfile", optional_argument, NULL, 'o'},
        {"iterations", optional_argument, NULL, 'n'},
        {"width", optional_argument, NULL, 'w'},
        {"height", optional_argument, NULL, 'h'},
        {"help", no_argument, NULL, '?'}};

    TracerArgs opts = {"", false, 640, 480, 0};

    int longindex = 0;
    char flag = 0;
    while ((flag = getopt_long(argc, argv, "o:n:h:w:", longopts, &longindex)) !=
           -1) {
        switch (flag) {
            case 'o':
                opts.output = true;
                opts.outfile = optarg;
                break;
            case 'n':
                opts.iterations = stoi(optarg);
                break;
            case 'w':
                opts.width = stoi(optarg);
                break;
            case 'h':
                opts.height = stoi(optarg);
                break;
            case '?':
            default:
                cout << "Usage: tracer [options...]" << endl;
                cout << " -o, --outfile    output the image to a bitmap file"
                     << endl;
                cout << " -n, --iterations run for a given number of iterations"
                     << endl;
                cout << " -w, --width of screen"
                     << endl;
                cout << " -h, --height of screen"
                     << endl;
                exit(-1);
        }
    }

    return opts;
}
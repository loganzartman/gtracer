#include <getopt.h>
#include <cstddef>
#include <ctime>
#include <iostream>
#include <sstream>
#include <string>
#include <thread>
#include "Vec3.hh"

const static std::string COLOR_RED = "\e[31m";
const static std::string COLOR_GREEN = "\e[32m";
const static std::string COLOR_BOLD = "\e[1m";
const static std::string COLOR_REVERSE = "\e[7m";
const static std::string COLOR_CLEAR = "\e[0m";

struct TracerArgs {
    std::string outfile;
    std::string infile;
    bool output;
    int width;
    int height;
    size_t iterations;
    unsigned threads;
    bool gpu;
    bool csv;
    bool time;
    bool accel;
    float scale;
    Float3 translate;
};

void print_usage() {
    using namespace std;
    cout << "Usage: tracer [options...] file.obj" << endl;
    cout << " -o, --outfile    output the image to a bitmap file" << endl;
    cout << " -n, --iterations run for a given number of iterations" << endl;
    cout << " -w, --width      width of screen" << endl;
    cout << " -h, --height     height of screen" << endl;
    cout << " -t, --threads    number of CPU threads (default autodetected)"
         << endl;
    cout << " -g, --gpu        use GPU acceleration" << endl;
    cout << " -c, --csv        output machine-readable CSV data" << endl;
    cout << " -j, --time       output timing data" << endl;
    cout << " -d, --daccel     disable acceleration structures used to speed "
         << "up path tracing" << endl;
    cout << " -S, --scale      scale factor for model (default 100)" << endl;
    cout << " -T, --translate  translate model coordinates (\"x y z\")" << endl;
}

void print_banner(const TracerArgs &opts) {
    using namespace std;
    cout << COLOR_GREEN << COLOR_REVERSE << " gtracer " << COLOR_CLEAR << endl;
    cout << COLOR_BOLD;
    if (opts.gpu)
        cout << COLOR_GREEN << "GPU Enabled";
    else
        cout << "CPU: " << opts.threads << " threads";
    cout << COLOR_CLEAR << endl;
}

TracerArgs parse_args(int argc, char *argv[]) {
    using namespace std;
    const static struct option longopts[] = {
        {"outfile", optional_argument, NULL, 'o'},
        {"iterations", optional_argument, NULL, 'n'},
        {"width", optional_argument, NULL, 'w'},
        {"height", optional_argument, NULL, 'h'},
        {"threads", optional_argument, NULL, 't'},
        {"gpu", no_argument, NULL, 'g'},
        {"csv", no_argument, NULL, 'c'},
        {"time", no_argument, NULL, 'j'},
        {"daccel", no_argument, NULL, 'd'},
        {"scale", optional_argument, NULL, 'S'},
        {"translate", optional_argument, NULL, 'T'},
        {"help", no_argument, NULL, '?'}};

    const unsigned thread_count = thread::hardware_concurrency();
    TracerArgs opts = {"", "",    false, 640,   480, 0,
                       0,  false, false, false, true, 100.f, Float3()};

    int longindex = 0;
    char flag = 0;
    while ((flag = getopt_long(argc, argv, "o:n:h:w:t:gcjdS:T:", longopts,
                               &longindex)) != -1) {
        stringstream s(optarg == nullptr ? "" : optarg);
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
            case 't':
                opts.threads = stoi(optarg);
                break;
            case 'g':
                opts.gpu = true;
                break;
            case 'c':
                opts.csv = true;
                break;
            case 'j':
                opts.time = true;
                break;
            case 'd':
                opts.accel = false;
                break;
            case 'S':
                opts.scale = stof(optarg);
                break;
            case 'T':
                float x, y, z;
                s >> x >> y >> z;
                opts.translate = Float3(x, y, z);
                break;
            case '?':
            default:
                print_usage();
                exit(-1);
        }
    }

    // automatic threads defaults
    if (opts.threads == 0) {
        if (!opts.gpu)
            opts.threads = thread_count;
    } else if (opts.gpu) {
        cout << COLOR_RED << "Warning: " << COLOR_CLEAR
             << "Specifying threads has no effect in GPU mode." << endl;
    }

    // read input filenames
    if (optind < argc) {
        opts.infile = argv[optind];
        optind++;
    } else {
        cout << COLOR_RED << "No input file specified." << COLOR_CLEAR << endl;
        print_usage();
        exit(-1);
    }

    if (!opts.csv)
        print_banner(opts);
    return opts;
}

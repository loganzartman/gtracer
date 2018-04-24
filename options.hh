#include <getopt.h>
#include <cstddef>
#include <ctime>
#include <iostream>
#include <string>
#include <thread>

struct TracerArgs {
    std::string outfile;
    std::string infile;
    bool output;
    int width;
    int height;
    size_t iterations;
    unsigned threads;
    bool gpu;
};

void print_usage() {
    using namespace std;
    cout << "Usage: tracer [options...] file.obj" << endl;
    cout << " -o, --outfile    output the image to a bitmap file"
         << endl;
    cout << " -n, --iterations run for a given number of iterations"
         << endl;
    cout << " -w, --width      width of screen" << endl;
    cout << " -h, --height     height of screen" << endl;
    cout << " -t, --threads    number of CPU threads" << endl;
    cout << " -g, --gpu        use GPU acceleration" << endl;
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
        {"help", no_argument, NULL, '?'}};

    const unsigned thread_count = thread::hardware_concurrency();
    TracerArgs opts = {"", "", false, 640, 480, 0, thread_count, false};

    int longindex = 0;
    char flag = 0;
    while ((flag = getopt_long(argc, argv, "o:n:h:w:t:g", longopts,
                               &longindex)) != -1) {
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
            case '?':
            default:
                print_usage();
                exit(-1);
        }
    }

    if(optind < argc){
        opts.infile = argv[optind];
        optind++;
    }else{
        cout << "No input file specified." << endl;
        print_usage();
        exit(-1);
    }

    return opts;
}
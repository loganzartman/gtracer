#include "util.hh"
#include <random>
#include "util.hh"

float randf(float lo, float hi) {
    using namespace std;
    static random_device rd;
    static mt19937 mt(rd());
    return (float)mt() / mt.max() * (hi - lo) + lo;
}

float mix(float a, float b, float ratio) { return a * ratio + b * (1 - ratio); }

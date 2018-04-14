#include "util.hh"
#include <random>

float randf(float lo, float hi) {
    using namespace std;
    static random_device rd;
    static mt19937 mt(rd());
    return (float)mt() / mt.max() * (hi - lo) + lo;
}
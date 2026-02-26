#include "utility.hpp"

#include <cmath>

namespace ahc061 {

const int DX[4] = {-1, 1, 0, 0};
const int DY[4] = {0, 0, -1, 1};

double clip(double x, double lo, double hi) {
    if(x < lo) return lo;
    if(x > hi) return hi;
    return x;
}

RNG::RNG(uint64_t seed) : state(seed), has_spare(false), spare(0.0) {
    if(state == 0) state = 202520252025ULL;
}

uint64_t RNG::next_u64() {
    uint64_t x = state;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    state = x;
    return x;
}

double RNG::uniform01() {
    constexpr int float_bits = 53;
    uint64_t value = next_u64() >> (64 - float_bits);
    return static_cast<double>(value) / static_cast<double>(1ULL << float_bits);
}

double RNG::uniform(double lo, double hi) {
    return lo + (hi - lo) * uniform01();
}

int RNG::randint(int lo, int hi) {
    if(lo > hi) std::swap(lo, hi);
    uint64_t span = static_cast<uint64_t>(hi - lo + 1);
    return lo + static_cast<int>(next_u64() % span);
}

double RNG::gauss(double mean, double sd) {
    if(sd <= 0.0) return mean;
    if(has_spare) {
        has_spare = false;
        return mean + sd * spare;
    }
    double u1 = uniform01();
    double u2 = uniform01();
    if(u1 < 1e-12) u1 = 1e-12;
    constexpr double kPi = 3.14159265358979323846;
    double r = std::sqrt(-2.0 * std::log(u1));
    double theta = 2.0 * kPi * u2;
    spare = r * std::sin(theta);
    has_spare = true;
    return mean + sd * (r * std::cos(theta));
}

TimeKeeper::TimeKeeper(double time_threshold_ms) : start_time_(std::chrono::steady_clock::now()), time_threshold_ms_(time_threshold_ms) {
}

double TimeKeeper::getElapsedTime() const {
    auto diff = std::chrono::steady_clock::now() - start_time_;
    return std::chrono::duration<double, std::milli>(diff).count();
}

bool TimeKeeper::isTimeOver() const {
    return getElapsedTime() >= time_threshold_ms_;
}

} // namespace ahc061

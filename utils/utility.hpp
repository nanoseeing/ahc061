#pragma once

#include <chrono>
#include <cstdint>

namespace ahc061 {

extern const int DX[4];
extern const int DY[4];

constexpr double W_LO = 0.3;
constexpr double W_HI = 1.0;
constexpr double E_LO = 0.1;
constexpr double E_HI = 0.5;

double clip(double x, double lo, double hi);

struct RNG {
    uint64_t state;
    bool has_spare;
    double spare;

    explicit RNG(uint64_t seed);

    uint64_t next_u64();
    double uniform01();
    double uniform(double lo, double hi);
    double gauss(double mean, double sd);
    int randint(int lo, int hi);
};

class TimeKeeper {
  private:
    std::chrono::steady_clock::time_point start_time_;
    double time_threshold_ms_;

  public:
    explicit TimeKeeper(double time_threshold_ms);

    double getElapsedTime() const;
    bool isTimeOver() const;
};

} // namespace ahc061

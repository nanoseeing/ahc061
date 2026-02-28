#pragma once

#include <cstdint>

namespace ahc061 {

struct XorShift64 {
    std::uint64_t x = 88172645463325252ULL;
    XorShift64() = default;
    explicit XorShift64(std::uint64_t seed) : x(seed ? seed : 88172645463325252ULL) {}

    std::uint64_t next_u64() {
        x ^= x << 7;
        x ^= x >> 9;
        return x;
    }

    int next_int(int lo, int hi) {  // inclusive
        const std::uint64_t span = static_cast<std::uint64_t>(hi - lo + 1);
        return lo + static_cast<int>(next_u64() % span);
    }

    double next_double01() {
        // [0, 1)
        constexpr double INV_U64 = 1.0 / 18446744073709551616.0;  // 2^64
        return static_cast<double>(next_u64()) * INV_U64;
    }
};

}  // namespace ahc061

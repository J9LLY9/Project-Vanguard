#pragma once

#include <cstddef>
#include <string>

namespace vanguard {

struct GpuProfile {
    std::string name;
    size_t total_vram;              // bytes
    std::string compute_capability; // e.g. "8.6"
    int mp_count;                   // streaming multiprocessor count
    int bus_width;                  // memory bus width, bits
    int memory_clock_khz;           // memory clock rate, kHz
    int max_threads_per_block;
};

class HardwareScanner {
public:
    HardwareScanner() = default;

    // Detects device 0 and fills the profile. Returns false if no
    // CUDA-capable GPU is present. Throws std::runtime_error on any
    // other CUDA Runtime API failure.
    bool scan();

    // Accessors below require a prior successful scan(); throws
    // std::runtime_error otherwise.
    const GpuProfile& profile() const;

    double theoretical_peak_bandwidth_gbps() const;

    void print_report() const;

private:
    GpuProfile profile_{};
    bool scanned_ = false;
};

} // namespace vanguard

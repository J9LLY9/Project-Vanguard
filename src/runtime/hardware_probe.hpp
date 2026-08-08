#pragma once

#include <cstddef>
#include <string>

namespace vanguard {

struct GpuProfile {
    std::string name;
    size_t total_vram = 0;              // bytes
    std::string compute_capability;     // e.g. "8.6"
    int mp_count = 0;                   // streaming multiprocessor count
    int bus_width = 0;                  // memory bus width, bits
    int memory_clock_khz = 0;           // memory clock rate, kHz
    int max_threads_per_block = 0;
};

class HardwareProbe {
public:
    HardwareProbe() = default;

    // Detects GPU device 0. Returns false if no CUDA-capable GPU is present.
    bool probe();

    const GpuProfile& profile() const;
    double theoretical_peak_bandwidth_gbps() const;
    void print_report() const;

private:
    GpuProfile profile_{};
    bool probed_ = false;
};

} // namespace vanguard

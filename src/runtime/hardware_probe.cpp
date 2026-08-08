#include "hardware_probe.hpp"

#include <cuda_runtime.h>
#include <cstdio>
#include <stdexcept>
#include <string>

namespace vanguard {

bool HardwareProbe::probe() {
    int device_count = 0;
    cudaError_t err = cudaGetDeviceCount(&device_count);

    if (err == cudaErrorNoDevice || device_count == 0) {
        probed_ = false;
        return false;
    }
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("HardwareProbe::probe cudaGetDeviceCount failed: ") +
                                 cudaGetErrorString(err));
    }

    cudaDeviceProp prop{};
    err = cudaGetDeviceProperties(&prop, 0);
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("HardwareProbe::probe cudaGetDeviceProperties failed: ") +
                                 cudaGetErrorString(err));
    }

    profile_.name = prop.name;
    profile_.total_vram = prop.totalGlobalMem;
    profile_.compute_capability = std::to_string(prop.major) + "." + std::to_string(prop.minor);
    profile_.mp_count = prop.multiProcessorCount;
    profile_.bus_width = prop.memoryBusWidth;
    profile_.memory_clock_khz = prop.memoryClockRate;
    profile_.max_threads_per_block = prop.maxThreadsPerBlock;

    probed_ = true;
    return true;
}

const GpuProfile& HardwareProbe::profile() const {
    if (!probed_) {
        throw std::runtime_error("HardwareProbe: profile() called before successful probe()");
    }
    return profile_;
}

double HardwareProbe::theoretical_peak_bandwidth_gbps() const {
    if (!probed_) {
        throw std::runtime_error("HardwareProbe: theoretical_peak_bandwidth_gbps() called before successful probe()");
    }
    double clock_hz = static_cast<double>(profile_.memory_clock_khz) * 1000.0;
    double bus_width_bytes = static_cast<double>(profile_.bus_width) / 8.0;
    // DDR multiplier = 2
    double bytes_per_sec = clock_hz * bus_width_bytes * 2.0;
    return bytes_per_sec / 1e9;
}

void HardwareProbe::print_report() const {
    if (!probed_) {
        std::printf("[HardwareProbe] No CUDA-capable GPU detected.\n");
        return;
    }
    std::printf("====================================================\n");
    std::printf(" Vanguard GPU Telemetry Report                      \n");
    std::printf("====================================================\n");
    std::printf(" GPU Name:                  %s\n", profile_.name.c_str());
    std::printf(" Compute Capability:        %s\n", profile_.compute_capability.c_str());
    std::printf(" Total VRAM:                %.2f GB (%zu bytes)\n",
                static_cast<double>(profile_.total_vram) / (1024.0 * 1024.0 * 1024.0),
                profile_.total_vram);
    std::printf(" Streaming Multiprocessors: %d SMs\n", profile_.mp_count);
    std::printf(" Memory Bus Width:          %d bits\n", profile_.bus_width);
    std::printf(" Memory Clock Rate:         %.2f GHz\n",
                static_cast<double>(profile_.memory_clock_khz) / 1e6);
    std::printf(" Peak Memory Bandwidth:     %.2f GB/s\n", theoretical_peak_bandwidth_gbps());
    std::printf(" Max Threads / Block:       %d\n", profile_.max_threads_per_block);
    std::printf("====================================================\n\n");
}

} // namespace vanguard

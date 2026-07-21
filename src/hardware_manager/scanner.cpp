#include "scanner.hpp"

#include <cuda_runtime.h>

#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>

namespace vanguard {

namespace {

void cuda_check(cudaError_t err, const char* what) {
    if (err != cudaSuccess) {
        std::ostringstream oss;
        oss << "CUDA error during " << what << ": " << cudaGetErrorString(err);
        throw std::runtime_error(oss.str());
    }
}

} // namespace

bool HardwareScanner::scan() {
    int device_count = 0;
    cuda_check(cudaGetDeviceCount(&device_count), "cudaGetDeviceCount");

    if (device_count == 0) {
        return false;
    }

    // Phase 1 targets single-GPU detection; multi-GPU handling is
    // deferred, so we always profile device 0.
    const int device_index = 0;
    cudaDeviceProp props{};
    cuda_check(cudaGetDeviceProperties(&props, device_index), "cudaGetDeviceProperties");

    profile_.name = props.name;
    profile_.total_vram = props.totalGlobalMem;

    std::ostringstream cc;
    cc << props.major << '.' << props.minor;
    profile_.compute_capability = cc.str();

    profile_.mp_count = props.multiProcessorCount;
    profile_.bus_width = props.memoryBusWidth;
    profile_.memory_clock_khz = props.memoryClockRate;
    profile_.max_threads_per_block = props.maxThreadsPerBlock;

    scanned_ = true;
    return true;
}

const GpuProfile& HardwareScanner::profile() const {
    if (!scanned_) {
        throw std::runtime_error("HardwareScanner::profile() called before a successful scan()");
    }
    return profile_;
}

double HardwareScanner::theoretical_peak_bandwidth_gbps() const {
    if (!scanned_) {
        throw std::runtime_error(
            "HardwareScanner::theoretical_peak_bandwidth_gbps() called before a successful scan()");
    }

    const double memory_clock_hz = static_cast<double>(profile_.memory_clock_khz) * 1000.0;
    const double bus_width_bytes = static_cast<double>(profile_.bus_width) / 8.0;

    // DDR memory transfers on both clock edges, hence the factor of 2.
    return (memory_clock_hz * 2.0 * bus_width_bytes) / 1e9;
}

void HardwareScanner::print_report() const {
    if (!scanned_) {
        throw std::runtime_error("HardwareScanner::print_report() called before a successful scan()");
    }

    const double vram_gb = static_cast<double>(profile_.total_vram) / (1024.0 * 1024.0 * 1024.0);
    const double bandwidth_gbps = theoretical_peak_bandwidth_gbps();

    std::ostringstream vram_str;
    vram_str << std::fixed << std::setprecision(2) << vram_gb << " GB";

    std::ostringstream bandwidth_str;
    bandwidth_str << std::fixed << std::setprecision(2) << bandwidth_gbps << " GB/s";

    constexpr int label_width = 30;
    constexpr int value_width = 22;
    const std::string rule(static_cast<size_t>(label_width + value_width + 3), '-');

    auto row = [&](const std::string& label, const std::string& value) {
        std::cout << " " << std::left << std::setw(label_width) << label << std::right
                   << std::setw(value_width) << value << " \n";
    };

    std::cout << rule << '\n';
    std::cout << " PROJECT VANGUARD :: HARDWARE MANAGER\n";
    std::cout << rule << '\n';

    row("GPU Name", profile_.name);
    row("Compute Capability", profile_.compute_capability);
    row("Streaming Multiprocessors", std::to_string(profile_.mp_count));
    row("Total VRAM", vram_str.str());
    row("Memory Bus Width", std::to_string(profile_.bus_width) + " bit");
    row("Memory Clock Rate", std::to_string(profile_.memory_clock_khz / 1000) + " MHz");
    row("Max Threads / Block", std::to_string(profile_.max_threads_per_block));
    row("Theoretical Peak Bandwidth", bandwidth_str.str());

    std::cout << rule << '\n';
}

} // namespace vanguard

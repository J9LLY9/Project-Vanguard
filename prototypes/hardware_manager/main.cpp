#include "scanner.hpp"

#include <exception>
#include <iostream>

int main() {
    vanguard::HardwareScanner scanner;

    try {
        if (!scanner.scan()) {
            std::cerr << "No CUDA-capable GPU detected.\n";
            return 1;
        }
    } catch (const std::exception& e) {
        std::cerr << "Hardware scan failed: " << e.what() << '\n';
        return 1;
    }

    scanner.print_report();
    return 0;
}

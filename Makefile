CXX = g++
NVCC = nvcc
CXXFLAGS = -O3 -Wall -Wextra -std=c++17 -Isrc
NVCCFLAGS = -O3 -std=c++17 -Isrc -arch=sm_86 -DVANGUARD_WITH_CUDA

BUILD_DIR = build
BIN_DIR = bin

.PHONY: all clean test cpu_test cuda_test cli_app

all:
	@echo "No build targets defined yet. Targets will be added as implementation is rebuilt."

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

$(BIN_DIR):
	mkdir -p $(BIN_DIR)

clean:
	rm -rf $(BUILD_DIR) $(BIN_DIR)

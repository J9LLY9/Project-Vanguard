CXX = g++
NVCC = nvcc
CXXFLAGS = -O3 -Wall -Wextra -std=c++17 -Isrc
NVCCFLAGS = -O3 -std=c++17 -Isrc -arch=sm_86 -DVANGUARD_WITH_CUDA

BUILD_DIR = build
BIN_DIR = bin

CPU_SRCS = src/core/model/gpt2_weights.cpp \
           src/core/model/gpt2_tokenizer.cpp \
           src/backends/cpu/cpu_kernels.cpp \
           src/backends/cpu/cpu_engine.cpp

CUDA_SRCS = src/backends/cuda/kv_cache.cu \
            src/backends/cuda/cuda_engine.cu \
            src/runtime/hardware_probe.cpp

.PHONY: all clean test cpu_test cuda_test cli_app

all: $(BIN_DIR)/test_cpu_correctness $(BIN_DIR)/test_cuda_correctness $(BIN_DIR)/vanguard_cli

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

$(BIN_DIR):
	mkdir -p $(BIN_DIR)

# CPU Test Target
$(BIN_DIR)/test_cpu_correctness: tests/integration/test_cpu_correctness.cpp $(CPU_SRCS) | $(BIN_DIR)
	$(CXX) $(CXXFLAGS) $^ -o $@

# CUDA Test Target
$(BIN_DIR)/test_cuda_correctness: tests/integration/test_cuda_correctness.cpp $(CPU_SRCS) $(CUDA_SRCS) | $(BIN_DIR)
	$(NVCC) $(NVCCFLAGS) -lcuda -lcudart $^ -o $@

# CLI App Target
$(BIN_DIR)/vanguard_cli: apps/cli/main.cpp $(CPU_SRCS) $(CUDA_SRCS) | $(BIN_DIR)
	$(NVCC) $(NVCCFLAGS) -lcuda -lcudart $^ -o $@

cpu_test: $(BIN_DIR)/test_cpu_correctness
	./$(BIN_DIR)/test_cpu_correctness gpt2_124M.bin vocab.json

cuda_test: $(BIN_DIR)/test_cuda_correctness
	./$(BIN_DIR)/test_cuda_correctness gpt2_124M.bin vocab.json

cli_app: $(BIN_DIR)/vanguard_cli

test: cpu_test cuda_test

clean:
	rm -rf $(BUILD_DIR) $(BIN_DIR)

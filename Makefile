CXX = g++
NVCC = nvcc
CXXFLAGS = -O3 -Wall -Wextra -std=c++17 -Isrc
NVCCFLAGS = -O3 -std=c++17 -Isrc -arch=sm_80

BUILD_DIR = build
BIN_DIR = bin

CPU_SRCS = src/model/gpt2_weights.cpp \
           src/tokenizer/gpt2_tokenizer.cpp \
           src/cpu/cpu_kernels.cpp \
           src/cpu/cpu_engine.cpp

CUDA_SRCS = src/cuda/kv_cache.cu \
            src/cuda/cuda_engine.cu \
            src/runtime/hardware_probe.cpp

.PHONY: all clean test cpu_test cuda_test

all: $(BIN_DIR)/test_cpu_correctness $(BIN_DIR)/test_cuda_correctness

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

$(BIN_DIR):
	mkdir -p $(BIN_DIR)

# CPU Test Target
$(BIN_DIR)/test_cpu_correctness: tests/test_cpu_correctness.cpp $(CPU_SRCS) | $(BIN_DIR)
	$(CXX) $(CXXFLAGS) $^ -o $@

# CUDA Test Target
$(BIN_DIR)/test_cuda_correctness: tests/test_cuda_correctness.cpp $(CPU_SRCS) $(CUDA_SRCS) | $(BIN_DIR)
	$(NVCC) $(NVCCFLAGS) -lcuda -lcudart $^ -o $@

cpu_test: $(BIN_DIR)/test_cpu_correctness
	./$(BIN_DIR)/test_cpu_correctness gpt2_124M.bin vocab.json

cuda_test: $(BIN_DIR)/test_cuda_correctness
	./$(BIN_DIR)/test_cuda_correctness gpt2_124M.bin vocab.json

test: cpu_test cuda_test

clean:
	rm -rf $(BUILD_DIR) $(BIN_DIR)

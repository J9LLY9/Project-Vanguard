NVCC ?= nvcc
NVCCFLAGS := -std=c++17 -O2

TARGET := gpt2_engine
SRCS := gpt2_engine.cu main.cu
HEADERS := gpt2_engine.hpp

.PHONY: all clean run

all: $(TARGET)

$(TARGET): $(SRCS) $(HEADERS)
	$(NVCC) $(NVCCFLAGS) -o $(TARGET) $(SRCS)

run: $(TARGET)
	./$(TARGET) gpt2_124M.bin

clean:
	rm -f $(TARGET)

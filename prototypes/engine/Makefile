NVCC ?= nvcc
NVCCFLAGS := -std=c++17 -O2

TARGET := gpt2_engine
SRCS := gpt2_engine.cu main.cu gpt2_tokenizer.cpp
HEADERS := gpt2_engine.hpp gpt2_tokenizer.hpp

.PHONY: all clean run

all: $(TARGET)

$(TARGET): $(SRCS) $(HEADERS)
	$(NVCC) $(NVCCFLAGS) -o $(TARGET) $(SRCS)

run: $(TARGET)
	./$(TARGET) gpt2_124M.bin

clean:
	rm -f $(TARGET)

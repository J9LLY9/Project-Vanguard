NVCC ?= nvcc
CXXFLAGS := -std=c++17 -O2 -Xcompiler -Wall,-Wextra
TARGET := hardware_manager
SRCS := scanner.cpp main.cpp

.PHONY: all clean run

all: $(TARGET)

$(TARGET): $(SRCS) scanner.hpp
	$(NVCC) $(CXXFLAGS) -o $(TARGET) $(SRCS)

run: all
	./$(TARGET)

clean:
	rm -f $(TARGET)

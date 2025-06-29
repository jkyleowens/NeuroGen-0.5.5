# Compiler
CXX=clang++
NVCC=/opt/cuda/bin/nvcc

# Directories
SRC_DIR=src
OBJ_DIR=obj
INC_DIR=include
CUDA_SRC_DIR=$(SRC_DIR)/cuda
CUDA_OBJ_DIR=$(OBJ_DIR)/cuda

# Flags
CXXFLAGS=-std=c++17 -Wall -Wextra -I$(INC_DIR) -I/opt/cuda/include -I/usr/include/jsoncpp -O2 -g -MMD -MP
NVCCFLAGS=-std=c++17 -I$(INC_DIR) -O2 -Xcompiler -fPIC
LDFLAGS=-L/opt/cuda/lib64 -L/usr/lib -ljsoncpp -lcudart

# Source files (finds all .cpp and .cu files automatically)
SOURCES := $(wildcard $(SRC_DIR)/*.cpp)
SOURCES += $(wildcard $(CUDA_SRC_DIR)/*.cpp)
CUDA_SOURCES := $(wildcard $(CUDA_SRC_DIR)/*.cu)

# Object files
OBJECTS := $(patsubst $(SRC_DIR)/%.cpp,$(OBJ_DIR)/%.o,$(filter %.cpp,$(SOURCES)))
OBJECTS += $(patsubst $(CUDA_SRC_DIR)/%.cpp,$(CUDA_OBJ_DIR)/%.o,$(filter %.cpp,$(SOURCES)))
CUDA_OBJECTS := $(patsubst $(CUDA_SRC_DIR)/%.cu,$(CUDA_OBJ_DIR)/%.o,$(CUDA_SOURCES))

# Target
TARGET=NeuroGen

.PHONY: all clean mirror_headers

# Main rule
all: $(TARGET)

# Link the final executable
$(TARGET): mirror_headers $(OBJECTS) $(CUDA_OBJECTS)
	@echo "Linking..."
	$(CXX) -o $@ $^ $(LDFLAGS)

# Rule to mirror headers before compilation using standard shell commands
mirror_headers:
	@echo "Mirroring source headers to 'include/NeuroGen'..."
	@mkdir -p $(INC_DIR)/NeuroGen
	@cd $(SRC_DIR) && for file in $$(find . -name '*.h' -o -name '*.cuh'); do \
		target_dir="../$(INC_DIR)/NeuroGen/$$(dirname $$file)"; \
		mkdir -p $$target_dir; \
		cp $$file $$target_dir/; \
	done

# Rule for compiling .cpp files
$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cpp
	@mkdir -p $(OBJ_DIR)
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Rule for compiling .cpp files in the cuda directory
$(CUDA_OBJ_DIR)/%.o: $(CUDA_SRC_DIR)/%.cpp
	@mkdir -p $(CUDA_OBJ_DIR)
	$(CXX) $(CXXFLAGS) -x cu -c $< -o $@

# Rule for compiling .cu files
$(CUDA_OBJ_DIR)/%.o: $(CUDA_SRC_DIR)/%.cu
	@mkdir -p $(CUDA_OBJ_DIR)
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

# Clean rule
clean:
	@echo "Cleaning up..."
	rm -rf $(OBJ_DIR) $(TARGET) $(INC_DIR)/NeuroGen
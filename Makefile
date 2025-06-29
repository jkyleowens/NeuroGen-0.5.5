# NeuroGen Alpha - Biologically Inspired Modular Neural Network
# Makefile for CUDA-accelerated brain simulation

# =============================================================================
# COMPILER CONFIGURATION
# =============================================================================
CXX = clang++
NVCC = /opt/cuda/bin/nvcc

# Check for alternative CUDA installations
ifeq ($(wildcard /usr/local/cuda/bin/nvcc),)
    ifeq ($(wildcard /opt/cuda/bin/nvcc),)
        $(error CUDA not found. Please install CUDA or set CUDA_PATH)
    endif
endif

# =============================================================================
# DIRECTORIES
# =============================================================================
SRC_DIR = src
OBJ_DIR = obj
INC_DIR = include
CUDA_SRC_DIR = $(SRC_DIR)/cuda
CUDA_OBJ_DIR = $(OBJ_DIR)/cuda
BUILD_DIR = build
DEP_DIR = $(BUILD_DIR)/deps

# =============================================================================
# GPU ARCHITECTURE DETECTION AND CONFIGURATION
# =============================================================================
# Auto-detect GPU architecture or use default
GPU_ARCH ?= sm_75
CUDA_ARCHS = -gencode arch=compute_75,code=sm_75 \
             -gencode arch=compute_80,code=sm_80 \
             -gencode arch=compute_86,code=sm_86 \
             -gencode arch=compute_89,code=sm_89

# Enable for broader compatibility
ifdef MULTI_GPU_SUPPORT
    CUDA_ARCH_FLAGS = $(CUDA_ARCHS)
else
    CUDA_ARCH_FLAGS = -arch=$(GPU_ARCH)
endif

# =============================================================================
# COMPILER FLAGS
# =============================================================================
# C++ compilation flags
CXXFLAGS = -std=c++17 \
           -Wall -Wextra -Wpedantic \
           -I$(INC_DIR) \
           -I/opt/cuda/include \
           -I/usr/include/jsoncpp \
           -O3 -g -MMD -MP \
           -march=native \
           -ffast-math

# CUDA compilation flags  
NVCCFLAGS = -std=c++17 \
            -I$(INC_DIR) \
            $(CUDA_ARCH_FLAGS) \
            -O3 -g -lineinfo \
            -Xcompiler -fPIC \
            -Xcompiler -Wall \
            -use_fast_math \
            --expt-relaxed-constexpr \
            --expt-extended-lambda

# Linker flags
LDFLAGS = -L/opt/cuda/lib64 \
          -L/usr/lib \
          -ljsoncpp \
          -lcudart \
          -lcurand \
          -lcublas \
          -lcufft

# Debug flags
ifdef DEBUG
    CXXFLAGS += -DDEBUG -O0 -fsanitize=address
    NVCCFLAGS += -DDEBUG -O0 -G
    LDFLAGS += -fsanitize=address
endif

# Release optimizations
ifdef RELEASE
    CXXFLAGS += -DNDEBUG -O3 -flto
    NVCCFLAGS += -DNDEBUG -O3
    LDFLAGS += -flto
endif

# =============================================================================
# SOURCE FILE DISCOVERY
# =============================================================================
# Find all source files
CPP_SOURCES := $(shell find $(SRC_DIR) -name '*.cpp' -not -path "$(CUDA_SRC_DIR)/*")
CUDA_CPP_SOURCES := $(wildcard $(CUDA_SRC_DIR)/*.cpp)
CUDA_SOURCES := $(wildcard $(CUDA_SRC_DIR)/*.cu)

# Generate object file paths
CPP_OBJECTS := $(patsubst $(SRC_DIR)/%.cpp,$(OBJ_DIR)/%.o,$(CPP_SOURCES))
CUDA_CPP_OBJECTS := $(patsubst $(CUDA_SRC_DIR)/%.cpp,$(CUDA_OBJ_DIR)/%.o,$(CUDA_CPP_SOURCES))
CUDA_OBJECTS := $(patsubst $(CUDA_SRC_DIR)/%.cu,$(CUDA_OBJ_DIR)/%.o,$(CUDA_SOURCES))

# All objects
ALL_OBJECTS := $(CPP_OBJECTS) $(CUDA_CPP_OBJECTS) $(CUDA_OBJECTS)

# Dependency files
DEP_FILES := $(ALL_OBJECTS:.o=.d)

# =============================================================================
# TARGETS
# =============================================================================
TARGET = NeuroGen
TEST_TARGET = NeuroGen_test

.PHONY: all clean test install uninstall help debug release format check-gpu mirror_headers

# Default target
all: check-gpu $(TARGET)

# =============================================================================
# GPU COMPATIBILITY CHECK
# =============================================================================
check-gpu:
	@echo "Checking GPU compatibility..."
	@if command -v nvidia-smi >/dev/null 2>&1; then \
		echo "NVIDIA GPU detected:"; \
		nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader,nounits | head -1; \
	else \
		echo "Warning: nvidia-smi not found. Assuming GPU compatibility."; \
	fi
	@echo "Using GPU architecture: $(GPU_ARCH)"

# =============================================================================
# MAIN BUILD RULES
# =============================================================================
# Link the final executable
$(TARGET): mirror_headers $(ALL_OBJECTS)
	@echo "Linking $(TARGET)..."
	@mkdir -p $(BUILD_DIR)
	$(CXX) -o $@ $(ALL_OBJECTS) $(LDFLAGS)
	@echo "Build complete: $(TARGET)"

# =============================================================================
# HEADER MIRRORING FOR MODULAR ARCHITECTURE
# =============================================================================
mirror_headers:
	@echo "Mirroring source headers to 'include/NeuroGen'..."
	@mkdir -p $(INC_DIR)/NeuroGen
	@cd $(SRC_DIR) && for file in $$(find . -name '*.h' -o -name '*.cuh'); do \
		target_dir="../$(INC_DIR)/NeuroGen/$$(dirname $$file)"; \
		mkdir -p "$$target_dir"; \
		cp "$$file" "$$target_dir/"; \
	done
	@echo "Header mirroring complete."

# =============================================================================
# COMPILATION RULES
# =============================================================================
# Rule for compiling regular C++ files
$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cpp
	@echo "Compiling C++ source: $<"
	@mkdir -p $(dir $@)
	@mkdir -p $(DEP_DIR)/$(dir $*)
	$(CXX) $(CXXFLAGS) -MF $(DEP_DIR)/$*.d -c $< -o $@

# Rule for compiling C++ files in CUDA directory (as regular C++)
$(CUDA_OBJ_DIR)/%.o: $(CUDA_SRC_DIR)/%.cpp
	@echo "Compiling CUDA C++ source: $<"
	@mkdir -p $(dir $@)
	@mkdir -p $(DEP_DIR)/cuda/$(dir $*)
	$(CXX) $(CXXFLAGS) -MF $(DEP_DIR)/cuda/$*.d -c $< -o $@

# Rule for compiling CUDA files
$(CUDA_OBJ_DIR)/%.o: $(CUDA_SRC_DIR)/%.cu
	@echo "Compiling CUDA source: $<"
	@mkdir -p $(dir $@)
	@mkdir -p $(DEP_DIR)/cuda/$(dir $*)
	$(NVCC) $(NVCCFLAGS) -M $< -MT $@ -MF $(DEP_DIR)/cuda/$*.d
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

# =============================================================================
# TESTING AND VALIDATION
# =============================================================================
test: $(TEST_TARGET)
	@echo "Running neural network tests..."
	./$(TEST_TARGET)

$(TEST_TARGET): mirror_headers $(ALL_OBJECTS)
	@echo "Building test executable..."
	$(CXX) -DTESTING -o $@ $(ALL_OBJECTS) $(LDFLAGS)

# =============================================================================
# DEVELOPMENT TOOLS
# =============================================================================
# Format code using clang-format
format:
	@echo "Formatting source code..."
	@find $(SRC_DIR) -name "*.cpp" -o -name "*.h" -o -name "*.cu" -o -name "*.cuh" | \
		xargs clang-format -i -style="{BasedOnStyle: Google, IndentWidth: 4, ColumnLimit: 100}"

# Static analysis
analyze:
	@echo "Running static analysis..."
	@clang-tidy $(CPP_SOURCES) $(CUDA_CPP_SOURCES) -- $(CXXFLAGS)

# Profile GPU kernels
profile: $(TARGET)
	@echo "Profiling GPU kernels..."
	nvprof --analysis-metrics -o profile.nvvp ./$(TARGET)

# =============================================================================
# BUILD VARIANTS
# =============================================================================
debug:
	$(MAKE) DEBUG=1 all

release:
	$(MAKE) RELEASE=1 all

# Multi-GPU support
multi-gpu:
	$(MAKE) MULTI_GPU_SUPPORT=1 all

# =============================================================================
# INSTALLATION
# =============================================================================
PREFIX ?= /usr/local
BINDIR = $(PREFIX)/bin
INCDIR = $(PREFIX)/include

install: $(TARGET)
	@echo "Installing NeuroGen..."
	install -d $(BINDIR)
	install -m 755 $(TARGET) $(BINDIR)
	install -d $(INCDIR)/NeuroGen
	cp -r $(INC_DIR)/NeuroGen/* $(INCDIR)/NeuroGen/

uninstall:
	@echo "Uninstalling NeuroGen..."
	rm -f $(BINDIR)/$(TARGET)
	rm -rf $(INCDIR)/NeuroGen

# =============================================================================
# CLEANUP
# =============================================================================
clean:
	@echo "Cleaning build artifacts..."
	rm -rf $(OBJ_DIR) $(BUILD_DIR) $(TARGET) $(TEST_TARGET)
	rm -rf $(INC_DIR)/NeuroGen
	rm -f *.nvvp profile.*
	@echo "Clean complete."

# Deep clean including generated files
distclean: clean
	@echo "Deep cleaning..."
	rm -f tags cscope.out
	find . -name "*.orig" -delete
	find . -name "*.rej" -delete

# =============================================================================
# HELP
# =============================================================================
help:
	@echo "NeuroGen Alpha - Modular Neural Network Build System"
	@echo ""
	@echo "Available targets:"
	@echo "  all          - Build the main executable (default)"
	@echo "  debug        - Build with debug flags"
	@echo "  release      - Build optimized release version"
	@echo "  multi-gpu    - Build with multi-GPU support"
	@echo "  test         - Build and run tests"
	@echo "  clean        - Remove build artifacts"
	@echo "  distclean    - Deep clean including generated files"
	@echo "  format       - Format source code"
	@echo "  analyze      - Run static analysis"
	@echo "  profile      - Profile GPU kernels"
	@echo "  install      - Install to system (PREFIX=$(PREFIX))"
	@echo "  uninstall    - Remove from system"
	@echo "  check-gpu    - Check GPU compatibility"
	@echo "  help         - Show this help message"
	@echo ""
	@echo "Build options:"
	@echo "  DEBUG=1      - Enable debug mode"
	@echo "  RELEASE=1    - Enable release optimizations"
	@echo "  GPU_ARCH=sm_XX - Set specific GPU architecture"
	@echo "  MULTI_GPU_SUPPORT=1 - Enable multiple GPU architectures"

# =============================================================================
# DEPENDENCY INCLUSION
# =============================================================================
# Include dependency files if they exist
-include $(DEP_FILES)

# =============================================================================
# BUILD INFO
# =============================================================================
.PHONY: info
info:
	@echo "NeuroGen Build Configuration:"
	@echo "  CXX: $(CXX)"
	@echo "  NVCC: $(NVCC)"
	@echo "  GPU Architecture: $(GPU_ARCH)"
	@echo "  C++ Sources: $(words $(CPP_SOURCES)) files"
	@echo "  CUDA C++ Sources: $(words $(CUDA_CPP_SOURCES)) files"
	@echo "  CUDA Sources: $(words $(CUDA_SOURCES)) files"
	@echo "  Total Objects: $(words $(ALL_OBJECTS)) files"
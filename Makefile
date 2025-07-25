# ============================================================================
# NEUROGEN MODULAR NEURAL NETWORK - FIXED MAKEFILE
# ============================================================================

# Compiler settings
CXX = clang++
NVCC = nvcc
CXXFLAGS = -std=c++17 -O3 -Wall -Wextra -fPIC -I./include -ferror-limit=50
NVCCFLAGS = -std=c++17 -O3 -Xcompiler -fPIC -I./include

# Library paths and links - FIXED X11 LINKING
CUDA_PATH = /opt/cuda
LDFLAGS = -L$(CUDA_PATH)/lib64 -L/usr/lib -L/usr/lib/x86_64-linux-gnu
LIBS = -ljsoncpp -lcudart -lcurand -lcublas -lcufft

# Source directories
SRC_DIR = src
CUDA_SRC_DIR = src/cuda
OBJ_DIR = obj

# Create object directory
$(shell mkdir -p $(OBJ_DIR))

# C++ source files - EXCLUDE CONFLICTING SOURCES
CPP_SOURCES = $(filter-out $(SRC_DIR)/main.cpp $(SRC_DIR)/NlpAgentImplementation.cpp $(SRC_DIR)/InputController.cpp, $(wildcard $(SRC_DIR)/*.cpp))

# CUDA source files
CUDA_SOURCES = $(wildcard $(CUDA_SRC_DIR)/*.cu)

# Object files
CPP_OBJECTS = $(CPP_SOURCES:$(SRC_DIR)/%.cpp=$(OBJ_DIR)/%.o)
CUDA_OBJECTS = $(CUDA_SOURCES:$(CUDA_SRC_DIR)/%.cu=$(OBJ_DIR)/%.o)

# Final targets
TARGET = NeuroGen
AUTONOMOUS_TARGET = NeuroGenAutonomous

# Default target
all: $(AUTONOMOUS_TARGET)

# Build autonomous version only (main_autonomous.cpp)
$(AUTONOMOUS_TARGET): $(CPP_OBJECTS) $(CUDA_OBJECTS) $(OBJ_DIR)/main_autonomous.o
	@echo "Linking $(AUTONOMOUS_TARGET)..."
	$(CXX) -o $@ $^ $(LDFLAGS) $(LIBS)
	@echo "✅ Build complete: $(AUTONOMOUS_TARGET)"

# Original main (if needed separately)
$(TARGET): $(CPP_OBJECTS) $(CUDA_OBJECTS) $(OBJ_DIR)/main.o
	@echo "Linking $(TARGET)..."
	$(CXX) -o $@ $^ $(LDFLAGS) $(LIBS)
	@echo "✅ Build complete: $(TARGET)"

# Compile C++ source files
$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cpp
	@echo "Compiling $<..."
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Compile CUDA source files
$(OBJ_DIR)/%.o: $(CUDA_SRC_DIR)/%.cu
	@echo "Compiling CUDA $<..."
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

# Clean build artifacts
clean:
	rm -rf $(OBJ_DIR)
	rm -f $(TARGET) $(AUTONOMOUS_TARGET)

# Install dependencies (Ubuntu/Debian)
deps:
	sudo apt-get update
	sudo apt-get install -y libx11-dev libxtst-dev libxi-dev libjsoncpp-dev

# Debug build
debug: CXXFLAGS += -g -DDEBUG
debug: NVCCFLAGS += -g -DDEBUG
debug: $(AUTONOMOUS_TARGET)

.PHONY: all clean deps debug
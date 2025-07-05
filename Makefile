# Compiler and Linker
CXX := clang++
NVCC := /opt/cuda/bin/nvcc
LINK := clang++

# Directories
SRC_DIR := src
OBJ_DIR := obj
BUILD_DIR := build
INCLUDE_DIR := include
CUDA_SRC_DIR := $(SRC_DIR)/cuda_disabled
CUDA_OBJ_DIR := $(OBJ_DIR)/cuda
DEPS_DIR := $(BUILD_DIR)/deps
CUDA_DEPS_DIR := $(DEPS_DIR)/cuda

# CUDA Path
CUDA_PATH := /opt/cuda

# Executable Name
TARGET := NeuroGen
TARGET_AUTONOMOUS := NeuroGen_Autonomous

# Compiler Flags
# Note: The -I$(INCLUDE_DIR) flag tells the compilers where to find your header files.
CXXFLAGS := -std=c++17 -I$(INCLUDE_DIR) -I$(CUDA_PATH)/include -I/usr/include/opencv4 -O3 -g -fPIC -Wall
NVCCFLAGS := -std=c++17 -I$(INCLUDE_DIR) -I$(CUDA_PATH)/include -arch=sm_75 -O3 -g -lineinfo \
             -Xcompiler -fPIC -Xcompiler -Wall -use_fast_math \
             --expt-relaxed-constexpr --expt-extended-lambda -ccbin /usr/bin/clang++

# Linker Flags
LDFLAGS := -L/usr/lib
LDLIBS := -ljsoncpp \
          -lX11 -lXext -lXfixes -lXtst \
          -lopencv_core -lopencv_imgproc -lopencv_imgcodecs -lopencv_objdetect \
          -ltesseract -lleptonica

# --- Source Files ---

# Automatically find all .cpp and .cu files
CPP_SOURCES := $(wildcard $(SRC_DIR)/*.cpp)
CUDA_SOURCES :=

# --- Object Files ---

# Generate object file names from source file names
CPP_OBJECTS := $(patsubst $(SRC_DIR)/%.cpp,$(OBJ_DIR)/%.o,$(CPP_SOURCES))
CUDA_OBJECTS := $(patsubst $(CUDA_SRC_DIR)/%.cu,$(CUDA_OBJ_DIR)/%.o,$(CUDA_SOURCES))

# Combine all object files
OBJECTS := $(CPP_OBJECTS) $(CUDA_OBJECTS)

# --- Dependency Files ---
DEPS := $(patsubst $(SRC_DIR)/%.cpp,$(DEPS_DIR)/%.d,$(CPP_SOURCES))
CUDA_DEPS := $(patsubst $(CUDA_SRC_DIR)/%.cu,$(CUDA_DEPS_DIR)/%.d,$(CUDA_SOURCES))

# --- Build Rules ---

all: $(TARGET)

autonomous: $(TARGET_AUTONOMOUS)

# Linking the final executable
$(TARGET): $(OBJECTS)
	@echo "Linking $(TARGET)..."
	$(LINK) -o $@ $^ $(LDFLAGS) $(LDLIBS)

# Linking the autonomous learning executable
$(TARGET_AUTONOMOUS): $(filter-out $(OBJ_DIR)/main.o,$(OBJECTS)) $(OBJ_DIR)/main_autonomous.o
	@echo "Linking $(TARGET_AUTONOMOUS)..."
	$(LINK) -o $@ $^ $(LDFLAGS) $(LDLIBS)

# C++ compilation rule
$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cpp | $(OBJ_DIR) $(DEPS_DIR)
	@echo "Compiling C++ source: $<"
	$(CXX) $(CXXFLAGS) -c $< -o $@
	@$(CXX) $(CXXFLAGS) -MM $< -MT $@ -MF $(patsubst $(SRC_DIR)/%.cpp,$(DEPS_DIR)/%.d,$<)

# CUDA compilation rule
$(CUDA_OBJ_DIR)/%.o: $(CUDA_SRC_DIR)/%.cu | $(CUDA_OBJ_DIR) $(CUDA_DEPS_DIR)
	@echo "Compiling CUDA source: $<"
	$(NVCC) $(NVCCFLAGS) -c $< -o $@
	@$(NVCC) $(NVCCFLAGS) -M $< -MT $@ -MF $(patsubst $(CUDA_SRC_DIR)/%.cu,$(CUDA_DEPS_DIR)/%.d,$<)

# --- Directory Creation ---

# Create directories if they don't exist
$(OBJ_DIR) $(CUDA_OBJ_DIR) $(DEPS_DIR) $(CUDA_DEPS_DIR):
	mkdir -p $@

# --- Housekeeping ---

clean:
	@echo "Cleaning up..."
	rm -rf $(OBJ_DIR) $(TARGET) $(TARGET_AUTONOMOUS) $(DEPS_DIR)

# Test targets
test_brain_architecture: test_brain_module_architecture.cpp $(OBJECTS)
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDFLAGS) $(LDLIBS)

test_phase1: test_phase1_integration.cpp $(filter-out $(OBJ_DIR)/main.o $(OBJ_DIR)/main_autonomous.o,$(OBJECTS))
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDFLAGS) $(LDLIBS)

.PHONY: all autonomous clean test_brain_architecture test_phase1

# Include dependency files
-include $(DEPS)
-include $(CUDA_DEPS)

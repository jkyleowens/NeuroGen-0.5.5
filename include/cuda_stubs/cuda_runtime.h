// Minimal CUDA runtime stub for CPU-only builds
#ifndef CUDA_RUNTIME_STUB_H
#define CUDA_RUNTIME_STUB_H

#include <cstddef>

// Stub CUDA function qualifiers (turn GPU functions into regular C++ functions)
#define __global__
#define __device__
#define __host__
#define __shared__
#define __constant__

// Stub dim3 structure for CPU builds
struct dim3 {
    unsigned int x, y, z;
    dim3(unsigned int _x = 1, unsigned int _y = 1, unsigned int _z = 1) : x(_x), y(_y), z(_z) {}
};

// Stub CUDA error type
typedef int cudaError_t;
#define cudaSuccess 0

// Stub memory management functions (no-ops for CPU)
inline cudaError_t cudaMalloc(void** ptr, size_t size) { *ptr = nullptr; return cudaSuccess; }
inline cudaError_t cudaFree(void* ptr) { return cudaSuccess; }
inline cudaError_t cudaMemcpy(void* dst, const void* src, size_t count, int kind) { return cudaSuccess; }
inline cudaError_t cudaDeviceSynchronize() { return cudaSuccess; }
inline cudaError_t cudaGetLastError() { return cudaSuccess; }
inline const char* cudaGetErrorString(cudaError_t error) { return "No error (CPU stub)"; }

// Memory copy kinds
#define cudaMemcpyHostToDevice 0
#define cudaMemcpyDeviceToHost 1
#define cudaMemcpyDeviceToDevice 2

// Stub stream and event types
typedef void* cudaStream_t;
typedef void* cudaEvent_t;
typedef void* cudaGraph_t;
typedef void* cudaGraphExec_t;
typedef void* cudaMemPool_t;

// Stub device properties structure
struct cudaDeviceProp {
    char name[256];
    size_t totalGlobalMem;
    size_t sharedMemPerBlock;
    int regsPerBlock;
    int warpSize;
    size_t memPitch;
    int maxThreadsPerBlock;
    int maxThreadsDim[3];
    int maxGridSize[3];
    int clockRate;
    size_t totalConstMem;
    int major;
    int minor;
    int multiProcessorCount;
    int l2CacheSize;
    int maxThreadsPerMultiProcessor;
    int computeMode;
};

// Stub device functions
inline cudaError_t cudaGetDeviceProperties(cudaDeviceProp* prop, int device) {
    if (prop) prop->name[0] = '\0';
    return cudaSuccess;
}

#endif // CUDA_RUNTIME_STUB_H

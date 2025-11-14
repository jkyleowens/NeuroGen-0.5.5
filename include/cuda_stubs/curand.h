// CUDA curand.h stub for CPU-only builds
#ifndef CURAND_H_STUB
#define CURAND_H_STUB

// Stub types
typedef void* curandGenerator_t;

// Stub enums
typedef enum {
    CURAND_STATUS_SUCCESS = 0
} curandStatus_t;

typedef enum {
    CURAND_RNG_PSEUDO_DEFAULT = 100
} curandRngType_t;

// Stub function declarations (inline no-ops for CPU builds)
inline curandStatus_t curandCreateGenerator(curandGenerator_t*, curandRngType_t) { return CURAND_STATUS_SUCCESS; }
inline curandStatus_t curandDestroyGenerator(curandGenerator_t) { return CURAND_STATUS_SUCCESS; }
inline curandStatus_t curandSetPseudoRandomGeneratorSeed(curandGenerator_t, unsigned long long) { return CURAND_STATUS_SUCCESS; }
inline curandStatus_t curandGenerateUniform(curandGenerator_t, float*, size_t) { return CURAND_STATUS_SUCCESS; }
inline curandStatus_t curandGenerateNormal(curandGenerator_t, float*, size_t, float, float) { return CURAND_STATUS_SUCCESS; }

#endif // CURAND_H_STUB

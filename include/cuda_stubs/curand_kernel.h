// CUDA curand_kernel.h stub for CPU-only builds
#ifndef CURAND_KERNEL_H_STUB
#define CURAND_KERNEL_H_STUB

// Stub type for curand state
typedef struct {
    unsigned int dummy;
} curandState;

typedef curandState curandState_t;

// Stub inline functions (no-ops for CPU builds)
inline void curand_init(unsigned long long seed, unsigned long long sequence,
                       unsigned long long offset, curandState* state) {
    (void)seed; (void)sequence; (void)offset; (void)state;
}

inline float curand_uniform(curandState* state) {
    (void)state;
    return 0.5f; // Return dummy value
}

inline float curand_normal(curandState* state) {
    (void)state;
    return 0.0f; // Return dummy value
}

inline unsigned int curand(curandState* state) {
    (void)state;
    return 0; // Return dummy value
}

#endif // CURAND_KERNEL_H_STUB

// Minimal cuBLAS stub for CPU-only builds
#ifndef CUBLAS_V2_STUB_H
#define CUBLAS_V2_STUB_H

// Stub cuBLAS types
typedef void* cublasHandle_t;
typedef int cublasStatus_t;

// Stub cuBLAS status codes
#define CUBLAS_STATUS_SUCCESS 0

// Stub cuBLAS functions (no-ops for CPU)
inline cublasStatus_t cublasCreate(cublasHandle_t* handle) { *handle = nullptr; return CUBLAS_STATUS_SUCCESS; }
inline cublasStatus_t cublasDestroy(cublasHandle_t handle) { return CUBLAS_STATUS_SUCCESS; }

#endif // CUBLAS_V2_STUB_H

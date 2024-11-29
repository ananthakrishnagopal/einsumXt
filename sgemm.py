
import ctypes
from ctypes import POINTER, byref, c_int, c_float, c_void_p, c_size_t
import numpy as np
import cupy as cp

# Constants
CUBLAS_STATUS_SUCCESS = 0

# Load cuBLAS library
def load_cublas_library():
    try:
        return ctypes.cdll.LoadLibrary("libcublas.so")
    except OSError:
        raise RuntimeError("cuBLAS library not found. Ensure libcublas.so is in your LD_LIBRARY_PATH.")

# Initialize cuBLAS ctypes signatures
def initialize_cublas_signatures(libcublas):
    libcublas.cublasXtCreate.restype = c_int
    libcublas.cublasXtCreate.argtypes = [POINTER(c_void_p)]
    libcublas.cublasXtDestroy.restype = c_int
    libcublas.cublasXtDestroy.argtypes = [c_void_p]
    libcublas.cublasXtDeviceSelect.restype = c_int
    libcublas.cublasXtDeviceSelect.argtypes = [c_void_p, c_int, POINTER(c_int)]
    libcublas.cublasXtSgemm.restype = c_int
    libcublas.cublasXtSgemm.argtypes = [
        c_void_p, c_int, c_int,
        c_size_t, c_size_t, c_size_t,
        POINTER(c_float),
        POINTER(c_float), c_size_t,
        POINTER(c_float), c_size_t,
        POINTER(c_float),
        POINTER(c_float), c_size_t
    ]

# Initialize cuBLAS
def initialize_cublas(libcublas):
    handle = c_void_p()
    status = libcublas.cublasXtCreate(byref(handle))
    if status != CUBLAS_STATUS_SUCCESS:
        raise RuntimeError("Failed to create cuBLAS handle.")
    return handle

# Set devices for cuBLAS
def set_cublas_device(libcublas, handle, device_ids):
    device_ids_np = np.array(device_ids, dtype=np.int32)
    status = libcublas.cublasXtDeviceSelect(handle, len(device_ids), device_ids_np.ctypes.data_as(POINTER(c_int)))
    if status != CUBLAS_STATUS_SUCCESS:
        raise RuntimeError("Failed to set devices for cuBLAS.")

# Perform SGEMM
def sgemm(libcublas, handle, m, n, k, A, B, C, alpha=1.0, beta=0.0):
    lda, ldb, ldc = m, k, m  # Leading dimensions
    alpha_ctypes = c_float(alpha)
    beta_ctypes = c_float(beta)

    # Get raw pointers
    A_ptr = get_array_pointer(A)
    B_ptr = get_array_pointer(B)
    C_ptr = get_array_pointer(C)

    # Call cuBLAS SGEMM
    status = libcublas.cublasXtSgemm(
        handle,
        0, 0,  # No transpose
        m,n,k,
        byref(alpha_ctypes),
        A_ptr,lda,
        B_ptr,ldb,
        byref(beta_ctypes),
        C_ptr,ldc
    )
    if status != CUBLAS_STATUS_SUCCESS:
        raise RuntimeError("SGEMM operation failed!")

# Helper function remains unchanged
def get_array_pointer(arr):
    if isinstance(arr, np.ndarray):
        return arr.ctypes.data_as(POINTER(c_float))
    elif isinstance(arr, cp.ndarray):
        return ctypes.cast(arr.data.ptr,POINTER(c_float))
    else:
        raise TypeError("Input must be a NumPy or CuPy array.")

# Cleanup cuBLAS
def cleanup_cublas(libcublas, handle):
    status = libcublas.cublasXtDestroy(handle)
    if status != CUBLAS_STATUS_SUCCESS:
        raise RuntimeError("Failed to destroy cuBLAS handle.")

def matmulXt(libcublas,handle,A,B):
  m,k = A.shape
  k,n = B.shape
  C_cp = cp.zeros((m, n), dtype=cp.float32)  # CuPy array for result
  sgemm(libcublas, handle, m, n, k, A_np, B_cp, C_cp, alpha=1.0, beta=0.0)
  return C_cp

# Main execution block (example usage)
if __name__ == "__main__":
    libcublas = load_cublas_library()
    initialize_cublas_signatures(libcublas)
    handle = initialize_cublas(libcublas)

    try:
        # Set device(s)
        set_cublas_device(libcublas, handle, [0])

        # Matrix dimensions
        m, n, k = 1024, 2048, 512

        # Example matrices (NumPy and CuPy)
        A_np = cp.ones((m, k), dtype=cp.float32)  # NumPy array
        B_cp = cp.ones((k, n), dtype=cp.float32)  # CuPy array
        # C_cp = cp.zeros((m, n), dtype=cp.float32)  # CuPy array

        # Perform SGEMM
        # sgemm(libcublas, handle, m, n, k, A_np, B_cp, C_cp, alpha=1.0, beta=0.0)
        C_cp = matmulXt(libcublas,handle,A_np,B_cp)
        # Verify results (move CuPy result back to NumPy for comparison)
        
        assert np.allclose(A_np @ B_cp, C_cp), "Result mismatch!"

        print("SGEMM operation successful!")
    finally:
        cleanup_cublas(libcublas, handle)

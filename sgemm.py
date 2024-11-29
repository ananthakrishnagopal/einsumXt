import ctypes
from ctypes import POINTER, byref, c_int, c_float
import numpy as np
# Load cuBLAS library
try:
    libcublas = ctypes.cdll.LoadLibrary("libcublas.so")
except OSError:
    raise RuntimeError("cuBLAS library not found. Ensure libcublas.so is in your LD_LIBRARY_PATH.")


libcublas.cublasXtCreate.restype = int
libcublas.cublasXtCreate.argtypes = [ctypes.c_void_p]
libcublas.cublasXtDestroy.restype = int
libcublas.cublasXtDestroy.argtypes = [ctypes.c_void_p]
libcublas.cublasXtDeviceSelect.restype = int
libcublas.cublasXtDeviceSelect.argtypes = [ctypes.c_void_p,
                                            ctypes.c_int,
                                            ctypes.c_void_p]
libcublas.cublasXtSgemm.restype = int
libcublas.cublasXtSgemm.argtypes = [ctypes.c_void_p,
                                     ctypes.c_int,
                                     ctypes.c_int,
                                     ctypes.c_size_t,
                                     ctypes.c_size_t,
                                     ctypes.c_size_t,
                                     ctypes.c_void_p,
                                     ctypes.c_void_p,
                                     ctypes.c_size_t,
                                     ctypes.c_void_p,
                                     ctypes.c_size_t,
                                     ctypes.c_void_p,
                                     ctypes.c_void_p,
                                     ctypes.c_size_t]

handle = ctypes.c_void_p()
libcublas.cublasXtCreate(ctypes.byref(handle))
deviceId = np.array([0], np.int32)
status = libcublas.cublasXtDeviceSelect(handle, 1,
                                         deviceId.ctypes.data)
if status:
    raise RuntimeError

def sgemm( m, n, k, A, B, C, alpha=1.0, beta=0.0):
        # Single-precision matrix multiplication
        lda, ldb, ldc = m, k, m
        alpha_ctypes = c_float(alpha)
        beta_ctypes = c_float(beta)

        # Convert arrays to ctypes
        A_ctypes = (c_float * len(A))(*A)
        B_ctypes = (c_float * len(B))(*B)
        C_ctypes = (c_float * len(C))(*C)

        status = libcublas.cublasXtSgemm(
            handle,
            0, 0,  # No transpose
            m, n, k,
            byref(alpha_ctypes),
            A_ctypes, lda,
            B_ctypes, ldb,
            byref(beta_ctypes),
            C_ctypes, ldc
        )
        if status != 0:
            raise RuntimeError("SGEMM operation failed!")

        # Return the updated C matrix
        return list(C_ctypes)

'''
# Matrix dimensions
m, n, k = 1024, 2048, 512

# Example matrices
A = [1.0] * (m * k)  # m x k matrix
B = [2.0] * (k * n)  # k x n matrix
C = [0.0] * (m * n)  # m x n result matrix

# Perform SGEMM
C_result = sgemm(m, n, k, A, B, C, alpha=1.0, beta=0.0)

print(np.array(A).reshape(m,k)@np.array(B).reshape(k,n)==np.array(C_result).reshape(m,n))
'''

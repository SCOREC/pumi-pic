#pragma once

#ifdef PP_USE_GPU
#define PP_INLINE __host__ __device__ inline
#define PP_DEVICE __device__ inline
#define PS_LAMBDA [=] __device__
#define PP_DEVICE_VAR __device__
#else
#define PP_INLINE inline
#define PP_DEVICE inline
#define PS_LAMBDA [=]
#define PP_DEVICE_VAR
#endif

//Cuda-aware check for OpenMPI 2.0+ taken from https://github.com/kokkos/kokkos/issues/2003
//Adapted from: https://docs.open-mpi.org/en/v5.0.x/man-openmpi/man3/MPIX_Query_rocm_support.3.html#mpix-query-rocm-support
#if defined(MPIX_CUDA_AWARE_SUPPORT) && MPIX_CUDA_AWARE_SUPPORT
#define PS_GPU_AWARE_MPI
#elif defined(OMPI_HAVE_MPI_EXT_ROCM) && OMPI_HAVE_MPI_EXT_ROCM
#define PS_GPU_AWARE_MPI
#endif

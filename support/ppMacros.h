#pragma once

#ifdef PP_USE_CUDA
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

#pragma once

#ifdef PS_USE_CUDA
#define PS_DEVICE __device__ inline
#define PS_LAMBDA [=] __device__
#define PS_DEVICE_VAR __device__
#else
#define PS_DEVICE inline
#define PS_LAMBDA [=]
#define PS_DEVICE_VAR
#endif

#ifndef __SCS_MACROS_H__
#define __SCS_MACROS_H__

#ifdef PS_USE_CUDA
#define SCS_DEVICE __device__ inline
#define SCS_LAMBDA [=] __device__
#define SCS_DEVICE_VAR __device__
#else
#define SCS_DEVICE inline
#define SCS_LAMBDA [=]
#define SCS_DEVICE_VAR
#endif

#endif

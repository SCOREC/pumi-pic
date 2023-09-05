#ifndef __SCS_MACROS_H__
#define __SCS_MACROS_H__

#ifdef PP_USE_GPU
#define SCS_DEVICE __device__ inline [[deprecated("Replaced with PS_DEVICE")]]
#define SCS_LAMBDA [=] __device__ [[deprecated("Replaced with PS_LAMBDA")]]
#define SCS_DEVICE_VAR __device__ [[deprecated("Replaced with PS_DEVICE_VAR")]]
#else
#define SCS_DEVICE inline [[deprecated("Replaced with PS_DEVICE")]]
#define SCS_LAMBDA [=] [[deprecated("Replaced with PS_LAMBDA")]]
#define SCS_DEVICE_VAR [[deprecated("Replaced with PS_DEVICE_VAR")]]
#endif

#endif

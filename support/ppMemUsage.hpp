#pragma once
#include <malloc.h> //warning - this is GNU-specific

#if defined(PP_USE_CUDA)
typedef Kokkos::CudaSpace DeviceSpace;
#elif defined(PP_USE_HIP)
typedef Kokkos::HIPSpace DeviceSpace;
#else
typedef Kokkos::HostSpace DeviceSpace;
#endif

static void hostGetMem(size_t* free, size_t* total) {
  const double M = 1024*1024;
#if defined(__GNUC__) && defined(PUMIPIC_HAS_MALLINFO2)
  struct mallinfo2 meminfo_now = mallinfo2();
  *total = meminfo_now.arena/M;
  *free = meminfo_now.fordblks/M;
#elif defined(__GNUC__)
  struct mallinfo meminfo_now = mallinfo();
  *total = meminfo_now.arena/M;
  *free = meminfo_now.fordblks/M;
#endif
}

static void getMemUsage(size_t* free, size_t* total)
{
#if defined(PP_USE_CUDA)
  cudaMemGetInfo(free, total);
#elif defined(PP_USE_HIP)
  hipMemGetInfo(free, total);
#else
  hostGetMem(free, total);
#endif
}

template <typename FunctionType>
FunctionType* gpuMemcpy(FunctionType& fn)
{
  FunctionType* fn_d;
  #ifdef PP_USE_CUDA
    cudaMalloc(&fn_d, sizeof(FunctionType));
    cudaMemcpy(fn_d,&fn, sizeof(FunctionType), cudaMemcpyHostToDevice);
  #elif defined(PP_USE_HIP)
    hipMalloc(&fn_d, sizeof(FunctionType));
    hipMemcpy(fn_d,&fn, sizeof(FunctionType), cudaMemcpyHostToDevice);
  #else
    fn_d = &fn;
  #endif
  return fn_d;
}

template <typename FunctionType>
void gpuFree(FunctionType& fn_d)
{
  #ifdef PP_USE_CUDA
    cudaFree(fn_d);
  #elif defined(PP_USE_HIP)
    hipFree(fn_d);
  #endif
}

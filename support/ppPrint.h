#pragma once

#ifdef SPDLOG_ENABLED
  #include "spdlog/spdlog.h"
#endif

#include <Kokkos_Core.hpp>

namespace pumipic {

  #if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__) || defined(__SYCL_DEVICE_ONLY__)
    #define ACTIVE_GPU_EXECUTION
  #endif

  template<typename... Args>
  void pPrintError(const char* fmt, const Args&... args) {
    #if defined(SPDLOG_ENABLED) && defined(PP_PRINT_ENABLED)
      spdlog::error(fmt, args...);
    #elif defined(PP_PRINT_ENABLED)
      fprintf(stderr, fmt, args...);
    #endif
  }

  template<typename... Args>
  __host__ __device__
  void pPrintInfo(const char* fmt, const Args&... args) {
    #if defined(SPDLOG_ENABLED) && defined(PP_PRINT_ENABLED) && !defined(ACTIVE_GPU_EXECUTION)
      spdlog::info(fmt, args...);
    #elif defined(PP_PRINT_ENABLED)
      Kokkos::printf(fmt, args...);
    #endif
  }


} 
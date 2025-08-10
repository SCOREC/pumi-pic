#ifndef PUMIPIC_PRINT_H
#define PUMIPIC_PRINT_H

#ifdef PUMIPIC_SPDLOG_ENABLED
  #include "spdlog/spdlog.h"
  #include <spdlog/fmt/bundled/printf.h>
#endif

#include <Kokkos_Core.hpp>
#include "ppMacros.h"
#include <stdio.h>
#include <stdarg.h>

namespace pumipic {

  #if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__) || defined(__SYCL_DEVICE_ONLY__)
    #define ACTIVE_GPU_EXECUTION
  #endif

  FILE* getStdout();
  FILE* getStderr();

  void setStdout(FILE* out);
  void setStderr(FILE* err);

  void printError(const char* fmt, ... );

  PP_INLINE
  void printInfo(const char* fmt, ...) {
    va_list ap;
    va_start(ap,fmt);
    #if defined(PUMIPIC_SPDLOG_ENABLED) && defined(PUMIPIC_PRINT_ENABLED) && !defined(ACTIVE_GPU_EXECUTION)
      spdlog::info("{}", fmt::vsprintf(fmt, ap));
    #elif defined(PUMIPIC_PRINT_ENABLED) && !defined(ACTIVE_GPU_EXECUTION)
      vfprintf(getStdout(), fmt, ap);
    #endif
    va_end(ap);
  }

}

#endif //PUMIPIC_PRINT_H

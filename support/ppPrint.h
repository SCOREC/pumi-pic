#ifndef PUMIPIC_PRINT_H
#define PUMIPIC_PRINT_H

#ifdef PUMIPIC_SPDLOG_ENABLED
  #include "spdlog/spdlog.h"
  #include <spdlog/fmt/bundled/printf.h>
#endif

#include <Kokkos_Core.hpp>
#include "ppMacros.h"

namespace pumipic {

  #if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__) || defined(__SYCL_DEVICE_ONLY__)
    #define ACTIVE_GPU_EXECUTION
  #endif

  FILE* getStdout();
  FILE* getStderr();

  void setStdout(FILE* out);
  void setStderr(FILE* err);

  template<typename... Args>
  void printError(std::string fmt, const Args&... args) {
    #if defined(PUMIPIC_SPDLOG_ENABLED) && defined(PUMIPIC_PRINT_ENABLED)
      spdlog::error("{}", fmt::sprintf(fmt, args...));
    #elif defined(PUMIPIC_PRINT_ENABLED)
      fprintf(getStderr(), ("[ERROR]"+fmt).c_str(), args...);
    #endif
  }

  template<typename... Args>
  PP_INLINE
  void printInfo(const char* fmt, const Args&... args) {
    #if defined(PUMIPIC_SPDLOG_ENABLED) && defined(PUMIPIC_PRINT_ENABLED) && !defined(ACTIVE_GPU_EXECUTION)
      spdlog::info("{}", fmt::sprintf(fmt, args...));
    #elif defined(PUMIPIC_PRINT_ENABLED) && !defined(ACTIVE_GPU_EXECUTION)
      fprintf(getStdout(), fmt, args...);
    #endif
  }

}

#endif //PUMIPIC_PRINT_H
#pragma once

#ifdef SPDLOG_ENABLED
  #include "spdlog/spdlog.h"
  #include <spdlog/fmt/bundled/printf.h>
#endif

#include <Kokkos_Core.hpp>
#include "ppMacros.h"

namespace pumipic {

  #if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__) || defined(__SYCL_DEVICE_ONLY__)
    #define ACTIVE_GPU_EXECUTION
  #endif

  inline FILE* pp_stdout = stdout;
  inline FILE* pp_stderr = stderr;

  inline void pp_set_stdout(FILE* out) {
    assert(out != NULL);
    pp_stdout = out;
  }

  inline void pp_set_stderr(FILE* err) {
    assert(err != NULL);
    pp_stderr = err;
  }

  template<typename... Args>
  void pPrintError(const char* fmt, const Args&... args) {
    #if defined(SPDLOG_ENABLED) && defined(PP_PRINT_ENABLED)
      spdlog::error("{}", fmt::sprintf(fmt, args...));
    #elif defined(PP_PRINT_ENABLED)
      fprintf(pp_stderr, fmt, args...);
    #endif
  }

  template<typename... Args>
  PP_INLINE
  void pPrintInfo(const char* fmt, const Args&... args) {
    #if defined(SPDLOG_ENABLED) && defined(PP_PRINT_ENABLED) && !defined(ACTIVE_GPU_EXECUTION)
      spdlog::info("{}", fmt::sprintf(fmt, args...));
    #elif defined(PP_PRINT_ENABLED) && !defined(ACTIVE_GPU_EXECUTION)
      fprintf(pp_stdout, fmt, args...);
    #endif
  }

}
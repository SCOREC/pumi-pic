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

  inline FILE* pp_stdout = stdout;
  inline FILE* pp_stderr = stderr;

  inline void setStdout(FILE* out) {
    assert(out != NULL);
    pp_stdout = out;
  }

  inline void setStderr(FILE* err) {
    assert(err != NULL);
    pp_stderr = err;
  }

  template<typename... Args>
  void printError(std::string fmt, const Args&... args) {
    #if defined(PUMIPIC_SPDLOG_ENABLED) && defined(PUMIPIC_PRINT_ENABLED)
      spdlog::error("{}", fmt::sprintf(fmt, args...));
    #elif defined(PUMIPIC_PRINT_ENABLED)
      fprintf(pp_stderr, ("[ERROR]"+fmt).c_str(), args...);
    #endif
  }

  template<typename... Args>
  PP_INLINE
  void printInfo(const char* fmt, const Args&... args) {
    #if defined(PUMIPIC_SPDLOG_ENABLED) && defined(PUMIPIC_PRINT_ENABLED) && !defined(ACTIVE_GPU_EXECUTION)
      spdlog::info("{}", fmt::sprintf(fmt, args...));
    #elif defined(PUMIPIC_PRINT_ENABLED) && !defined(ACTIVE_GPU_EXECUTION)
      fprintf(pp_stdout, fmt, args...);
    #endif
  }

}

#endif //PUMIPIC_PRINT_H
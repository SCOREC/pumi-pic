#pragma once

#ifdef SPDLOG_ENABLED
  #include "spdlog/spdlog.h"
#endif

#include <Kokkos_Core.hpp>

namespace pumipic {

  template<typename... Args>
  void pPrintError(const char* fmt, const Args&... args) {
    #ifdef SPDLOG_ENABLED
      spdlog::error(fmt, args...);
    #else
      fprintf(stderr, fmt, args...);
    #endif
  }

  template<typename... Args>
  void pPrintInfo(const char* fmt, const Args&... args) {
    #ifdef SPDLOG_ENABLED
      spdlog::info(fmt, args...);
    #else
      Kokkos::printf(fmt, args...);
    #endif
  }


} 
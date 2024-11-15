#pragma once

#ifdef SPDLOG_ENABLED
  #include "spdlog/spdlog.h"
#endif

#include <Kokkos_Core.hpp>

namespace pumipic {

  enum ppPrintOption { enable, disable };

  inline ppPrintOption ppCurrPrintOption = enable;

  inline void pSetLog(const ppPrintOption option){
    ppCurrPrintOption = option;
  }

  template<typename... Args>
  void pPrintError(const char* fmt, const Args&... args) {
    #ifdef SPDLOG_ENABLED
      if (ppCurrPrintOption == enable) spdlog::error(fmt, args...);
    #else
      if (ppCurrPrintOption == enable) fprintf(stderr, fmt, args...);
    #endif
  }

  template<typename... Args>
  void pPrintInfo(const char* fmt, const Args&... args) {
    #ifdef SPDLOG_ENABLED
      if (ppCurrPrintOption == enable) spdlog::info(fmt, args...);
    #else
      if (ppCurrPrintOption == enable) Kokkos::printf(fmt, args...);
    #endif
  }


} 
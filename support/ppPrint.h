#pragma once

#include "spdlog/spdlog.h"

namespace pumipic {

  template<typename... Args>
  void pPrintError(const char* fmt, const Args&... args) {
    spdlog::error(fmt, args...);
  }
} 
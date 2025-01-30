#include "ppPrint.h"

namespace pumipic {

  FILE* pp_stdout = stdout;
  FILE* pp_stderr = stderr;

  FILE* getStdout() { return pp_stdout; }
  FILE* getStderr() { return pp_stderr; }

  void setStdout(FILE* out) {
    assert(out != NULL);
    pp_stdout = out;
  }

  void setStderr(FILE* err) {
    assert(err != NULL);
    pp_stderr = err;
  }

  void printError(const char* fmt, ... ) {
    va_list ap;
    va_start(ap,fmt);
    #if defined(PUMIPIC_SPDLOG_ENABLED) && defined(PUMIPIC_PRINT_ENABLED)
      spdlog::error("{}", fmt::vsprintf(fmt, ap));
    #elif defined(PUMIPIC_PRINT_ENABLED)
      fprintf(getStderr(), "[ERROR]");
      vfprintf(getStderr(), fmt, ap);
    #endif
    va_end(ap);
  }
}
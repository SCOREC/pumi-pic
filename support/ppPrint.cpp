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
}
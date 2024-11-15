#include "ppAssert.h"
#include <cstdio>
#include <cstdlib>
#include "ppPrint.h"

namespace pumipic {

// borrowed from:
// https://github.com/SCOREC/core/blob/2472c570dc3a9e5acc23146288822733759558a1/pcu/pcu_util.[h|c]
void Assert_Fail(const char* msg) {
    pPrintError( "%s", msg);
    abort();
}

}

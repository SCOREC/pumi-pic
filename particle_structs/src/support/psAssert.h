#ifndef PSASSERT_H_
#define PSASSERT_H_

// borrowed from:
// ttps://github.com/SCOREC/core/blob/2472c570dc3a9e5acc23146288822733759558a1/pcu/pcu_util.[h|c]

#define ALWAYS_ASSERT(cond)                   \
  do {                                            \
    if (! (cond)) {                               \
      char omsg[2048];                            \
      sprintf(omsg, "%s failed at %s + %d \n",    \
              #cond, __FILE__, __LINE__);         \
      Assert_Fail(omsg);                      \
    }                                             \
  } while (0)

void Assert_Fail(const char* msg);

#endif

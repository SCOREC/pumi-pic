#include <malloc.h> //warning - this is GNU-specific

void hostGetMem(size_t* free, size_t* total) {
  const double M = 1024*1024;
#if defined(__GNUC__) && defined(PUMI_HAS_MALLINFO2)
  struct mallinfo2 meminfo_now = mallinfo2();
  *total = meminfo_now.arena/M;
  *free = meminfo_now.fordblks/M;
#elif defined(__GNUC__)
  struct mallinfo meminfo_now = mallinfo();
  *total = meminfo_now.arena/M;
  *free = meminfo_now.fordblks/M;
#endif
}

void getMemUsage(size_t* free, size_t* total)
{
#ifdef PP_USE_CUDA
  cudaMemGetInfo(free, total);
#else
  hostGetMem(free, total);
#endif
}
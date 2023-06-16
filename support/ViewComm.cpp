#include "ViewComm.h"
#include <unordered_map>
#include <mpi.h>
#include <functional>
namespace pumipic {

  Irecv_Map lambda_map;

  Irecv_Map& get_map() {return lambda_map;}

  //Adapted from https://www.open-mpi.org/faq/?category=runcuda
  bool checkCudaAwareMPI() {
    printf("Compile time check:\n");
#if defined(MPIX_CUDA_AWARE_SUPPORT) && MPIX_CUDA_AWARE_SUPPORT
    printf("This MPI library has CUDA-aware support.\n", MPIX_CUDA_AWARE_SUPPORT);
    return true;
#elif defined(MPIX_CUDA_AWARE_SUPPORT) && !MPIX_CUDA_AWARE_SUPPORT
    printf("This MPI library does not have CUDA-aware support.\n");
    return false;
#else
    printf("This MPI library cannot determine if there is CUDA-aware support.\n");
#endif /* MPIX_CUDA_AWARE_SUPPORT */

    printf("Run time check:\n");
#if defined(MPIX_CUDA_AWARE_SUPPORT)
    if (1 == MPIX_Query_cuda_support()) {
      printf("This MPI library has CUDA-aware support.\n");
    } else {
      printf("This MPI library does not have CUDA-aware support.\n");
    }
#else /* !defined(MPIX_CUDA_AWARE_SUPPORT) */
    printf("This MPI library cannot determine if there is CUDA-aware support.\n");
#endif /* MPIX_CUDA_AWARE_SUPPORT */
    return false;
  }

  //Adapted from: https://docs.open-mpi.org/en/v5.0.x/man-openmpi/man3/MPIX_Query_rocm_support.3.html#mpix-query-rocm-support
  bool checkROCmAwareMPI()
  {
    bool happy = false;
#if defined(OMPI_HAVE_MPI_EXT_ROCM) && OMPI_HAVE_MPI_EXT_ROCM
    happy = (bool) MPIX_Query_rocm_support();
#endif

    if (happy) {
      printf("This Open MPI installation has ROCm-aware support.\n");
      return true;
    } 
    else {
      printf("This Open MPI installation does not have ROCm-aware support.\n");
      return false;
    }
  }

  bool checkGPUAwareMPI()
  {
    return checkCudaAwareMPI() || checkROCmAwareMPI();
  }
}

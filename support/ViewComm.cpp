#include "ViewComm.h"
#include <unordered_map>
#include <mpi.h>
#include <functional>
namespace pumipic {

  Irecv_Map lambda_map;

  Irecv_Map& get_map() {return lambda_map;}

  //Adapted from https://www.open-mpi.org/faq/?category=runcuda
  bool checkCudaAwareMPI() {
    printf("Compile time check: ");
#if defined(MPIX_CUDA_AWARE_SUPPORT) && MPIX_CUDA_AWARE_SUPPORT
    printf("This MPI library has CUDA-aware support.\n", MPIX_CUDA_AWARE_SUPPORT);
    return true;
#elif defined(MPIX_CUDA_AWARE_SUPPORT) && !MPIX_CUDA_AWARE_SUPPORT
    printf("This MPI library does not have CUDA-aware support.\n");
    return false;
#else
    printf("This MPI library cannot determine if there is CUDA-aware support.\n");
#endif /* MPIX_CUDA_AWARE_SUPPORT */

    printf("Run time check: ");
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

  bool checkPSGPUAwareMPI()
  {
#ifdef PS_GPU_AWARE_MPI
    printf("GPU aware MPI has been enabled.\n");
    return true;
#else
    printf("GPU aware MPI is disabled.\n");
    return false;
#endif
  }

  bool checkGPUAwareMPI()
  {
    return checkCudaAwareMPI() || checkPSGPUAwareMPI();
  }
}

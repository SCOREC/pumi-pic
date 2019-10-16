#include "pumipic_profiling.hpp"
#include <mpi.h>
namespace {
  static bool pumipic_prebarrier_enabled = false;
}

void pumipic_enable_prebarrier() {
  pumipic_prebarrier_enabled = true;
}

double pumipic_prebarrier() {
  if(pumipic_prebarrier_enabled) {
    Kokkos::Timer timer;
    MPI_Barrier(MPI_COMM_WORLD);
    return timer.seconds();
  } else {
    return 0.0;
  }
}

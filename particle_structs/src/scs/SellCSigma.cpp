#include "SellCSigma.h"
#include <Kokkos_Core.hpp>
namespace particle_structs {
  namespace {
    static bool ps_prebarrier_enabled = false;
  };

  void enable_prebarrier() {
    ps_prebarrier_enabled = true;
  }

  double prebarrier() {
    if(ps_prebarrier_enabled) {
      Kokkos::Timer timer;
      MPI_Barrier(MPI_COMM_WORLD);
      return timer.seconds();
    } else {
      return 0.0;
    }
  }
}

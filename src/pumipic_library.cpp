#include "pumipic_library.hpp"

#include <Kokkos_Core.hpp>
#include <mpi.h>
namespace {
}

namespace pumipic {

  Library::Library(int* argc, char*** argv) {
  int is_mpi_init;
  MPI_Initialized(&is_mpi_init);
  own_mpi = !is_mpi_init;
  if (own_mpi) {
    MPI_Init(argc, argv);
  }
  own_kokkos = !Kokkos::is_initialized();
  if (own_kokkos) {
    Kokkos::initialize(*argc, *argv);
  }
  oh_lib = new Omega_h::Library(argc, argv);
}
Library::~Library() {
  delete oh_lib;
  if (own_kokkos) {
    Kokkos::finalize();
  }
  if (own_mpi) {
    MPI_Finalize();
  }
}

}

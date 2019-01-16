
#include <stdio.h>
#include <Kokkos_Core.hpp>

int main(int argc, char** argv) {

  Kokkos::initialize(argc,argv);
  fprintf(stderr, "Before policy\n");
  Kokkos::TeamPolicy<Kokkos::DefaultExecutionSpace> po(128, Kokkos::AUTO);
  fprintf(stderr, "size %d\n", po.team_size());

  Kokkos::finalize();
  printf("All tests passed\n");
  return 0;
}

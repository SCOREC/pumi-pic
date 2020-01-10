#include <stdio.h>
#include <Kokkos_Core.hpp>

#include <MemberTypes.h>
#include <SellCSigma.h>
#include <SCS_Macros.h>

#include <psAssert.h>
#include "Distribute.h"

using particle_structs::SellCSigma;
using particle_structs::MemberTypes;
using particle_structs::distribute_elements;
using particle_structs::distribute_particles;


typedef MemberTypes<int> Type;
typedef Kokkos::DefaultExecutionSpace exe_space;
typedef SellCSigma<Type,exe_space> SCS;
int main(int argc, char* argv[]) {
  MPI_Init(&argc, &argv);
  Kokkos::initialize(argc, argv);

  int ne = 5;
  int np = 20;
  int* ptcls_per_elem = new int[ne];
  std::vector<int>* ids = new std::vector<int>[ne];
  distribute_particles(ne, np, 0, ptcls_per_elem, ids);
  delete [] ids;
  {
    SCS::kkLidView ptcls_per_elem_v("ptcls_per_elem_v", ne);
    SCS::kkGidView element_gids_v("", 0);

    particle_structs::hostToDevice(ptcls_per_elem_v, ptcls_per_elem);

    Kokkos::TeamPolicy<exe_space> po(4, 4);
    SCS* scs = new SCS(po, 5, 2, ne, np, ptcls_per_elem_v, element_gids_v);

    delete [] ptcls_per_elem;

    scs->printFormatDevice();
    delete scs;
  }
  Kokkos::finalize();
  MPI_Finalize();
  printf("All tests passed\n");
  return 0;
}

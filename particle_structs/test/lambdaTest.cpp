#include <stdio.h>
#include <Kokkos_Core.hpp>

#include <particle_structs.hpp>

#include "Distribute.h"
#include "team_policy.hpp"

using particle_structs::SellCSigma;
using particle_structs::MemberTypes;


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
  {
    Kokkos::TeamPolicy<exe_space> po = pumipic::TeamPolicyAuto(4, 32);
    SCS::kkLidView ptcls_per_elem_v("ptcls_per_elem_v", ne);
    SCS::kkGidView element_gids_v("", 0);
    particle_structs::hostToDevice(ptcls_per_elem_v, ptcls_per_elem);

    SellCSigma<Type, exe_space>* scs =
      new SellCSigma<Type, exe_space>(po, 5, 2, ne, np, ptcls_per_elem_v, element_gids_v);
    delete [] ptcls_per_elem;
    delete [] ids;

    auto lamb = PS_LAMBDA(const int& eid, const int& pid, const int& mask) {
      if (mask > 0)
        printf("SECOND: %d %d\n", eid, pid);
    };

    scs->parallel_for(lamb);

    delete scs;
  }
  Kokkos::finalize();
  MPI_Finalize();
  printf("All tests passed\n");
  return 0;
}

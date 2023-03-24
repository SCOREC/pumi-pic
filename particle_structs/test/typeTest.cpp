
#include <stdio.h>
#include <Kokkos_Core.hpp>

#include <particle_structs.hpp>
#include "Distribute.h"
#include <ppAssert.h>
#include "team_policy.hpp"

using particle_structs::SellCSigma;
using particle_structs::MemberTypes;

int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);
  Kokkos::initialize(argc,argv);
  typedef MemberTypes<int> Type1;
  typedef MemberTypes<int,double[2]> Type2;
  typedef MemberTypes<int[3],double[2],char> Type3;

  printf("Type1: %lu\n",Type1::memsize);
  PS_ALWAYS_ASSERT(Type1::memsize == sizeof(int));
  printf("Type2: %lu\n",Type2::memsize);
  PS_ALWAYS_ASSERT(Type2::memsize == sizeof(int) + 2*sizeof(double));
  printf("Type3: %lu\n",Type3::memsize);
  PS_ALWAYS_ASSERT(Type3::memsize == 3*sizeof(int) + 2*sizeof(double) + sizeof(char));
  printf("Type3 start of doubles: %lu\n",Type3::sizeToIndex<1>());
  PS_ALWAYS_ASSERT(Type3::sizeToIndex<1>() == 3*sizeof(int));

  int ne = 5;
  int np = 10;
  int* ptcls_per_elem = new int[ne];
  std::vector<int>* ids = new std::vector<int>[ne];
  distribute_particles(ne,np, 0, ptcls_per_elem, ids);
  typedef Kokkos::DefaultExecutionSpace exe_space;
  Kokkos::TeamPolicy<exe_space> po = TeamPolicyAuto(128, 4);
  typedef SellCSigma<Type2> SCS;
  {
    SCS::kkLidView ptcls_per_elem_v("ptcls_per_elem_v", ne);
    SCS::kkGidView element_gids_v("", 0);
    particle_structs::hostToDevice(ptcls_per_elem_v, ptcls_per_elem);
    delete [] ptcls_per_elem;
    delete [] ids;

    SCS* scs = new SCS(po, 1, 10000, ne, np, ptcls_per_elem_v, element_gids_v);

    scs->printFormat();

    auto scs_first = scs->get<0>(); //int
    auto scs_second = scs->get<1>(); //double[2]

    auto setValues = PS_LAMBDA(int element_id, int particle_id, bool mask) {
      if (mask) {
        scs_first(particle_id) = element_id;
        scs_second(particle_id, 0) = 1.0;
        scs_second(particle_id, 1) = 2.0;
      }
      else {
        scs_first(particle_id) = -1;
        scs_second(particle_id, 0) = 0;
        scs_second(particle_id, 1) = 0;
      }
    };
    scs->parallel_for(setValues);
    delete scs;
  }

  Kokkos::finalize();
  MPI_Finalize();
  printf("All tests passed\n");
  return 0;
}

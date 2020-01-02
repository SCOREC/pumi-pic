#include <stdio.h>
#include <Kokkos_Core.hpp>

#include <MemberTypes.h>
#include <SellCSigma.h>

#include <psAssert.h>
#include "Distribute.h"

using particle_structs::SellCSigma;
using particle_structs::MemberTypes;
using particle_structs::distribute_elements;
using particle_structs::distribute_particles;


typedef MemberTypes<int> Type;
typedef Kokkos::DefaultExecutionSpace exe_space;
typedef SellCSigma<Type,exe_space> SCS;

bool defaultTest(int ne, int np, SCS::kkLidView ptcls_per_elem, SCS::kkGidView element_gids);
bool noSortTest(int ne, int np, SCS::kkLidView ptcls_per_elem, SCS::kkGidView element_gids);
bool largeCTest(int ne, int np, SCS::kkLidView ptcls_per_elem, SCS::kkGidView element_gids);

int main(int argc, char* argv[]) {
  MPI_Init(&argc, &argv);
  Kokkos::initialize(argc, argv);
  int ne = 5;
  int np = 20;
  int* ptcls_per_elem = new int[ne];
  std::vector<int>* ids = new std::vector<int>[ne];
  distribute_particles(ne, np, 2, ptcls_per_elem, ids);
  bool success = true;
  {
    SCS::kkLidView ptcls_per_elem_v("ptcls_per_elem_v", ne);
    SCS::kkGidView element_gids_v("", 0);
    particle_structs::hostToDevice(ptcls_per_elem_v, ptcls_per_elem);
    delete [] ptcls_per_elem;
    delete [] ids;

    success &= defaultTest(ne, np, ptcls_per_elem_v, element_gids_v);
    success &= noSortTest(ne, np, ptcls_per_elem_v, element_gids_v);
    success &= largeCTest(ne, np, ptcls_per_elem_v, element_gids_v);
  }
  Kokkos::finalize();
  MPI_Finalize();
  if (success)
    printf("All tests passed\n");
  return !success;
}

bool defaultTest(int ne, int np, SCS::kkLidView ptcls_per_elem, SCS::kkGidView element_gids) {
  printf("\nBeginning DefaultTest\n");
  int sigma = INT_MAX;
  int V = 2;
  Kokkos::TeamPolicy<exe_space> po(4, 4);
  SellCSigma<Type, exe_space>* scs =
    new SellCSigma<Type, exe_space>(po, sigma, V, ne, np, ptcls_per_elem, element_gids);

  scs->printFormat();
  SCS::kkLidView scs_ppe("scs_ppe",ne);
  auto lamb = PS_LAMBDA(const int& eid, const int& pid, const int& mask) {
    if (mask > 0)
      Kokkos::atomic_fetch_add(&scs_ppe(eid),1);
  };
  scs->parallel_for(lamb);
  SCS::kkLidView fail("fail",1);
  auto check = PS_LAMBDA(const int i) {
    if (scs_ppe(i) != ptcls_per_elem(i)) {
      printf("Element %d has incorrect number of particles (%d != %d)\n", i, scs_ppe(i), ptcls_per_elem(i));
      fail(0) = 1;
    }
  };
  Kokkos::parallel_for(ne, check);
  int f = particle_structs::getLastValue<particle_structs::lid_t>(fail);
  delete scs;
  return f == 0;
}

bool noSortTest(int ne, int np, SCS::kkLidView ptcls_per_elem, SCS::kkGidView element_gids) {
  printf("\nBeginning NoSort Test\n");
  int sigma = 1;
  int V = 2;
  Kokkos::TeamPolicy<exe_space> po(4, 4);
  SellCSigma<Type, exe_space>* scs =
    new SellCSigma<Type, exe_space>(po, sigma, V, ne, np, ptcls_per_elem, element_gids);

  scs->printFormat();
  SCS::kkLidView scs_ppe("scs_ppe",ne);
  auto lamb = PS_LAMBDA(const int& eid, const int& pid, const int& mask) {
    if (mask > 0)
      Kokkos::atomic_fetch_add(&scs_ppe(eid),1);
  };
  scs->parallel_for(lamb);
  SCS::kkLidView fail("fail",1);
  auto check = PS_LAMBDA(const int i) {
    if (scs_ppe(i) != ptcls_per_elem(i)) {
      printf("Element %d has incorrect number of particles (%d != %d)\n", i, scs_ppe(i), ptcls_per_elem(i));
      fail(0) = 1;
    }
  };
  Kokkos::parallel_for(ne, check);
  int f = particle_structs::getLastValue<particle_structs::lid_t>(fail);
  delete scs;
  return f == 0;
}

bool largeCTest(int ne, int np, SCS::kkLidView ptcls_per_elem, SCS::kkGidView element_gids) {
  printf("\nBeginning Large C/V Test\n");
  int sigma = INT_MAX;
  int V = 1024;
  Kokkos::TeamPolicy<exe_space> po(4, 32);
  SellCSigma<Type, exe_space>* scs =
    new SellCSigma<Type, exe_space>(po, sigma, V, ne, np, ptcls_per_elem, element_gids);

  scs->printFormat();
  SCS::kkLidView scs_ppe("scs_ppe",ne);
  auto lamb = PS_LAMBDA(const int& eid, const int& pid, const int& mask) {
    if (mask > 0)
      Kokkos::atomic_fetch_add(&scs_ppe(eid),1);
  };
  scs->parallel_for(lamb);
  SCS::kkLidView fail("fail",1);
  auto check = PS_LAMBDA(const int i) {
    if (scs_ppe(i) != ptcls_per_elem(i)) {
      printf("Element %d has incorrect number of particles (%d != %d)\n", i, scs_ppe(i), ptcls_per_elem(i));
      fail(0) = 1;
    }
  };
  Kokkos::parallel_for(ne, check);
  int f = particle_structs::getLastValue<particle_structs::lid_t>(fail);
  delete scs;
  return f == 0;
}

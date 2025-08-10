#include <stdio.h>
#include <Kokkos_Core.hpp>

#include <particle_structs.hpp>

#include <ppAssert.h>
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
  int f = 0;
  int ne = 5;
  int np = 20;
  int ppe = np/ne;
  int* ptcls_per_elem = new int[ne];
  std::vector<int>* ids = new std::vector<int>[ne];
  distribute_particles(ne, np, 0, ptcls_per_elem, ids);

  Kokkos::TeamPolicy<exe_space> po = pumipic::TeamPolicyAuto(4, 4);
  {
    SCS::kkLidView ptcls_per_elem_v("ptcls_per_elem_v", ne);
    SCS::kkGidView element_gids_v("", 0);
    SCS::kkLidView particle_element("particle_element", np);
    auto particle_info = particle_structs::createMemberViews<Type>(np);
    auto elem_info = particle_structs::getMemberView<Type, 0>(particle_info);
    Kokkos::parallel_for(np, KOKKOS_LAMBDA(const int i) {
      particle_element(i) = i / ppe;
      elem_info(i) = i / ppe;
    });
    particle_structs::hostToDevice(ptcls_per_elem_v, ptcls_per_elem);

    int sigma = INT_MAX;
    int V = 2;
    SellCSigma<Type, exe_space>* scs =
      new SellCSigma<Type, exe_space>(po, sigma, V, ne, np, ptcls_per_elem_v, element_gids_v,
                                      particle_element, particle_info);

    scs->printFormat();
    SCS::kkLidView fail("fail",1);
    auto elem_scs = scs->get<0>();
    auto check = PS_LAMBDA(const int eid, const int pid, const bool mask) {
      if (mask && (eid != elem_scs(pid))) {
        Kokkos::printf("Particle %d is not assigned to the correct element (%d != %d)\n", pid,
               eid, elem_scs(pid));
        fail(0) = 1;
      }
    };
    scs->parallel_for(check);
    f = particle_structs::getLastValue(fail);
    delete scs;
    delete [] ptcls_per_elem;
    delete [] ids;
    particle_structs::destroyViews<Type>(particle_info);

  }
  Kokkos::finalize();
  MPI_Finalize();
  if (!f)
    printf("All tests passed\n");
  return f;
}

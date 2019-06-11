#include <stdio.h>
#include <Kokkos_Core.hpp>

#include <MemberTypes.h>
#include <SellCSigma.h>
#include <SCS_Macros.h>

#include <psAssert.h>
#include <Distribute.h>

using particle_structs::SellCSigma;
using particle_structs::MemberTypes;
using particle_structs::distribute_elements;
using particle_structs::distribute_particles;


typedef MemberTypes<int> Type;
typedef Kokkos::DefaultExecutionSpace exe_space;
typedef SellCSigma<Type,exe_space> SCS;

int main(int argc, char* argv[]) {
  Kokkos::initialize(argc, argv);
  int f = 0;
  int ne = 5;
  int np = 20;
  int* ptcls_per_elem = new int[ne];
  std::vector<int>* ids = new std::vector<int>[ne];
  distribute_particles(ne, np, 2, ptcls_per_elem, ids);
  Kokkos::TeamPolicy<exe_space> po(4, 4);
  {
    SCS::kkLidView ptcls_per_elem_v("ptcls_per_elem_v", ne);
    SCS::kkGidView element_gids_v("", 0);
    particle_structs::hostToDevice(ptcls_per_elem_v, ptcls_per_elem);

    int sigma = INT_MAX;
    int V = 2;
    SellCSigma<Type, exe_space>* scs =
      new SellCSigma<Type, exe_space>(po, sigma, V, ne, np, ptcls_per_elem_v, element_gids_v);

    scs->printFormatDevice();
    SCS::kkLidView scs_ppe("scs_ppe",ne);
    auto lamb = SCS_LAMBDA(const int& eid, const int& pid, const int& mask) {
      if (mask > 0)
        Kokkos::atomic_fetch_add(&scs_ppe(eid),1);
    };
    scs->parallel_for(lamb);
    SCS::kkLidView fail("fail",1);
    auto check = SCS_LAMBDA(const int i) {
      if (scs_ppe(i) != ptcls_per_elem_v(i)) {
        printf("Element %d has incorrect number of particles (%d != %d)\n", i, scs_ppe(i), ptcls_per_elem_v(i));
        fail(0) = 1;
      }
    };
    Kokkos::parallel_for(ne, check);
    f = particle_structs::getLastValue<particle_structs::lid_t>(fail);
    delete scs;
    delete [] ptcls_per_elem;
    delete [] ids;

  }
  Kokkos::finalize();
  if (!f)
    printf("All tests passed\n");
  return f;
}

#include <stdio.h>
#include <Kokkos_Core.hpp>

#include <particle_structs.hpp>
#include "Distribute.h"
#include "team_policy.hpp"

namespace ps=particle_structs;
using particle_structs::SellCSigma;
using particle_structs::MemberTypes;
using particle_structs::getLastValue;
using particle_structs::lid_t;
typedef Kokkos::DefaultExecutionSpace exe_space;
typedef MemberTypes<int> Type;
typedef SellCSigma<Type> SCS;
typedef ps::SCS_Input<Type> Input;
bool padEvenly(Input& input);
bool padProportionally(Input& input);
bool padInversely(Input& input);

int main(int argc, char* argv[]) {
  MPI_Init(&argc, &argv);
  Kokkos::initialize(argc, argv);

  int fails = 0;

  // Create everything to build the SCS
  int ne = 100;
  int np = 100000;
  int* ptcls_per_elem = new int[ne];
  std::vector<int>* ids = new std::vector<int>[ne];
  distribute_particles(ne, np, 2, ptcls_per_elem, ids);
  int C = 4;
  Kokkos::TeamPolicy<exe_space> po = pumipic::TeamPolicyAuto(128, C);

  {
    SCS::kkLidView ptcls_per_elem_v("ptcls_per_elem_v", ne);
    SCS::kkGidView element_gids_v("", 0);
    particle_structs::hostToDevice(ptcls_per_elem_v, ptcls_per_elem);
    delete [] ptcls_per_elem;
    delete [] ids;
    Input input(po, ne, 1024, ne, np, ptcls_per_elem_v, element_gids_v);
    input.extra_padding = 0;
    if (!padEvenly(input)) {
      ++fails;
      printf("[ERROR] padEvenly() failed\n");
    }
    if (!padProportionally(input)) {
      ++fails;
      printf("[ERROR] padProportionally() failed\n");
    }
    if (!padInversely(input)) {
      ++fails;
      printf("[ERROR] padInversely() failed\n");
    }
  }
  Kokkos::finalize();
  MPI_Finalize();
  if (fails == 0) {
    printf("All tests passed\n");
    return 0;
  }
  else {
    printf("[ERROR] %d tests failed\n", fails);
    return 1;
  }
}

bool padEvenly(Input& input) {
  input.padding_strat = ps::PAD_EVENLY;
  SCS* scs = new SCS(input);
  printf("\nPadEvenly\nNum Ptcls %d, Capacity %d Asked for %d\n",scs->nPtcls(), scs->capacity(),
         (lid_t)(scs->nPtcls() * (1.0 + input.shuffle_padding)));
  scs->printMetrics();
  auto vals = scs->get<0>();
  auto lamb = PS_LAMBDA(const lid_t& elem, const lid_t& ptcl, const lid_t& mask) {
    vals(ptcl) = mask;
  };
  ps::parallel_for(scs, lamb, "pad_evenly");
  delete scs;
  return true;
}

bool padProportionally(Input& input) {
  input.padding_strat = ps::PAD_PROPORTIONALLY;
  SCS* scs = new SCS(input);
  printf("\nPadProportionally\nNum Ptcls %d, Capacity %d Asked for %d\n",scs->nPtcls(),
         scs->capacity(), (lid_t)(scs->nPtcls() * (1.0 + input.shuffle_padding)));
  scs->printMetrics();
  auto vals = scs->get<0>();
  auto lamb = PS_LAMBDA(const lid_t& elem, const lid_t& ptcl, const lid_t& mask) {
    vals(ptcl) = mask;
  };
  ps::parallel_for(scs, lamb, "pad_proportionally");
  delete scs;
  return true;
}

bool padInversely(Input& input) {
  input.padding_strat = ps::PAD_INVERSELY;
  SCS* scs = new SCS(input);
  printf("\nPadInversely\nNum Ptcls %d, Capacity %d Asked for %d\n",scs->nPtcls(),
         scs->capacity(), (lid_t)(scs->nPtcls() * (1.0 + input.shuffle_padding)));
  scs->printMetrics();
  auto vals = scs->get<0>();
  auto lamb = PS_LAMBDA(const lid_t& elem, const lid_t& ptcl, const lid_t& mask) {
    vals(ptcl) = mask;
  };
  ps::parallel_for(scs, lamb, "pad_inversely");
  delete scs;
  return true;
}

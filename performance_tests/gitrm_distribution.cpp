#include <particle_structs.hpp>
#include <ppTiming.hpp>
#include <Kokkos_Random.hpp>
#include "perfTypes.hpp"

void gitrm_distribute(int np, int cutoff, double percent, Kokkos::View<int*> ptcls_per_elem) {
  assert(percent <= 1);
  int ptcls_first = ceil(np*percent);
  int ptcls_second = np - ptcls_first;
  int ne = ptcls_per_elem.size();

  int seed = std::chrono::system_clock::now().time_since_epoch().count();
  Kokkos::Random_XorShift64_Pool<Kokkos::DefaultExecutionSpace> pool(seed);
  Kokkos::parallel_for(ptcls_first, KOKKOS_LAMBDA(const int i) {
    auto generator = pool.get_state();
    int index = generator.urand(0,cutoff);
    Kokkos::atomic_increment<int>(&ptcls_per_elem(index));
  });
  Kokkos::parallel_for(ptcls_second, KOKKOS_LAMBDA(const int i) {
    auto generator = pool.get_state();
    int index = generator.urand(cutoff,ne);
    Kokkos::atomic_increment<int>(&ptcls_per_elem(index));
  });
}
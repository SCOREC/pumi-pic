
#include <stdio.h>
#include <Kokkos_Core.hpp>
#include <mpi.h>
#include "SupportKK.h"
#include <Kokkos_Random.hpp>

int main(int argc, char* argv[]) {

  Kokkos::initialize(argc, argv);
  MPI_Init(&argc, &argv);

  for (int n = 100; n <= 1000000; n*=100) {
    Kokkos::View<uint32_t*> numbers("numbers", n);
    Kokkos::View<uint32_t*> result("result", n+1);

    Kokkos::Random_XorShift64_Pool<> random_pool(/*seed=*/n);
    Kokkos::parallel_for("set_arrays", n, KOKKOS_LAMBDA(const int i) {
      auto generator = random_pool.get_state();
      numbers(i) = generator.urand(/*max=*/1000);
      random_pool.free_state(generator);
    });

    pumipic::exclusive_scan(numbers, result, Kokkos::DefaultExecutionSpace());
    Kokkos::parallel_for("test_exclusive_scan", n, KOKKOS_LAMBDA(const int i) {
      assert(result(i) == result(i-1) + numbers(i-1));
    });

    pumipic::inclusive_scan(numbers, result, Kokkos::DefaultExecutionSpace());
    Kokkos::parallel_for("test_inclusive_scan", n, KOKKOS_LAMBDA(const int i) {
      assert(result(i) == result(i-1) + numbers(i));
    });
  }

  Kokkos::finalize();
  MPI_Finalize();
  return 0;
}

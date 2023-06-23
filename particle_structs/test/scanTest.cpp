
#include <stdio.h>
#include <Kokkos_Core.hpp>
#include <mpi.h>
#include "SupportKK.h"

int main(int argc, char* argv[]) {

  Kokkos::initialize(argc, argv);
  MPI_Init(&argc, &argv);

  for (int n = 100; n <= 1000000; n*=100) {
    Kokkos::View<int*> numbers("numbers", n);
    Kokkos::View<int*> result("result", n+1);

    Kokkos::parallel_for("set_arrays", n, KOKKOS_LAMBDA(const int i) {
      numbers(i) = i;
    });

    pumipic::exclusive_scan(numbers, result, Kokkos::DefaultExecutionSpace());
    Kokkos::parallel_for("set_arrays", n, KOKKOS_LAMBDA(const int i) {
      assert(result(i) == result(i-1) + numbers(i-1));
    });

    pumipic::inclusive_scan(numbers, result, Kokkos::DefaultExecutionSpace());
    Kokkos::parallel_for("set_arrays", n, KOKKOS_LAMBDA(const int i) {
      assert(result(i) == result(i-1) + numbers(i));
    });
  }

  Kokkos::finalize();
  MPI_Finalize();
  return 0;
}

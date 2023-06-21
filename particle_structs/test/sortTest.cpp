
#include <stdio.h>
#include <Kokkos_Core.hpp>
#include <Kokkos_Sort.hpp>
#include <mpi.h>

void performSorting(Kokkos::View<int*> arr, const char* name) {
  Kokkos::View<int*> copy("arr_copy", arr.size());
  Kokkos::parallel_for("copy_array", arr.size(), KOKKOS_LAMBDA(const int i) {
      copy[i] = arr[i];
  });
  Kokkos::Timer t;
  Kokkos::sort(arr, 0, arr.size());
  double kokkos_t = t.seconds();

  printf("%s sort time: %.6f\n", name, kokkos_t);
}

int main(int argc, char** argv) {

  Kokkos::initialize(argc,argv);
  MPI_Init(&argc, &argv);
  int n = atoi(argv[1]);

  {
    Kokkos::View<int*> sorted_arr("sorted",n);
    Kokkos::View<int*> backwards_arr("backwards",n);
    Kokkos::View<int*> jumbled_arr("jumbled",n);
    Kokkos::View<int*> two_buckets_arr("two_buckets",n);

    Kokkos::parallel_for("set_arrays", n, KOKKOS_LAMBDA(const int i) {
        sorted_arr(i) = i;
        backwards_arr(i) = n-i;
        jumbled_arr(i) = i*i % n;
        two_buckets_arr[i] = i < 10;
      });

    performSorting(sorted_arr, "Sorted");
    performSorting(backwards_arr, "Backwards");
    performSorting(jumbled_arr, "Jumbled");
    performSorting(two_buckets_arr, "Two Buckets");

  }

  Kokkos::finalize();
  MPI_Finalize();
  return 0;
}

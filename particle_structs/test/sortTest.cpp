
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
  Kokkos::fence();
  printf("%s sort time: %.6f\n", name, t.seconds());
}

void sigmaSort(Kokkos::View<int*> arr, int sigma, const char* name){
  Kokkos::Timer t;
  using ExecSpace = Kokkos::DefaultExecutionSpace;
  using TeamPol = Kokkos::TeamPolicy<ExecSpace>;
  using TeamMem = typename TeamPol::member_type;
  Kokkos::View<int*> index("index", arr.size());
  Kokkos::parallel_for(arr.size(), KOKKOS_LAMBDA(const int& i) {
    index(i) = i;
  });

  int n_sigma = arr.size()/sigma;
  Kokkos::parallel_for( TeamPol(n_sigma, 1), KOKKOS_LAMBDA(const TeamMem& t){
    int start = t.league_rank() * sigma;
    int end = (t.league_rank() == n_sigma-1) ? arr.size() : start + sigma;
    auto range = Kokkos::make_pair(start, end);
    auto ptcl_subview = Kokkos::subview(arr, range);
    auto index_subview = Kokkos::subview(index, range);
    Kokkos::Experimental::sort_by_key_thread(t, ptcl_subview, index_subview);
  });
  Kokkos::fence();
  printf("%s sort time: %.6f\n", name, t.seconds());
}

int main(int argc, char** argv) {

  Kokkos::initialize(argc,argv);
  MPI_Init(&argc, &argv);
  int n = atoi(argv[1]);
  
  printf("Sorting views of size %d\n", n);
  {
    Kokkos::View<int*> sorted_arr("sorted",n);
    Kokkos::View<int*> backwards_arr("Backwards",n);
    Kokkos::View<int*> jumbled_arr("Jumbled",n);
    Kokkos::View<int*> two_buckets_arr("Two Buckets",n);

    Kokkos::parallel_for("set_arrays", n, KOKKOS_LAMBDA(const int i) {
      sorted_arr(i) = i;
      backwards_arr(i) = n-i;
      jumbled_arr(i) = i*i % n;
      two_buckets_arr(i) = i < 10;
    });

    performSorting(sorted_arr, "Sorted");
    performSorting(backwards_arr, "Backwards");
    performSorting(jumbled_arr, "Jumbled");
    performSorting(two_buckets_arr, "Two Buckets");

    Kokkos::parallel_for("set_arrays", n, KOKKOS_LAMBDA(const int i) {
      sorted_arr(i) = i;
      backwards_arr(i) = n-i;
      jumbled_arr(i) = i*i % n;
      two_buckets_arr(i) = i < 10;
    });

    int sigma = 100;
    printf("Sigma: %d\n", sigma);
    sigmaSort(sorted_arr, sigma, "Sigma Sorted");
    sigmaSort(backwards_arr, sigma, "Sigma Backwards");
    sigmaSort(jumbled_arr, sigma, "Sigma Jumbled");
    sigmaSort(two_buckets_arr, sigma, "Sigma Two Buckets");
  }

  Kokkos::finalize();
  MPI_Finalize();
  return 0;
}

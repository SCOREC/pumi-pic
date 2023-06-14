#include <ppView.h>
#include <particle_structs.hpp>

namespace pp = pumipic;

int constructTypes();
int parallelFor();
int parallelReduce();

int main(int argc, char* argv[]) {
  Kokkos::initialize(argc, argv);
  MPI_Init(&argc, &argv);
  int fails = 0;

  fails += constructTypes();
  fails += parallelFor();
  fails += parallelReduce();
  Kokkos::finalize();
  MPI_Finalize();
  if (!fails) {
    printf("All Tests Passed\n");
  }
  else {
    printf("%d Tests Failed\n", fails);
  }
  return fails;
}


typedef double Vector3f[3];
int constructTypes() {
  int fails = 0;
  pp::View<int*> intView(10);
  if (intView.size() != 10 || intView->size() != 10) {
    fprintf(stderr, "[ERROR] Incorrect size on view of ints [(view) %d != %d (actual)]\n", intView.size(), 10);
    ++fails;
  }
  pp::View<Vector3f*> dblView("dblView", 20);
  if (dblView.size() != 3 * 20 || dblView->size() != 3 * 20) {
    fprintf(stderr, "[ERROR] Incorrect size on view of doubles\n");
    ++fails;
  }
  return fails;
}

int parallelFor() {
  int fails = 0;
  pp::View<int*> intView(10);
  Kokkos::parallel_for(10, KOKKOS_LAMBDA(const int& i) {
      intView(i) = i;
    });
  pp::View<Vector3f*> dblView("dblView", 20);
  Kokkos::parallel_for(20, KOKKOS_LAMBDA(const int& i) {
    for (int j = 0; j < 3; ++j)
      dblView(i, j) = intView(i/2) * j * M_PI;
  });

  return fails;
}

int parallelReduce() {
  int fails = 0;
  pp::View<int*> intView(10);
  Kokkos::parallel_for(10, KOKKOS_LAMBDA(const int& i) {
      intView(i) = i;
    });
  int total;
  Kokkos::parallel_reduce(10, KOKKOS_LAMBDA(const int& i, int& sum) {
      sum += intView(i);
    }, total);
  int res = (9 + 0) * 10/2;
  if (total != res) {
    fprintf(stderr, "[ERROR] Reduction over intView does not give proper result"
            "[(view) %d != %d (actual)]\n", total, res);
    ++fails;
  }
  pp::View<Vector3f*> dblView("dblView", 20);
  Kokkos::parallel_for(20, KOKKOS_LAMBDA(const int& i) {
    for (int j = 0; j < 3; ++j)
      dblView(i, j) = 1.0/3;
  });
  double tot;
  Kokkos::parallel_reduce(20, KOKKOS_LAMBDA(const int& i, double& sum) {
      for (int j = 0; j < 3; ++j)
        sum += dblView(i,j);
    }, tot);
  if (fabs(tot - 20) > .0001) {
    fprintf(stderr, "[ERROR] Reduction over dblView does not give proper result"
            "[(view) %f != %d (actual)]\n", tot, 20);
    ++fails;
  }
  return fails;
}

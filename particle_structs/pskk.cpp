#include <stdio.h>
#include <cstdlib>
#include <vector>
#include "SellCSigma.h"
#include "Distribute.h"
#include "Push.h"
#include <math.h>
#include <time.h>
#include <cassert>

#include <Kokkos_Core.hpp>

void positionsMatch(int np,
    double* x1, double* y1, double* z1,
    double* x2, double* y2, double* z2) {
  //Confirm all particles were pushed
  double EPSILON = 0.0001;
  for (int i = 0; i < np; ++i) {
    if(abs(x1[i] - x2[i]) > EPSILON) exit(EXIT_FAILURE);
    if(abs(y1[i] - y2[i]) > EPSILON) exit(EXIT_FAILURE);
    if(abs(z1[i] - z2[i]) > EPSILON) exit(EXIT_FAILURE);
  }
}

void checkThenClear(int np,
    double* x1, double* y1, double* z1,
    double* x2, double* y2, double* z2) {
  positionsMatch(np, x1, y1, z1, x2, y2, z2);
  for(int i=0; i<np; i++)
    x2[i] = y2[i] = z2[i] = 0;
}

double randD(double fMin, double fMax)
{
    double f = (double)rand() / RAND_MAX;
    return fMin + f * (fMax - fMin);
}
int main(int argc, char* argv[]) {
  Kokkos::initialize(argc,argv);
  printf("Kokkos execution space memory %s name %s\n",
      typeid (Kokkos::DefaultExecutionSpace::memory_space).name(),
      typeid (Kokkos::DefaultExecutionSpace).name());
  printf("Kokkos host execution space %s name %s\n",
      typeid (Kokkos::DefaultHostExecutionSpace::memory_space).name(),
      typeid (Kokkos::DefaultHostExecutionSpace).name());
  srand(time(NULL));
  if (argc != 6) {
    printf("Usage: %s <number of elements> <number of particles> <distribution strategy (0-3)> "
           "<C> <sigma>\n", argv[0]);
    return 1;
  }
  int ne = atoi(argv[1]);
  int np = atoi(argv[2]);
  int strat = atoi(argv[3]);

  //Distribute particles to 'elements'
  int* ptcls_per_elem = new int[ne];
  std::vector<int>* ids = new std::vector<int>[ne];
  if (!distribute_particles(ne,np,strat,ptcls_per_elem, ids)) {
    return 1;
  }

#ifdef DEBUG
  int ps = 0;
  printf("Particle Distribution\n");
  for (int i = 0;i < ne; ++i) {
    printf("Element %d has %d particles:",i, ptcls_per_elem[i]);
    ps += ptcls_per_elem[i];
    for (int j = 0; j < ptcls_per_elem[i]; ++j) {
      printf(" %d",ids[i][j]);
      assert(ids[i][j] >= 0 && ids[i][j] < np);
    }
    printf("\n");
  }
  assert(ps == np);
#endif

  //Create the SellCSigma for particles
  int C = atoi(argv[4]);
  int sigma = atoi(argv[5]);
  SellCSigma* scs = new SellCSigma(C, sigma, ne, np, ptcls_per_elem,ids);

  //Create Coordinates
  double* xs = new double[np];
  double* ys = new double[np];
  double* zs = new double[np];
  for (int i = 0; i < np; ++i) {
    xs[i] = 0.125;
    ys[i] = 5;
    zs[i] = M_PI;
  }

  //Push the particles
  double distance = M_E;
  double dx = randD(-10,10);
  double dy = randD(-10,10);
  double dz = randD(-10,10);
  double length = sqrt(dx * dx + dy * dy + dz * dz);
  dx /= length;
  dy /= length;
  dz /= length;

  double* new_xs1 = new double[np];
  double* new_ys1 = new double[np];
  double* new_zs1 = new double[np];
  Kokkos::Timer timer;
  push_array(np, xs, ys, zs, distance, dx, dy, dz, new_xs1, new_ys1, new_zs1);
  fprintf(stderr, "serial array push (seconds) %f\n", timer.seconds());

  double* new_xs2 = new double[np];
  double* new_ys2 = new double[np];
  double* new_zs2 = new double[np];

  timer.reset();
  push_scs(scs, xs, ys, zs, distance, dx, dy, dz, new_xs2, new_ys2, new_zs2);
  fprintf(stderr, "serial scs push (seconds) %f\n", timer.seconds());

  checkThenClear(np,
      new_xs1, new_ys1, new_zs1,
      new_xs2, new_ys2, new_zs2);

  timer.reset();
  push_array_kk(np, xs, ys, zs, distance, dx, dy, dz, new_xs2, new_ys2, new_zs2);
  fprintf(stderr, "kokkos array push and transfer (seconds) %f\n", timer.seconds());

  checkThenClear(np,
      new_xs1, new_ys1, new_zs1,
      new_xs2, new_ys2, new_zs2);

  timer.reset();
  push_scs_kk(scs, np, xs, ys, zs, distance, dx, dy, dz, new_xs2, new_ys2, new_zs2);
  fprintf(stderr, "kokkos scs push and transfer (seconds) %f\n", timer.seconds());

  checkThenClear(np,
      new_xs1, new_ys1, new_zs1,
      new_xs2, new_ys2, new_zs2);

  //Cleanup
  delete [] new_xs2;
  delete [] new_ys2;
  delete [] new_zs2;
  delete [] new_xs1;
  delete [] new_ys1;
  delete [] new_zs1;
  delete [] xs;
  delete [] ys;
  delete [] zs;
  delete scs;
  delete [] ids;
  delete [] ptcls_per_elem;
  Kokkos::finalize();
  return 0;
}

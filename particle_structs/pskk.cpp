#include <stdio.h>
#include <cstdlib>
#include <vector>
#include "SellCSigma.h"
#include "Distribute.h"
#include "Push.h"
#include "psTypes.h"
#include "psParams.h"
#include <math.h>
#include <time.h>
#include <cassert>
#include <chrono>
#include <thread>

#include <Kokkos_Core.hpp>

void printTimerResolution() {
  Kokkos::Timer timer;
  std::this_thread::sleep_for(std::chrono::milliseconds(1));
  fprintf(stderr, "kokkos timer reports 1ms as %f seconds\n", timer.seconds());
}

void matchFailed(int i) {
  fprintf(stderr, "position match failed on particle %d\n", i);
  exit(EXIT_FAILURE);
}

const double EPSILON = 0.0001;

void positionsMatch(int np,
    fp_t* x1, fp_t* y1, fp_t* z1,
    SellCSigma* scs) {
  //Confirm all particles were pushed
  for (int i = 0; i < np; ++i) {
    const int scsIdx = scs->arr_to_scs[i];
    if(abs(x1[i] - scs->scs_new_xs[scsIdx]) > EPSILON) {
      fprintf(stderr, "(%.2f) x[%d] != scs_x[%d] (%.2f)\n",
          x1[i], i, scsIdx, scs->scs_new_xs[scsIdx]);
      matchFailed(i);
    }
    if(abs(y1[i] - scs->scs_new_ys[scsIdx]) > EPSILON) matchFailed(i);
    if(abs(z1[i] - scs->scs_new_zs[scsIdx]) > EPSILON) matchFailed(i);
  }
}

void positionsMatch(int np,
    fp_t* x1, fp_t* y1, fp_t* z1,
    fp_t* x2, fp_t* y2, fp_t* z2) {
  //Confirm all particles were pushed
  for (int i = 0; i < np; ++i) {
    if(abs(x1[i] - x2[i]) > EPSILON) matchFailed(i);
    if(abs(y1[i] - y2[i]) > EPSILON) matchFailed(i);
    if(abs(z1[i] - z2[i]) > EPSILON) matchFailed(i);
  }
}

void checkThenClear(int np,
    fp_t* x1, fp_t* y1, fp_t* z1,
    SellCSigma* scs) {
  positionsMatch(np, x1, y1, z1, scs);
  for(int i=0; i<np; i++) {
    scs->scs_new_xs[i] = 0;
    scs->scs_new_ys[i] = 0;
    scs->scs_new_zs[i] = 0;
  }
}

void checkThenClear(int np,
    fp_t* x1, fp_t* y1, fp_t* z1,
    fp_t* x2, fp_t* y2, fp_t* z2) {
  positionsMatch(np, x1, y1, z1, x2, y2, z2);
  for(int i=0; i<np; i++)
    x2[i] = y2[i] = z2[i] = 0;
}

fp_t randD(fp_t fMin, fp_t fMax)
{
    fp_t f = (fp_t)rand() / RAND_MAX;
    return fMin + f * (fMax - fMin);
}
int main(int argc, char* argv[]) {
  Kokkos::initialize(argc,argv);
  printf("floating point value size (bits): %zu\n", sizeof(fp_t));
  printf("Kokkos execution space memory %s name %s\n",
      typeid (Kokkos::DefaultExecutionSpace::memory_space).name(),
      typeid (Kokkos::DefaultExecutionSpace).name());
  printf("Kokkos host execution space %s name %s\n",
      typeid (Kokkos::DefaultHostExecutionSpace::memory_space).name(),
      typeid (Kokkos::DefaultHostExecutionSpace).name());
  printTimerResolution();
  srand(time(NULL));
  if (argc != 8) {
    printf("Usage: %s <number of elements> <number of particles> <distribution strategy (0-3)> "
           "<C> <sigma> <V> <debug=0|1>\n", argv[0]);
    return 1;
  }
  int ne = atoi(argv[1]);
  int np = atoi(argv[2]);
  int strat = atoi(argv[3]);

  fprintf(stderr, "distribution %d-%s #elements %d #particles %d\n",
      strat, distribute_name(strat), ne, np);

  //Distribute particles to 'elements'
  int* ptcls_per_elem = new int[ne];
  std::vector<int>* ids = new std::vector<int>[ne];
  if (!distribute_particles(ne,np,strat,ptcls_per_elem, ids)) {
    return 1;
  }

  //The array based particle container has each particle store
  // the id of its parent element.
  int* ptcl_to_elem = new int[np];
  for (int i = 0;i < ne; ++i)
    for (int j = 0; j < ptcls_per_elem[i]; ++j)
      ptcl_to_elem[ ids[i][j] ] = i;

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

  //The point of this test is to have the particle push kernels access
  // mesh information.  We will assume (1) that relevent fields are
  // stored on mesh vertices and (2) that the vertex information
  // was preprocessed such that the information for the vertices that
  // bound each element are stored in a contiguous portion of the array
  // for that information.  This duplication of vertex info eliminates
  // 'jumps' through the vertex arrays and thus improves performance;
  // something a 'real' analysis code may do.
  elemCoords elems(ne,4);
  //Write something into the coordinate arrays. Does not matter.
  for( int i=0; i<ne*4; i++ ) {
    elems.x[i] = i*0.1;
    elems.y[i] = i*0.1;
    elems.z[i] = i*0.1;
  }

  //Create Coordinates
  fp_t* xs = new fp_t[np];
  fp_t* ys = new fp_t[np];
  fp_t* zs = new fp_t[np];
  for (int i = 0; i < np; ++i) {
    xs[i] = i;
    ys[i] = 0.125;
    zs[i] = M_PI;
  }

  //Create the SellCSigma for particles
  int C = atoi(argv[4]);
  int sigma = atoi(argv[5]);
  int V = atoi(argv[6]);
  bool debug = atoi(argv[7]);
  fprintf(stderr, "Sell-C-sigma C %d V %d sigma %d\n", C, V, sigma);
  SellCSigma* scs = new SellCSigma(C, sigma, V, ne, np, ptcls_per_elem,
				   ids, xs, ys, zs, debug);


  //Push the particles
  fp_t distance = M_E;
  fp_t dx = randD(-10,10);
  fp_t dy = randD(-10,10);
  fp_t dz = randD(-10,10);
  fp_t length = sqrt(dx * dx + dy * dy + dz * dz);
  dx /= length;
  dy /= length;
  dz /= length;

  fp_t* new_xs1 = new fp_t[np];
  fp_t* new_ys1 = new fp_t[np];
  fp_t* new_zs1 = new fp_t[np];
  push_array(np, xs, ys, zs, ptcl_to_elem, elems,
      distance, dx, dy, dz, new_xs1, new_ys1, new_zs1);

  push_scs(scs, ptcl_to_elem, elems, distance, dx, dy, dz);

  fprintf(stderr, "done serial\n");
  checkThenClear(np, new_xs1, new_ys1, new_zs1, scs);

  fprintf(stderr, "\n");
  fp_t* new_xs2 = new fp_t[np];
  fp_t* new_ys2 = new fp_t[np];
  fp_t* new_zs2 = new fp_t[np];
  Kokkos::Timer timer;
  push_array_kk(np, xs, ys, zs, ptcl_to_elem, elems,
      distance, dx, dy, dz, new_xs2, new_ys2, new_zs2);
  fprintf(stderr, "kokkos array push and transfer (seconds) %f\n", timer.seconds());

  checkThenClear(np,
      new_xs1, new_ys1, new_zs1,
      new_xs2, new_ys2, new_zs2);

  fprintf(stderr, "\n");
  timer.reset();
  push_scs_kk(scs, np, elems, distance, dx, dy, dz);
  fprintf(stderr, "kokkos scs push and transfer (seconds) %f\n", timer.seconds());

  checkThenClear(np, new_xs1, new_ys1, new_zs1, scs);

  //Cleanup
  delete [] new_xs1;
  delete [] new_ys1;
  delete [] new_zs1;
  delete [] new_xs2;
  delete [] new_ys2;
  delete [] new_zs2;
  delete [] xs;
  delete [] ys;
  delete [] zs;
  delete scs;
  delete [] ids;
  delete [] ptcls_per_elem;
  delete [] ptcl_to_elem;
  Kokkos::finalize();
  fprintf(stderr,"\ndone\n");
  return 0;
}

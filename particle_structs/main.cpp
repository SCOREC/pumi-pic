#include <stdio.h>
#include <cstdlib>
#include <vector>
#include "SellCSigma.h"
#include "Distribute.h"
#include "Push.h"
#include "psTypes.h"
#include <math.h>
#include <time.h>
#include <cassert>

double randD(double fMin, double fMax)
{
    double f = (double)rand() / RAND_MAX;
    return fMin + f * (fMax - fMin);
}
int main(int argc, char* argv[]) {
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
  fp_t* xs = new fp_t[np];
  fp_t* ys = new fp_t[np];
  fp_t* zs = new fp_t[np];
  for (int i = 0; i < np; ++i) {
    xs[i] = 0.125;
    ys[i] = 5;
    zs[i] = M_PI;
  }

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
  push_array(np, xs, ys, zs, distance, dx, dy, dz, new_xs1, new_ys1, new_zs1);

  fp_t* new_xs2 = new fp_t[np];
  fp_t* new_ys2 = new fp_t[np];
  fp_t* new_zs2 = new fp_t[np];
  push_scs(scs, xs, ys, zs, distance, dx, dy, dz, new_xs2, new_ys2, new_zs2);

  //Confirm all particles were pushed
  fp_t EPSILON = 0.0001;
  for (int i = 0; i < np; ++i) {
    if(abs(new_xs1[i] - new_xs2[i]) > EPSILON) exit(EXIT_FAILURE);
    if(abs(new_ys1[i] - new_ys2[i]) > EPSILON) exit(EXIT_FAILURE);
    if(abs(new_zs1[i] - new_zs2[i]) > EPSILON) exit(EXIT_FAILURE);
  }
  
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
  return 0;
}

#include <stdio.h>
#include <cstdlib>
#include <vector>
#include "SellCSigma.h"
#include "Distribute.h"


int main(int argc, char* argv[]) {
  if (argc != 4) {
    printf("Usage: %s <number of elements> <number of particles> <distribution strategy>\n", argv[0]);
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
  printf("Particle Distribution\n");
  for (int i = 0;i < ne; ++i) {
    printf("Element %d has %d particles:",i, ptcls_per_elem[i]);
    for (int j = 0; j < ptcls_per_elem[i]; ++j)
      printf(" %d",ids[i][j]);
    printf("\n");
  }
#endif

  //Create the SellCSigma for particles
  int C = 5;
  int sigma = 10;
  SellCSigma* scs = new SellCSigma(C, sigma, ne, np, ptcls_per_elem,ids);

  delete scs;
  delete [] ptcls_per_elem;
  return 0;
}

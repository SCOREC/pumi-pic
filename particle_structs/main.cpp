#include <stdio.h>
#include <cstdlib>
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

  int* ptcls_per_elem = new int[ne];

  if (!distribute_particles(ne,np,strat,ptcls_per_elem)) {
    return 1;
  }


  SellCSigma* scs = new SellCSigma(ne, np, ptcls_per_elem);

  
  delete scs;
  delete [] ptcls_per_elem;
  return 0;
}

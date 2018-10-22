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

  int* ptcls_per_elem = new int[ne];
  std::vector<int>* ids = new std::vector<int>[ne];
  if (!distribute_particles(ne,np,strat,ptcls_per_elem, ids)) {
    return 1;
  }

  for (int i = 0;i < ne; ++i) {
    printf("%d:",ptcls_per_elem[i]);
    for (int j = 0; j < ptcls_per_elem[i]; ++j)
      printf(" %d",ids[i][j]);
    printf("\n");
  }

  int C = 4;
  int sigma = 12;
  SellCSigma* scs = new SellCSigma(C, sigma, ne, np, ptcls_per_elem,ids);

  
  delete scs;
  delete [] ptcls_per_elem;
  return 0;
}

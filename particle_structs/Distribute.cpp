#include "Distribute.h"
#include <stdio.h>
void uniform_distribution(int ne, int np, int* ptcls_per_elem);
bool distribute_particles(int ne, int np, int strat, int* ptcls_per_elem) {
  if (strat==0) {
    uniform_distribution(ne,np,ptcls_per_elem);
  }
  else {
    printf("Unknown distribution strategy. Avaible distributions:\n"
           "  0 - Uniform\n");
    return false;
  }
  return true;
}

void uniform_distribution(int ne, int np, int* ptcls_per_elem) {
  int p = np / ne;
  int r = np % ne;
  for (int i = 0; i < r; ++i) {
    ptcls_per_elem[i] = p + 1;
  }
  for (int i = r; i < ne; ++i) {
    ptcls_per_elem[i] = p;
  }
}

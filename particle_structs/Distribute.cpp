#include "Distribute.h"
#include <stdio.h>
#include <time.h>
#include <cstdlib>
void uniform_distribution(int ne, int np, int* ptcls_per_elem);
void random_distribution(int ne, int np, int* ptcls_per_elem);
bool distribute_particles(int ne, int np, int strat, int* ptcls_per_elem) {
  if (strat==0) {
    uniform_distribution(ne,np,ptcls_per_elem);
  }
  else if (strat==1) {
    random_distribution(ne,np,ptcls_per_elem);
  }
  else {
    printf("Unknown distribution strategy. Avaible distributions:\n"
           "  0 - Uniform\n"
           "  1 - Random\n");
    return false;
  }
  return true;
}

void uniform_distribution(int ne, int np, int* ptcls_per_elem) {
  int p = np / ne;
  int r = np % ne;
  //Give remainder to first elements
  for (int i = 0; i < r; ++i) {
    ptcls_per_elem[i] = p + 1;
  }
  //Then fill remaining elements
  for (int i = r; i < ne; ++i) {
    ptcls_per_elem[i] = p;
  }
}
void random_distribution(int ne, int np, int* ptcls_per_elem) {
  srand(time(NULL));
  //Set particles to 0
  for (int i = 0; i < ne; ++i) {
    ptcls_per_elem[i] = 0;
  }
  //Randomly assign each particle to an element
  for (int i = 0; i < np; ++i) {
    int elem = rand() % ne;
    ptcls_per_elem[elem]++;
  }
}

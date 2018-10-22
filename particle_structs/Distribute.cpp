#include "Distribute.h"
#include <stdio.h>
#include <time.h>
#include <cstdlib>
void uniform_distribution(int ne, int np, int* ptcls_per_elem, std::vector<int>* ids);
void random_distribution(int ne, int np, int* ptcls_per_elem, std::vector<int>* ids);
bool distribute_particles(int ne, int np, int strat, int* ptcls_per_elem, std::vector<int>* ids) {
  if (strat==0) {
    uniform_distribution(ne,np,ptcls_per_elem,ids);
  }
  else if (strat==1) {
    random_distribution(ne,np,ptcls_per_elem,ids);
  }
  else {
    printf("Unknown distribution strategy. Avaible distributions:\n"
           "  0 - Uniform\n"
           "  1 - Random\n");
    return false;
  }
  return true;
}

void uniform_distribution(int ne, int np, int* ptcls_per_elem, std::vector<int>* ids) {
  int p = np / ne;
  int r = np % ne;
  int index = 0;
  //Give remainder to first elements
  for (int i = 0; i < r; ++i) {
    ptcls_per_elem[i] = p + 1;
    for (int j = 0; j < p + 1; ++j)
      ids[i].push_back(index++);
  }
  //Then fill remaining elements
  for (int i = r; i < ne; ++i) {
    ptcls_per_elem[i] = p;
    for (int j = 0; j < p; ++j)
      ids[i].push_back(index++);

  }
}
void random_distribution(int ne, int np, int* ptcls_per_elem, std::vector<int>* ids) {
  srand(time(NULL));
  //Set particles to 0
  for (int i = 0; i < ne; ++i) {
    ptcls_per_elem[i] = 0;
  }
  //Randomly assign each particle to an element
  int index = 0;
  for (int i = 0; i < np; ++i) {
    int elem = rand() % ne;
    ptcls_per_elem[elem]++;
    ids[elem].push_back(index++);
  }
}

#include "Distribute.h"
#include <chrono>
#include <stdio.h>
#include <random>
#include <math.h>
#include <cstdlib>
void even_distribution(int ne, int np, int* ptcls_per_elem, std::vector<int>* ids);
void uniform_distribution(int ne, int np, int* ptcls_per_elem, std::vector<int>* ids);
void gaussian_distribution(int ne, int np, int* ptcls_per_elem, std::vector<int>* ids);
void exponential_distribution(int ne, int np, int* ptcls_per_elem, std::vector<int>* ids);
bool distribute_particles(int ne, int np, int strat, int* ptcls_per_elem, std::vector<int>* ids) {
  if (strat==0) {
    even_distribution(ne,np,ptcls_per_elem,ids);
  }
  else if (strat==1) {
    uniform_distribution(ne,np,ptcls_per_elem,ids);
  }
  else if (strat==2) {
    gaussian_distribution(ne,np,ptcls_per_elem,ids);
  }
  else if (strat==3) {
    exponential_distribution(ne,np,ptcls_per_elem,ids);
  }
  else {
    printf("Unknown distribution strategy. Avaible distributions:\n"
           "  0 - Evenly\n"
           "  1 - Uniform\n"
           "  2 - Gaussian\n"
           "  3 - Exponential\n");
    return false;
  }
  return true;
}

void even_distribution(int ne, int np, int* ptcls_per_elem, std::vector<int>* ids) {
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
void uniform_distribution(int ne, int np, int* ptcls_per_elem, std::vector<int>* ids) {
  //Set particles to 0
  for (int i = 0; i < ne; ++i) {
    ptcls_per_elem[i] = 0;
  }
  int seed = std::chrono::system_clock::now().time_since_epoch().count();
  std::default_random_engine generator(seed);
  std::uniform_int_distribution<int> distribution(0, ne - 1);

  //Randomly assign each particle to an element
  int index = 0;
  for (int i = 0; i < np; ++i) {
    int elem = distribution(generator);
    ptcls_per_elem[elem]++;
    ids[elem].push_back(index++);
  }
}

void gaussian_distribution(int ne, int np, int* ptcls_per_elem, std::vector<int>* ids) {
  //Set particles to 0
  for (int i = 0; i < ne; ++i) {
    ptcls_per_elem[i] = 0;
  }
  //Distribute based on normal distribution
  int seed = std::chrono::system_clock::now().time_since_epoch().count();
  std::default_random_engine generator(seed);
  std::normal_distribution<double> distribution(ne/2.0, ne/5.0);
  int index = 0;
  for (int i = 0; i < np; ++i) {
    int elem;
    do {
      elem = round(distribution(generator));
      if (elem >= 0 && elem < ne) {
        ptcls_per_elem[elem]++;
        ids[elem].push_back(index++);
      }
    } while (elem < 0 || elem >= ne);
  }
}

void exponential_distribution(int ne, int np, int* ptcls_per_elem, std::vector<int>* ids) {
  //Set particles to 0
  for (int i = 0; i < ne; ++i) {
    ptcls_per_elem[i] = 0;
  }
  //Distribute based on normal distribution
  int seed = std::chrono::system_clock::now().time_since_epoch().count();
  std::default_random_engine generator(seed);
  std::exponential_distribution<double> distribution(4);
  int index = 0;
  for (int i = 0; i < np; ++i) {
    double rand;
    do {
      rand = distribution(generator);
      if (rand < 1.0) {
        int elem = ne * rand;
        ptcls_per_elem[elem]++;
        ids[elem].push_back(index++);
      }
    }
    while (rand >= 1.0);
  }
}

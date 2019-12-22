#include "Distribute.h"
#include <chrono>
#include <stdio.h>
#include <random>
#include <math.h>
#include <cstdlib>

namespace {

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

const int num_dist_funcs = 4;
typedef void (*dist_func)(int ne, int np, int* ptcls_per_elem, std::vector<int>* ids);
typedef const char* dist_name ;

dist_func funcs[num_dist_funcs] = {
  &even_distribution,
  &uniform_distribution,
  &gaussian_distribution,
  &exponential_distribution
};
dist_name names[num_dist_funcs] = {
  "Evenly",
  "Uniform",
  "Gaussian",
  "Exponential"
};

void distribute_help() {
  printf("\nUnknown distribution strategy. Avaible distributions:\n");
  for(int i=0; i<num_dist_funcs; i++)
    printf("%d - %s\n", i, particle_structs::distribute_name(i));
}

} //end unnamed namespace

namespace particle_structs {

const char* distribute_name(int strat) {
  if(strat >= 0 && strat < num_dist_funcs) {
    return names[strat];
  } else {
    distribute_help();
    exit(EXIT_FAILURE);
  }
}

bool distribute_elements(int ne, int strat, int comm_rank, int comm_size, gid_t* gids) {
  //For now only building a ring of elements
  //Assumes the number of elements on each process is the same
  int starting_index = (ne-1) * comm_rank;
  for (int i = 0; i < ne; ++i)
    gids[i] = starting_index+i;
  if (comm_rank == comm_size-1 && comm_size != 1)
    gids[ne-1] = 0;
  return true;
}

bool distribute_particles(int ne, int np, int strat, int* ptcls_per_elem, std::vector<int>* ids) {
  if(strat >= 0 && strat < num_dist_funcs)
    (*funcs[strat])(ne,np,ptcls_per_elem,ids);
  else {
    distribute_help();
    return false;
  }
  return true;
}

} // end particle_structs namespace

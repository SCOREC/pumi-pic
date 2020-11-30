#include "Distribute.h"
#include <chrono>
#include <stdio.h>
#include <random>
#include <math.h>
#include <cstdlib>
#ifdef PP_USE_CUDA
#include <cuda.h>
#include <curand.h>
#include <curand_kernel.h>
#endif
namespace {

void even_distribution(int ne, int np, int* ptcls_per_elem, std::vector<int>* ids) {
  int p;
  int r;
  if (ne == 0)
    p = r = 0;
  else {
    p= np / ne;
    r = np % ne;
  }
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

void even_distribution(int ne, int np, Kokkos::View<int*> ptcls_per_elem,
                       Kokkos::View<int*> elem_per_ptcl,float param) {
  int p;
  int r;
  if (ne == 0)
    p = r = 0;
  else {
    p= np / ne;
    r = np % ne;
  }
  //Give remainder to first elements
  Kokkos::parallel_for(r, KOKKOS_LAMBDA(const int index) {
    ptcls_per_elem[index] = p + 1;
  });
  Kokkos::parallel_for(r * (p+1), KOKKOS_LAMBDA(const int index) {
    elem_per_ptcl[index] = index / (p + 1);
  });
  //Then fill remaining elements
  Kokkos::parallel_for(ne-r, KOKKOS_LAMBDA(const int index) {
    ptcls_per_elem[index] = p;
  });
  Kokkos::parallel_for((ne -r) * p, KOKKOS_LAMBDA(const int index) {
    elem_per_ptcl[index] = r + index / (p);
  });
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

void uniform_distribution(int ne, int np, Kokkos::View<int*> ptcls_per_elem,
                          Kokkos::View<int*> elem_per_ptcl,float param) {
  int seed = std::chrono::system_clock::now().time_since_epoch().count();
  Kokkos::Random_XorShift64_Pool<Kokkos::DefaultExecutionSpace> pool(seed);
  Kokkos::parallel_for(np, KOKKOS_LAMBDA(const int index) {
    auto generator = pool.get_state();
    const int elem = generator.urand(ne);
    pool.free_state(generator);
    elem_per_ptcl[index] = elem;
    Kokkos::atomic_add(&(ptcls_per_elem[elem]), 1);

  });
}

void gaussian_distribution(int ne, int np, int* ptcls_per_elem, std::vector<int>* ids) {
  //Set particles to 0
  for (int i = 0; i < ne; ++i) {
    ptcls_per_elem[i] = 0;
  }
  //Distribute based on normal distribution
  int seed = std::chrono::system_clock::now().time_since_epoch().count();
  std::default_random_engine generator(seed);
  std::normal_distribution<double> distribution(ne/2.0, ne/8.0);
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

void gaussian_distribution(int ne, int np, Kokkos::View<int*> ptcls_per_elem,
                           Kokkos::View<int*> elem_per_ptcl,float param) {
  int seed = std::chrono::system_clock::now().time_since_epoch().count();
  Kokkos::Random_XorShift64_Pool<Kokkos::DefaultExecutionSpace> pool(seed);
  Kokkos::parallel_for(np, KOKKOS_LAMBDA(const int index) {
    auto generator = pool.get_state();
    int elem = generator.normal(ne/2.0, ne/8.0);
    pool.free_state(generator);
    if (elem < 0)
      elem = 0;
    if (elem >= ne)
      elem = ne - 1;
    elem_per_ptcl[index] = elem;
    Kokkos::atomic_add(&(ptcls_per_elem[elem]), 1);
  });
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

#ifdef PP_USE_CUDA
  const int num_states = 1024;
  Kokkos::View<curandState_t*> cuda_states;
  bool states_inited = false;
  void initStates() {
    if (!states_inited) {
      cuda_states = Kokkos::View<curandState_t*>("cuda_states", 1024);
      states_inited = true;
      int seed = std::chrono::system_clock::now().time_since_epoch().count();
      auto local_states = cuda_states;
      Kokkos::parallel_for(num_states, PS_LAMBDA(const int index) {
        curand_init(seed, index * 2 * 100, 1, &(local_states(index)));
      });
    }
  }
#endif


void exponential_distribution(int ne, int np, Kokkos::View<int*> ptcls_per_elem,
                              Kokkos::View<int*> elem_per_ptcl, float poisson_num) {

  //For now just call the CPU version
#ifdef PP_USE_CUDA
  //Initialize cuda states for PRNG
  initStates();

  //Initialize bin lengths/starts
  int base = pow(ne, .3);
  if (base < 2)
    base = 2;
  if (base > 32)
    base = 32;
  Kokkos::View<int*> bin_starts("bin_starts", 5);
  int bin_starts_host[5];
  bin_starts_host[0] = 0;
  for (int i = 1; i < 10; ++i) {
    bin_starts_host[i] = bin_starts_host[i-1] + pow(base, i);
  }
  pumipic::hostToDevice(bin_starts, bin_starts_host);
  int ptcls_per_state = np / num_states + 1;
  auto local_states = cuda_states;
  //Generate a random bin from poisson distribution then assign an element in the bin
  //  using uniform random distribution
  Kokkos::parallel_for(num_states, PS_LAMBDA(const int index) {
    curandState_t state = local_states[index];
    const int start_ptcl = ptcls_per_state * index;
    for (int i = 0; i < ptcls_per_state; ++i) {
      const int ptcl_index = start_ptcl + i;
      int bin = curand_poisson(&state, poisson_num);
      if (bin >= 5)
        bin = 4;
      const int range = pow(base, bin + 1);
      const int minElem = bin_starts[bin];
      int elem = minElem + curand(&state) % range;
      if (elem >= ne) {
        elem = ne - 1;
      }
      if (ptcl_index < np) {
        elem_per_ptcl[ptcl_index] = elem;
        Kokkos::atomic_add(&(ptcls_per_elem[elem]), 1);
      }
    }
  });
#else
  int* ppe = new int[ne];
  std::vector<int>* ids = new std::vector<int>[ne];
  exponential_distribution(ne, np, ppe, ids);
  int* new_elems = new int[np];
  for (int i = 0; i < ne; ++i) {
    for (std::size_t j = 0; j < ids[i].size(); ++j) {
      new_elems[ids[i][j]] = i;
    }
  }
  pumipic::hostToDevice(ptcls_per_elem, ppe);
  pumipic::hostToDevice(elem_per_ptcl, new_elems);
  delete [] new_elems;
  delete [] ppe;
  delete [] ids;
#endif
}

const int num_dist_funcs = 4;
typedef void (*dist_func)(int ne, int np, int* ptcls_per_elem, std::vector<int>* ids);
typedef void (*dist_func_gpu)(int, int, Kokkos::View<int*>, Kokkos::View<int*>,float);
typedef const char* dist_name ;

dist_func funcs[num_dist_funcs] = {
  &even_distribution,
  &uniform_distribution,
  &gaussian_distribution,
  &exponential_distribution
};
dist_func_gpu gpu_funcs[num_dist_funcs] = {
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
    printf("%d - %s\n", i, distribute_name(i));
}

} //end unnamed namespace

const char* distribute_name(int strat) {
  if(strat >= 0 && strat < num_dist_funcs) {
    return names[strat];
  } else {
    distribute_help();
    exit(EXIT_FAILURE);
  }
}

bool distribute_elements(int ne, int strat, int comm_rank, int comm_size, pumipic::gid_t* gids) {
  //For now only building a ring of elements
  //Assumes the number of elements on each process is the same
  int starting_index = (ne-1) * comm_rank;
  for (int i = 0; i < ne; ++i)
    gids[i] = starting_index+i;
  if (comm_rank == comm_size-1 && comm_size != 1 && ne > 0)
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
/*
bool distribute_particles(int ne, int np, int strat, Kokkos::View<int*> ptcls_per_elem,
                          Kokkos::View<int*> elem_per_ptcl) {
  if(strat >= 0 && strat < num_dist_funcs)
    (*gpu_funcs[strat])(ne,np,ptcls_per_elem,elem_per_ptcl);
  else {
    distribute_help();
    return false;
  }
  return true;

}
*/
bool distribute_particles(int ne, int np, int strat, Kokkos::View<int*> ptcls_per_elem,
                          Kokkos::View<int*> elem_per_ptcl, float param=.01) {
  if(strat == 3)
    (*gpu_funcs[3])(ne,np,ptcls_per_elem,elem_per_ptcl,param);
  else if(strat >= 0 && strat < num_dist_funcs)
    (*gpu_funcs[strat])(ne,np,ptcls_per_elem,elem_per_ptcl,0);
  else {
    distribute_help();
    return false;
  }
  return true;

}

void cleanup_distribution_memory() {
#ifdef PP_USE_CUDA
  cuda_states = Kokkos::View<curandState_t*>(0);
#endif
}

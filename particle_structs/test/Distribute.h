#ifndef DISTRIBUTE_H_
#define DISTRIBUTE_H_

#include <vector>
#include <SCS_Types.h>
#include <Kokkos_Random.hpp>
#include <particle_structs.hpp>
#include <SupportKK.h>


bool distribute_elements(int ne, int strat, pumipic::gid_t* gids);

bool distribute_particles(int ne, int np, int strat, int* ptcls_per_elem,
                          std::vector<int>* ids);

bool distribute_particles(int ne, int np, int strat, Kokkos::View<int*> ptcls_per_elem,
                          Kokkos::View<int*> elem_per_ptcl,float param=1.0);

template <typename PS>
bool redistribute_particles(PS* ptcls, int strat, double percentMoved,
                            typename PS::kkLidView new_elms);

const char* distribute_name(int strat);


//Define a seed so each call are roughly the "same" across different runs
#define DISTRIBUTE_SEED 1024 * 1024
template <typename PS>
bool redistribute_particles(PS* ptcls, int strat, double percentMoved,
                            typename PS::kkLidView new_elems) {
  assert(0 <= percentMoved);
  assert(percentMoved <= 1);
  Kokkos::Random_XorShift64_Pool<typename PS::execution_space> pool(DISTRIBUTE_SEED);
  typename PS::kkLidView is_moving("is_moving", ptcls->capacity());
  auto decideMovers = PS_LAMBDA(const int e, const int p, const bool mask) {
    if (mask) {
      auto generator = pool.get_state();
      double prob = generator.drand(1.0);
      pool.free_state(generator);
      is_moving[p] = (prob <= percentMoved);
    }
  };
  pumipic::parallel_for(ptcls, decideMovers, "decideMovers");
  typename PS::kkLidView mover_index("mover_index", ptcls->capacity());
  Kokkos::parallel_scan("indexMovers", is_moving.size(),
                        KOKKOS_LAMBDA(const int i, int& update, const bool fin) {
                          if (fin) {
                            mover_index[i] = -1;
                            if (is_moving[i])
                              mover_index[i] = update;
                          }
                          update+= is_moving[i];
                        });
  int total = 0;
  Kokkos::parallel_reduce("countMovers", is_moving.size(),
                          KOKKOS_LAMBDA(const int i, int& update) {
                            update += is_moving[i];
                          }, total);
  int numElems = ptcls->nElems();
  typename PS::kkLidView new_ppe("new_ppe", numElems);
  typename PS::kkLidView new_moves_d("new_moves", total);
  distribute_particles(numElems, total, strat, new_ppe, new_moves_d);
  auto assignMoves = PS_LAMBDA(const int e, const int p, const bool mask) {
    if (mask) {
      if (is_moving[p]) {
        int index = mover_index[p];
        new_elems[p] = new_moves_d[index];
        if (new_moves_d[index] < 0)
          printf("ERROR %d\n", new_moves_d[index]);
        else if (new_moves_d[index] >= numElems)
          printf("ERROR %d\n", new_moves_d[index]);
      }
      else {
        new_elems[p] = e;
      }
    }
    else {
      new_elems[p] = -1;
    }
  };
  pumipic::parallel_for(ptcls, assignMoves, "assignMoves");
  return true;
}

#endif

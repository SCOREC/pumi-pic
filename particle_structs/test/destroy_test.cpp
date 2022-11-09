#include <particle_structs.hpp>
#include "read_particles.hpp"
#include "Distribute.h"
#include <mpi.h>

const char* structure_names[4] = { "SCS", "CSR", "CabM", "DPS" };
int comm_rank, comm_size;

bool destroyConstructor(int ne_in, int np_in, int distribution, int structure);
bool destroyRebuild(int ne_in, int np_in, int distribution, int structure);
bool destroyMigrate(int ne_in, int np_in, int distribution, int structure);

int main(int argc, char* argv[]) {
  Kokkos::initialize(argc, argv);
  MPI_Init(&argc,&argv);

  MPI_Comm_rank(MPI_COMM_WORLD,&comm_rank);
  MPI_Comm_size(MPI_COMM_WORLD,&comm_size);

  int ne = 5;
  int np = 20;
  int distribution = 1;
  int fails = 0;

  for (int i = 0; i < 4; i++) {
    if (!comm_rank) fprintf(stderr,"destroyConstructor\n");
    fails += destroyConstructor(ne, np, distribution, i);
    if (!comm_rank) fprintf(stderr,"destroyRebuild\n");
    fails += destroyRebuild(ne, np, distribution, i);
    if (!comm_rank) fprintf(stderr,"destroyMigrate\n");
    fails += destroyMigrate(ne, np, distribution, i);
  }
  
  Kokkos::finalize();
  int total_fails = 0;
  MPI_Reduce(&fails, &total_fails, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
  MPI_Finalize();
  if (!comm_rank && total_fails == 0)
    printf("All tests passed\n");
  return total_fails;
}

bool destroyConstructor(int ne_in, int np_in, int distribution, int structure) {
  int fails = 0;

  ps::gid_t* gids = new ps::gid_t[ne_in];
  distribute_elements(ne_in, 0, comm_rank, comm_size, gids);
  int* ptcls_per_elem = new int[ne_in];
  std::vector<int>* ids = new std::vector<int>[ne_in];
  distribute_particles(ne_in, np_in, 2, ptcls_per_elem, ids);
  delete [] ids;

  PS::kkLidView ptcls_per_elem_v("ptcls_per_elem_v", ne_in);
  PS::kkGidView element_gids_v("element_gids_v", ne_in);
  ps::hostToDevice(ptcls_per_elem_v, ptcls_per_elem);
  ps::hostToDevice(element_gids_v, gids);
  delete [] ptcls_per_elem;
  delete [] gids;
  Kokkos::TeamPolicy<ExeSpace> po(4, 32);
  
  // create and destroy structure
  size_t free, total;
  cudaMemGetInfo(&free, &total);
  const long used_before=total-free;

  PS* ptcls;
  if (structure == 0) {
    ptcls = new ps::SellCSigma<Types, MemSpace>(po, 5,2, ne_in, np_in, ptcls_per_elem_v, element_gids_v);
  }
  else if (structure == 1) {
    ptcls = new ps::CSR<Types, MemSpace>(po, ne_in, np_in, ptcls_per_elem_v, element_gids_v);
  }
  else if (structure == 2) {
    ptcls = new ps::CabM<Types, MemSpace>(po, ne_in, np_in, ptcls_per_elem_v, element_gids_v);
  }
  else if (structure == 3) {
    ptcls = new ps::DPS<Types, MemSpace>(po, ne_in, np_in, ptcls_per_elem_v, element_gids_v);
  }

  delete ptcls;
  
  cudaMemGetInfo(&free, &total);
  const long used_after=total-free;
  if (used_before < used_after) {
    fprintf(stderr, "[ERROR] %s has allocated too much memory\n", structure_names[structure]);
    fails += 1;
  }
  return fails;
}

bool destroyRebuild(int ne_in, int np_in, int distribution, int structure) {
  int fails = 0;
  
  ps::gid_t* gids = new ps::gid_t[ne_in];
  distribute_elements(ne_in, 0, comm_rank, comm_size, gids);
  int* ptcls_per_elem = new int[ne_in];
  std::vector<int>* ids = new std::vector<int>[ne_in];
  distribute_particles(ne_in, np_in, 2, ptcls_per_elem, ids);
  delete [] ids;

  PS::kkLidView ptcls_per_elem_v("ptcls_per_elem_v", ne_in);
  PS::kkGidView element_gids_v("element_gids_v", ne_in);
  ps::hostToDevice(ptcls_per_elem_v, ptcls_per_elem);
  ps::hostToDevice(element_gids_v, gids);
  delete [] ptcls_per_elem;
  delete [] gids;
  Kokkos::TeamPolicy<ExeSpace> po(4, 32);

  // check rebuild doesn't allocate extra memory
  PS* ptcls;
  if (structure == 0) {
    ptcls = new ps::SellCSigma<Types, MemSpace>(po, 5,2, ne_in, np_in, ptcls_per_elem_v, element_gids_v);
  }
  else if (structure == 1) {
    ptcls = new ps::CSR<Types, MemSpace>(po, ne_in, np_in, ptcls_per_elem_v, element_gids_v);
  }
  else if (structure == 2) {
    ptcls = new ps::CabM<Types, MemSpace>(po, ne_in, np_in, ptcls_per_elem_v, element_gids_v);
  }
  else if (structure == 3) {
    ptcls = new ps::DPS<Types, MemSpace>(po, ne_in, np_in, ptcls_per_elem_v, element_gids_v);
  }
  PS::kkLidView new_element("new_element", ptcls->capacity());
  
  size_t free, total;
  cudaMemGetInfo(&free, &total);
  const long used_before=total-free;

  ptcls->rebuild(new_element);
  
  cudaMemGetInfo(&free, &total);
  const long used_after=total-free;
  if (used_before < used_after) {
    fprintf(stderr, "[ERROR] %s::rebuild has allocated too much memory\n", structure_names[structure]);
    fails += 1;
  }

  delete ptcls;
  return fails;
}

bool destroyMigrate(int ne_in, int np_in, int distribution, int structure) {
  int fails = 0;
  
  ps::gid_t* gids = new ps::gid_t[ne_in];
  distribute_elements(ne_in, 0, comm_rank, comm_size, gids);
  int* ptcls_per_elem = new int[ne_in];
  std::vector<int>* ids = new std::vector<int>[ne_in];
  distribute_particles(ne_in, np_in, 2, ptcls_per_elem, ids);
  delete [] ids;

  PS::kkLidView ptcls_per_elem_v("ptcls_per_elem_v", ne_in);
  PS::kkGidView element_gids_v("element_gids_v", ne_in);
  ps::hostToDevice(ptcls_per_elem_v, ptcls_per_elem);
  ps::hostToDevice(element_gids_v, gids);
  delete [] ptcls_per_elem;
  delete [] gids;
  Kokkos::TeamPolicy<ExeSpace> po(4, 32);

  // check rebuild doesn't allocate extra memory
  PS* ptcls;
  if (structure == 0) {
    ptcls = new ps::SellCSigma<Types, MemSpace>(po, 5,2, ne_in, np_in, ptcls_per_elem_v, element_gids_v);
  }
  else if (structure == 1) {
    ptcls = new ps::CSR<Types, MemSpace>(po, ne_in, np_in, ptcls_per_elem_v, element_gids_v);
  }
  else if (structure == 2) {
    ptcls = new ps::CabM<Types, MemSpace>(po, ne_in, np_in, ptcls_per_elem_v, element_gids_v);
  }
  else if (structure == 3) {
    ptcls = new ps::DPS<Types, MemSpace>(po, ne_in, np_in, ptcls_per_elem_v, element_gids_v);
  }
  PS::kkLidView new_element("new_element", ptcls->capacity()); // just move particles to first element
  PS::kkLidView new_process("new_process", ptcls->capacity()); // just keep particles where they are
  int* comm_rank_h = new int[1];
  comm_rank_h[0] = comm_rank;
  PS::kkLidView comm_rank_d("comm_rank_d", 1);
  ps::hostToDevice(comm_rank_d, comm_rank_h);
  Kokkos::parallel_for("assign_same_process", ptcls->capacity(),
    KOKKOS_LAMBDA(const int& i) {
      new_process(i) = comm_rank_d(0);
    });
  
  size_t free, total;
  cudaMemGetInfo(&free, &total);
  const long used_before=total-free;

  ptcls->migrate(new_element, new_process);
  
  cudaMemGetInfo(&free, &total);
  const long used_after=total-free;
  if (used_before < used_after) {
    fprintf(stderr, "[ERROR] %s::migrate has allocated too much memory\n", structure_names[structure]);
    fails += 1;
  }

  delete ptcls;
  return fails;
}

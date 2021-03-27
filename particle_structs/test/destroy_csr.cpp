#include <stdio.h>
#include <Kokkos_Core.hpp>

#include <particle_structs.hpp>
#include "Distribute.h"
#include <mpi.h>

using particle_structs::MemberTypes;

typedef MemberTypes<int, double[3]> Type;
typedef Kokkos::DefaultExecutionSpace exe_space;
typedef particle_structs::CSR<Type, exe_space> CSR;

int comm_rank, comm_size;

bool destroyConstructor(int ne_in, int np_in, int distribution);
bool destroyRebuildSwap(int ne_in, int np_in, int distribution);
bool destroyMigrateSwap(int ne_in, int np_in, int distribution);

int main(int argc, char* argv[]) {
  Kokkos::initialize(argc, argv);
  MPI_Init(&argc,&argv);

  MPI_Comm_rank(MPI_COMM_WORLD,&comm_rank);
  MPI_Comm_size(MPI_COMM_WORLD,&comm_size);

  int ne = 5;
  int np = 20;
  int distribution = 1;
  int fails = 0;

  fprintf(stderr,"destroyConstructor\n");
  fails += destroyConstructor(ne, np, distribution);
  fprintf(stderr,"destroyRebuildSwap\n");
  fails += destroyRebuildSwap(ne, np, distribution);
  fprintf(stderr,"destroyMigrateSwap\n");
  fails += destroyMigrateSwap(ne, np, distribution);
  
  Kokkos::finalize();
  int total_fails;
  MPI_Reduce(&fails, &total_fails, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
  MPI_Finalize();
  if (comm_rank == 0 && total_fails == 0)
    printf("All tests passed\n");
  return total_fails;
}

bool destroyConstructor(int ne_in, int np_in, int distribution) {
  int fails = 0;

  particle_structs::gid_t* gids = new particle_structs::gid_t[ne_in];
  distribute_elements(ne_in, 0, comm_rank, comm_size, gids);
  int* ptcls_per_elem = new int[ne_in];
  std::vector<int>* ids = new std::vector<int>[ne_in];
  distribute_particles(ne_in, np_in, 2, ptcls_per_elem, ids);
  delete [] ids;

  CSR::kkLidView ptcls_per_elem_v("ptcls_per_elem_v", ne_in);
  CSR::kkGidView element_gids_v("element_gids_v", ne_in);
  particle_structs::hostToDevice(ptcls_per_elem_v, ptcls_per_elem);
  particle_structs::hostToDevice(element_gids_v, gids);
  delete [] ptcls_per_elem;
  delete [] gids;
  Kokkos::TeamPolicy<exe_space> po(4, 32);
  
  // create and destroy structure
  size_t free, total;
  cudaMemGetInfo(&free, &total);
  const long used_before=total-free;

  CSR* csr = new CSR(po, ne_in, np_in, ptcls_per_elem_v, element_gids_v);
  delete csr;
  
  cudaMemGetInfo(&free, &total);
  const long used_after=total-free;
  if (used_before != used_after) {
    fprintf(stderr, "[ERROR] CSR has allocated too much memory\n");
    fails += 1;
  }
  return fails;
}

bool destroyRebuildSwap(int ne_in, int np_in, int distribution) {
  int fails = 0;
  
  particle_structs::gid_t* gids = new particle_structs::gid_t[ne_in];
  distribute_elements(ne_in, 0, comm_rank, comm_size, gids);
  int* ptcls_per_elem = new int[ne_in];
  std::vector<int>* ids = new std::vector<int>[ne_in];
  distribute_particles(ne_in, np_in, 2, ptcls_per_elem, ids);
  delete [] ids;

  CSR::kkLidView ptcls_per_elem_v("ptcls_per_elem_v", ne_in);
  CSR::kkGidView element_gids_v("element_gids_v", ne_in);
  particle_structs::hostToDevice(ptcls_per_elem_v, ptcls_per_elem);
  particle_structs::hostToDevice(element_gids_v, gids);
  delete [] ptcls_per_elem;
  delete [] gids;
  Kokkos::TeamPolicy<exe_space> po(4, 32);

  // check rebuild doesn't allocate extra memory
  CSR* csr = new CSR(po, ne_in, np_in, ptcls_per_elem_v, element_gids_v);
  CSR::kkLidView new_element("new_element", csr->capacity());
  
  size_t free, total;
  cudaMemGetInfo(&free, &total);
  const long used_before=total-free;

  csr->rebuild(new_element);
  
  cudaMemGetInfo(&free, &total);
  const long used_after=total-free;
  if (used_before != used_after) {
    fprintf(stderr, "[ERROR] CSR::rebuild has allocated too much memory\n");
    fails += 1;
  }

  delete csr;
  return fails;
}

bool destroyMigrateSwap(int ne_in, int np_in, int distribution) {
  int fails = 0;
  
  particle_structs::gid_t* gids = new particle_structs::gid_t[ne_in];
  distribute_elements(ne_in, 0, comm_rank, comm_size, gids);
  int* ptcls_per_elem = new int[ne_in];
  std::vector<int>* ids = new std::vector<int>[ne_in];
  distribute_particles(ne_in, np_in, 2, ptcls_per_elem, ids);
  delete [] ids;

  CSR::kkLidView ptcls_per_elem_v("ptcls_per_elem_v", ne_in);
  CSR::kkGidView element_gids_v("element_gids_v", ne_in);
  particle_structs::hostToDevice(ptcls_per_elem_v, ptcls_per_elem);
  particle_structs::hostToDevice(element_gids_v, gids);
  delete [] ptcls_per_elem;
  delete [] gids;
  Kokkos::TeamPolicy<exe_space> po(4, 32);

  // check rebuild doesn't allocate extra memory
  CSR* csr = new CSR(po, ne_in, np_in, ptcls_per_elem_v, element_gids_v);
  CSR::kkLidView new_element("new_element", csr->capacity()); // just move particles to first element
  CSR::kkLidView new_process("new_process", csr->capacity()); // just keep particles where they are
  int* comm_rank_h = new int[1];
  comm_rank_h[0] = comm_rank;
  CSR::kkLidView comm_rank_d("comm_rank_d", 1);
  particle_structs::hostToDevice(comm_rank_d, comm_rank_h);
  Kokkos::parallel_for("assign_same_process", csr->capacity(),
    KOKKOS_LAMBDA(const int& i) {
      new_process(i) = comm_rank_d(0);
    });
  
  size_t free, total;
  cudaMemGetInfo(&free, &total);
  const long used_before=total-free;

  csr->migrate(new_element, new_process);
  
  cudaMemGetInfo(&free, &total);
  const long used_after=total-free;
  if (used_before != used_after) {
    fprintf(stderr, "[ERROR] CSR::rebuild has allocated too much memory\n");
    fails += 1;
  }

  delete csr;
  return fails;
}
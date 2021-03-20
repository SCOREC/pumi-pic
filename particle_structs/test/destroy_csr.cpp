#include <stdio.h>
#include <Kokkos_Core.hpp>

#include <particle_structs.hpp>
#include "Distribute.h"
#include <mpi.h>

using particle_structs::MemberTypes;

typedef MemberTypes<int, double[3]> Type;
typedef Kokkos::DefaultExecutionSpace exe_space;
typedef particle_structs::CSR<Type, exe_space> CSR;

int main(int argc, char* argv[]) {
  Kokkos::initialize(argc, argv);
  MPI_Init(&argc, &argv);

  int comm_rank;
  int comm_size;
  MPI_Comm_rank(MPI_COMM_WORLD, &comm_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &comm_size);

  int ITER = 100;

  int ne = 5;
  int np = 20;
  int fails = 0;
  particle_structs::gid_t* gids = new particle_structs::gid_t[ne];
  distribute_elements(ne, 0, comm_rank, comm_size, gids);
  int* ptcls_per_elem = new int[ne];
  std::vector<int>* ids = new std::vector<int>[ne];
  distribute_particles(ne, np, 2, ptcls_per_elem, ids);
  delete [] ids;
  {
    CSR::kkLidView ptcls_per_elem_v("ptcls_per_elem_v", ne);
    CSR::kkGidView element_gids_v("element_gids_v", ne);
    particle_structs::hostToDevice(ptcls_per_elem_v, ptcls_per_elem);
    particle_structs::hostToDevice(element_gids_v, gids);
    delete [] ptcls_per_elem;
    delete [] gids;
    Kokkos::TeamPolicy<exe_space> po(4, 32);
    
    // create and destroy structure
    for (int i = 0; i < ITER; i++) {
      size_t free, total;
      cudaMemGetInfo(&free, &total);
      const long used_before=total-free;

      CSR* csr = new CSR(po, ne, np, ptcls_per_elem_v, element_gids_v);
      delete csr;
      
      cudaMemGetInfo(&free, &total);
      const long used_after=total-free;

      assert(used_before == used_after);
    }

  }

  Kokkos::finalize();
  int total_fails;
  MPI_Reduce(&fails, &total_fails, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
  MPI_Finalize();
  if (comm_rank == 0 && total_fails == 0)
    printf("All tests passed\n");
  return fails;
}
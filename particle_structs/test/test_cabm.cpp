#include <particle_structs.hpp>
#include "read_particles.hpp"

#ifdef PP_USE_CUDA
typedef Kokkos::CudaSpace DeviceSpace;
#else
typedef Kokkos::HostSpace DeviceSpace;
#endif
void finalize() {
  Kokkos::finalize();
  MPI_Finalize();
}

int comm_rank, comm_size;
int testCounts(PS* structure, lid_t num_elems, lid_t num_ptcls) {
  int fails = 0;
  if (structure->nElems() != num_elems) {
    fprintf(stderr, "[ERROR] Element count mismatch on rank %d "
            "[(structure)%d != %d(actual)]\n",
            comm_rank, structure->nElems(), num_elems);
    ++fails;
  }
  if (structure->nPtcls() != num_ptcls) {
    fprintf(stderr, "[ERROR] Particle count mismatch on rank %d "
            "[(structure)%d != %d(actual)]\n",
            comm_rank, structure->nPtcls(), num_ptcls);
    ++fails;
  }
  if (structure->numRows() < num_elems) {
    fprintf(stderr, "[ERROR] Number of rows is too small to fit elements on rank %d "
            "[(structure)%d < %d(actual)]\n", comm_rank,
            structure->numRows(), num_elems);
    ++fails;
  }
  if (structure->capacity() < num_ptcls) {
    fprintf(stderr, "[ERROR] Capcity is too small to fit particles on rank %d "
            "[(structure)%d < %d(actual)]\n", comm_rank,
            structure->capacity(), num_ptcls);
    ++fails;
  }
  return fails;
}

//Functionality tests
int testMetrics(PS* structure) {
  int fails = 0;
  try {
    structure->printMetrics();
  }
  catch(...) {
    fprintf(stderr, "[ERROR] Failed running printMetrics() on rank %d\n",
            comm_rank);
    ++fails;
  }
  return fails;
}

int main(int argc, char* argv[]) {
  Kokkos::initialize(argc, argv);
  MPI_Init(&argc, &argv);

  //Local count of fails
  int fails = 0;
  {
  MPI_Comm_rank(MPI_COMM_WORLD, &comm_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &comm_size);

  if (argc != 2) {
    if (!comm_rank)
      fprintf(stdout, "[ERROR] Format: %s <particle_file_prefix>\n", argv[0]);
    finalize();
    return 0;
  }

  char filename[256];
  sprintf(filename, "%s_%d.ptl", argv[1], comm_rank);
  //General structure parameters
  lid_t num_elems;
  lid_t num_ptcls;
  kkLidView ppe;
  kkGidView element_gids;
  kkLidView particle_elements;
  PS::MTVs particle_info;
  readParticles(filename, num_elems, num_ptcls, ppe, element_gids,
                particle_elements, particle_info);

  Kokkos::TeamPolicy<ExeSpace> policy(num_elems,32); //league_size, team_size
  /*ps::CabM<Types,MemSpace>* cabm = new ps::CabM<Types, MemSpace>(policy, num_elems, num_ptcls, 
                                      ppe, element_gids, particle_elements, particle_info);*/
  ps::CabM<Types,MemSpace>* cabm = new ps::CabM<Types, MemSpace>(policy, num_elems, num_ptcls, 
                                      ppe, element_gids); // temporary cabm without adding particles

  // manually add particle data
  /* won't work because don't want to expose AoSoA and get not for CabM
  auto aosoa = cabm.getAoSoA();
  cabm.parallel_for( PS_LAMBDA(const lid_t elm_id, const lid_t ptcl_id, bool mask) {
      /// @todo add content
  
  };)*/

  //insert parallel_for to copy data from MTV into cabm object - crappy pseudo code below
//  foo = MTV<0>.get(); //device array for the first member type
//  ourSlice = cabm<0>.get(); //device array for our storage for the first type
//  parallel_for(...., int e, int p) {
//    if(e == 0 && p < 5) {
//      ourSlice[p] = foo[p];
//    }
//  }

  //Run tests
  fails += testCounts(cabm, num_elems, num_ptcls);
  fails += testMetrics(cabm);

  //Cleanup
  ps::destroyViews<Types>(particle_info);

  //Finalize and print failures
  if (comm_rank == 0) {
    if(fails == 0)
      printf("All tests passed\n");
    else
      printf("%d tests failed\n", fails);
  }
  }
  finalize();
  return fails;
}

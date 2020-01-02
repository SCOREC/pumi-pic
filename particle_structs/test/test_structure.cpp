#include <particle_structs.hpp>
#include "read_particles.hpp"

void finalize() {
  Kokkos::finalize();
  MPI_Finalize();
}

int addSCSs(std::vector<PS*>& structures, lid_t num_elems, lid_t num_ptcls, kkLidView ppe,
            kkGidView element_gids, kkLidView particle_elements, PS::MTVs particle_info);

int main(int argc, char* argv[]) {
  Kokkos::initialize(argc, argv);
  MPI_Init(&argc, &argv);

  int comm_rank, comm_size;
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
  //Local count of fails
  int fails = 0;
  {
    //Vector of structures to run all the tests on
    std::vector<PS*> structures;

    //General structure parameters
    lid_t num_elems;
    lid_t num_ptcls;
    kkLidView ppe;
    kkGidView element_gids;
    kkLidView particle_elements;
    PS::MTVs particle_info;
    readParticles(filename, num_elems, num_ptcls, ppe, element_gids,
                  particle_elements, particle_info);

    fails += addSCSs(structures, num_elems, num_ptcls, ppe, element_gids,
                     particle_elements, particle_info);

    //Run each structure on every test


    //Cleanup
    for (size_t i = 0; i < structures.size(); ++i)
      delete structures[i];
    structures.clear();
  }
  //Finalize and print failures
  int total_fails = 0;
  MPI_Reduce(&fails, &total_fails, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
  finalize();
  if (comm_rank == 0) {
    if(total_fails == 0)
      printf("All tests passed\n");
    else
      printf("%d tests failed\n", total_fails);
  }
  return total_fails;

}

int addSCSs(std::vector<PS*>& structures, lid_t num_elems, lid_t num_ptcls, kkLidView ppe,
            kkGidView element_gids, kkLidView particle_elements, PS::MTVs particle_info) {
  int comm_rank, comm_size;
  MPI_Comm_rank(MPI_COMM_WORLD, &comm_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &comm_size);

  int fails = 0;
  //Build SCS with C = 32, sigma = ne, V = 1024
  try {
    lid_t maxC = 32;
    lid_t sigma = num_elems;
    lid_t V = 1024;
    Kokkos::TeamPolicy<ExeSpace> policy(4, maxC);
    PS* s = new ps::SellCSigma<Types, MemSpace>(policy, sigma, V, num_elems, num_ptcls, ppe,
                                                element_gids, particle_elements, particle_info);
    structures.push_back(s);
  }
  catch(...) {
    fprintf(stderr, "[ERROR] Construction of SCS (C=32, sigma=ne, V=1024) failed on rank %d\n",
            comm_rank);
    ++fails;
  }

  //Build SCS with C = 32, sigma = 1, V = 10
  try {
    lid_t maxC = 32;
    lid_t sigma = 1;
    lid_t V = 10;
    Kokkos::TeamPolicy<ExeSpace> policy(4, maxC);
    PS* s = new ps::SellCSigma<Types, MemSpace>(policy, sigma, V, num_elems, num_ptcls, ppe,
                                                element_gids, particle_elements, particle_info);
    structures.push_back(s);
  }
  catch(...) {
    fprintf(stderr, "[ERROR] Construction of SCS (C=32, sigma=1, V=10) failed on rank %d\n",
            comm_rank);
    ++fails;
  }
  return fails;
}

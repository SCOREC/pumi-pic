#include "read_particles.hpp"
#include "Distribute.h"
void finalize() {
  Kokkos::finalize();
  MPI_Finalize();
}

int main(int argc, char* argv[]) {
  Kokkos::initialize(argc, argv);
  MPI_Init(&argc, &argv);

  int comm_rank, comm_size;
  MPI_Comm_rank(MPI_COMM_WORLD, &comm_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &comm_size);

  if (argc != 6) {
    if (!comm_rank)
      fprintf(stderr, "[ERROR] Format: %s <num_elems> <num_ptcls> <element strat> "
              "<ptcl strat> <particle_file_prefix>\n", argv[0]);
    finalize();
    return 1;
  }
  int num_elems = atoi(argv[1]);
  int num_ptcls = atoi(argv[2]);
  int elem_strat = atoi(argv[3]);
  int ptcl_strat = atoi(argv[4]);

  #ifndef PP_USE_CUDA
  num_ptcls = std::min(num_ptcls, 100000);
  #endif

  {
    ps::gid_t* gids = new ps::gid_t[num_elems];
    distribute_elements(num_elems, elem_strat, gids);

    int* ppe = new int[num_elems];
    std::vector<int>* ids = new std::vector<int>[num_elems];
    distribute_particles(num_elems, num_ptcls, ptcl_strat, ppe, ids);

    int* pElems = new int[num_ptcls];
    for (int i = 0; i < num_elems; ++i)
      for (int j = 0; j < ids[i].size(); ++j)
        pElems[ids[i][j]] = i;
    kkLidView ppe_k("ppe", num_elems);
    kkGidView eGids_k("element gids", num_elems);
    kkLidView pElems_k("particle elements", num_ptcls);
    auto particle_info = ps::createMemberViews<Types>(num_ptcls);

    ps::hostToDevice(ppe_k, ppe);
    ps::hostToDevice(eGids_k, gids);
    ps::hostToDevice(pElems_k, pElems);
    delete [] gids;
    delete [] ppe;
    delete [] ids;
    delete [] pElems;

    auto pids = ps::getMemberView<Types, 0>(particle_info);
    auto val1 = ps::getMemberView<Types, 1>(particle_info);
    auto val2 = ps::getMemberView<Types, 2>(particle_info);
    auto val3 = ps::getMemberView<Types, 3>(particle_info);
    Kokkos::parallel_for(num_ptcls, KOKKOS_LAMBDA(const int &i) {
      pids(i) = i;
      for (int j = 0; j < 3; ++j) {
        val1(i,j) = 0;
      }
      val2(i) = 0;
      val3(i) = 0;
    });

    char filename[256];
    sprintf(filename, "%s_%d.ptl", argv[5], comm_rank);
    writeParticles(filename, num_elems, num_ptcls, ppe_k, eGids_k, pElems_k, particle_info);

  }
  MPI_Finalize();
  Kokkos::finalize();
  return 0;
}

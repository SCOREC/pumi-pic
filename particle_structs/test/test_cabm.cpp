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

int testParticleExistence(PS* structure, lid_t num_ptcls) {
  int fails = 0;
  kkLidView count("count", 1);
  auto checkExistence = PS_LAMBDA(const lid_t& e, const lid_t& p, const bool& mask) {
    Kokkos::atomic_fetch_add(&(count(0)), mask);
  };
  ps::parallel_for(structure, checkExistence, "check particle existence");
  lid_t c = ps::getLastValue<lid_t>(count);
  if (c != num_ptcls) {
    fprintf(stderr, "[ERROR] Number of particles found in parallel_for "
            "does not match the number of particles on rank %d"
            "[(parallel_for)%d != %d(actual)]]n", comm_rank,
            c, num_ptcls);
    ++fails;
  }
  return fails;
}

int setValues(PS* structure) {
  int fails = 0;
  auto dbls = structure->get<1>();
  auto bools = structure->get<2>();
  auto nums = structure->get<3>();
  int local_rank = comm_rank;
  auto setValues = PS_LAMBDA(const lid_t& e, const lid_t& p, const bool& mask) {
    if (mask) {
      dbls(p, 0) = p * e * 100.0;
      dbls(p, 1) = M_PI * p + M_PI / 2.0;
      dbls(p, 2) = M_E * 2.5;
      nums(p) = local_rank;
      bools(p) = true;
    }
    else {
      dbls(p, 0) = 0;
      dbls(p, 1) = 0;
      dbls(p, 2) = 0;
      nums(p) = -1;
      bools(p) = false;
    }
  };
  ps::parallel_for(structure, setValues, "setValues");
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

int testSegmentComp(PS* structure) {
  int fails = 0;
  kkLidView failures("fails", 1);

  auto dbls = structure->get<1>();
  auto setComponents = PS_LAMBDA(const lid_t e, const lid_t p, const bool mask) {
    auto dbl_seg = dbls.getComponents(p);
    for (int i = 0; i < 3; ++i)
      dbl_seg(i) = e * (i + 1);
  };
  pumipic::parallel_for(structure, setComponents, "Set components");

  const double TOL = .00001;
  auto checkComponents = PS_LAMBDA(const lid_t e, const lid_t p, const bool mask) {
    auto comps = dbls.getComponents(p);
    for (int i = 0; i < 3; ++i) {
      if (abs(comps[i] - e * (i + 1)) > TOL) {
        printf("[ERROR] component is wrong on ptcl %d comp %d (%.3f != %d)\n",
               p, i, comps[i], e * (i + 1));
        Kokkos::atomic_add(&(failures[0]), 1);
      }
    }
  };
  pumipic::parallel_for(structure, checkComponents, "Check components");
  fails += pumipic::getLastValue<lid_t>(failures);

  return fails;
}

int checkPtclElem(ps::CabM<Types>* structure, kkLidView particle_elements){
  //Fail count and accessing structure member variables
  int fails = 0;
  kkLidView failures("fails", 1);
  auto pIDs = structure->get<0>();

  //Parallel_for checking that all particles are in the correct
  //element (row) inside structure (rows indexed by offsets)
  auto checkPtclElems = PS_LAMBDA(const lid_t& elm, const lid_t& ptcl, const bool& mask) {
    lid_t id = pIDs(ptcl);
    if ( (mask == 1) && (particle_elements[id] != elm) ) {
      failures[0] += 1;
      printf("Particle %d\t assigned to incorrect element %d (should be element %d)\n", id, elm, particle_elements[id]);
    }
  };
  structure->parallel_for(checkPtclElems, "checkPtclElems");
  
  fails += pumipic::getLastValue<lid_t>(failures);
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
  ps::CabM<Types,MemSpace>* cabm = new ps::CabM<Types, MemSpace>(policy, num_elems, num_ptcls, 
                                      ppe, element_gids, particle_elements, particle_info);

  //Run tests
  fails += testCounts(cabm, num_elems, num_ptcls);
  fails += testParticleExistence(cabm, num_ptcls);
  fails += setValues(cabm);
  fails += testMetrics(cabm);
  fails += testSegmentComp(cabm);
  fails += checkPtclElem(cabm, particle_elements);

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

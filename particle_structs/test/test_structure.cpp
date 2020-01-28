#include <particle_structs.hpp>
#include "read_particles.hpp"

void finalize() {
  Kokkos::finalize();
  MPI_Finalize();
}

int comm_rank, comm_size;
//Structure adding functions
int addSCSs(std::vector<PS*>& structures, std::vector<std::string>& names,
            lid_t num_elems, lid_t num_ptcls, kkLidView ppe,
            kkGidView element_gids, kkLidView particle_elements, PS::MTVs particle_info);
int addCSRs(std::vector<PS*>& structures, std::vector<std::string>& names,
            lid_t num_elems, lid_t num_ptcls, kkLidView ppe,
            kkGidView element_gids, kkLidView particle_elements, PS::MTVs particle_info);

//Simple tests of construction
int testCounts(const char* name, PS* structure, lid_t num_elems, lid_t num_ptcls);
int testParticleExistence(const char* name, PS* structure, lid_t num_ptcls);
int setValues(const char* name, PS* structure);

//Functionality tests
int testRebuild(const char* name, PS* structure);
int testMigration(const char* name, PS* structure);
int testMetrics(const char* name, PS* structure);
int testCopy(const char* name, PS* structure);

int main(int argc, char* argv[]) {
  Kokkos::initialize(argc, argv);
  MPI_Init(&argc, &argv);

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
    //Vector of names for each structure
    std::vector<std::string> names;

    //General structure parameters
    lid_t num_elems;
    lid_t num_ptcls;
    kkLidView ppe;
    kkGidView element_gids;
    kkLidView particle_elements;
    PS::MTVs particle_info;
    readParticles(filename, num_elems, num_ptcls, ppe, element_gids,
                  particle_elements, particle_info);

    //Add SCS
    fails += addSCSs(structures, names, num_elems, num_ptcls, ppe, element_gids,
                     particle_elements, particle_info);
    //Add CSR
    /* Uncomment when CSR is being implemented
    fails += addCSRs(structures, names, num_elems, num_ptcls, ppe, element_gids,
                     particle_elements, particle_info);
    */



    //Run each structure on every test
    for (int i = 0; i < structures.size(); ++i) {
      fails += testCounts(names[i].c_str(), structures[i], num_elems, num_ptcls);
      fails += testParticleExistence(names[i].c_str(), structures[i], num_ptcls);
      fails += setValues(names[i].c_str(), structures[i]);
      fails += testMetrics(names[i].c_str(), structures[i]);
      fails += testCopy(names[i].c_str(), structures[i]);
    }

    //Cleanup
    ps::destroyViews<Types>(particle_info);
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

int addSCSs(std::vector<PS*>& structures, std::vector<std::string>& names,
            lid_t num_elems, lid_t num_ptcls, kkLidView ppe,
            kkGidView element_gids, kkLidView particle_elements, PS::MTVs particle_info) {
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
    names.push_back("scs_C32_SMAX_V1024");
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
    names.push_back("scs_C32_S1_V10");
  }
  catch(...) {
    fprintf(stderr, "[ERROR] Construction of SCS (C=32, sigma=1, V=10) failed on rank %d\n",
            comm_rank);
    ++fails;
  }
  return fails;
}

int addCSRs(std::vector<PS*>& structures, std::vector<std::string>& names,
            lid_t num_elems, lid_t num_ptcls, kkLidView ppe,
            kkGidView element_gids, kkLidView particle_elements, PS::MTVs particle_info) {
  int fails = 0;
  try {
    PS* s = new ps::CSR<Types, MemSpace>(num_elems, num_ptcls, ppe,
                                         element_gids, particle_elements, particle_info);
    structures.push_back(s);
    names.push_back("csr");
  }
  catch(...) {
    fprintf(stderr, "[ERROR] Construction of CSR failed on rank %d\n", comm_rank);
    ++fails;
  }
  return fails;
}


int testCounts(const char* name, PS* structure, lid_t num_elems, lid_t num_ptcls) {
  int fails = 0;
  if (structure->nElems() != num_elems) {
    fprintf(stderr, "[ERROR] Test %s: Element count mismatch on rank %d "
            "[(structure)%d != %d(actual)]\n", name,
            comm_rank, structure->nElems(), num_elems);
    ++fails;
  }
  if (structure->nPtcls() != num_ptcls) {
    fprintf(stderr, "[ERROR] Test %s: Particle count mismatch on rank %d "
            "[(structure)%d != %d(actual)]\n", name,
            comm_rank, structure->nPtcls(), num_ptcls);
    ++fails;
  }
  if (structure->numRows() < num_elems) {
    fprintf(stderr, "[ERROR] Test %s: Number of rows is too small to fit elements on rank %d "
            "[(structure)%d < %d(actual)]\n", name, comm_rank,
            structure->numRows(), num_elems);
    ++fails;
  }
  if (structure->capacity() < num_ptcls) {
    fprintf(stderr, "[ERROR] Test %s: Capcity is too small to fit particles on rank %d "
            "[(structure)%d < %d(actual)]\n", name, comm_rank,
            structure->capacity(), num_ptcls);
    ++fails;
  }
  return fails;
}
int testParticleExistence(const char* name, PS* structure, lid_t num_ptcls) {
  int fails = 0;
  kkLidView count("count", 1);
  auto checkExistence = PS_LAMBDA(const lid_t& e, const lid_t& p, const bool& mask) {
    Kokkos::atomic_fetch_add(&(count(0)), mask);
  };
  ps::parallel_for(structure, checkExistence, "check particle existence");
  lid_t c = ps::getLastValue<lid_t>(count);
  if (c != num_ptcls) {
    fprintf(stderr, "[ERROR] Test %s: Number of particles found in parallel_for "
            "does not match the number of particles on rank %d"
            "[(parallel_for)%d != %d(actual)]]n", name, comm_rank,
            c, num_ptcls);
    ++fails;
  }
  return fails;
}

int setValues(const char* name, PS* structure) {
  int fails = 0;
  auto dbls = structure->get<1>();
  auto nums = structure->get<2>();
  auto bools = structure->get<3>();
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
int testRebuild(const char* name, PS* structure) {
  int fails = 0;
  return fails;
}
int testMigration(const char* name, PS* structure) {
  int fails = 0;
  return fails;
}
int testMetrics(const char* name,PS* structure) {
  int fails = 0;
  try {
    structure->printMetrics();
  }
  catch(...) {
    fprintf(stderr, "[ERROR] Test %s: Failed running printMetrics() on rank %d\n",
            name, comm_rank);
    ++fails;
  }
  return fails;
}

int testCopy(const char* name, PS* structure) {
  int fails = 0;
  //Copy particle structure to the host
  PS::Mirror<Kokkos::HostSpace>* host_structure = ps::copy<Kokkos::HostSpace>(structure);
  if (host_structure->nElems() != structure->nElems()) {
    fprintf(stderr, "[ERROR] Test %s: Failed to copy nElems() on rank %d\n",
            name, comm_rank);
    ++fails;
  }
  if (host_structure->nPtcls() != structure->nPtcls()) {
    fprintf(stderr, "[ERROR] Test %s: Failed to copy nPtcls() on rank %d\n",
            name, comm_rank);
    ++fails;
  }
  if (host_structure->capacity() != structure->capacity()) {
    fprintf(stderr, "[ERROR] Test %s: Failed to copy capacity() on rank %d\n",
            name, comm_rank);
    ++fails;
  }
  if (host_structure->numRows() != structure->numRows()) {
    fprintf(stderr, "[ERROR] Test %s: Failed to copy numRows() on rank %d\n",
            name, comm_rank);
    ++fails;
  }
  //Copy particle structure back to the device
  PS* device_structure = ps::copy<Kokkos::CudaSpace>(host_structure);
  if (device_structure->nElems() != structure->nElems()) {
    fprintf(stderr, "[ERROR] Test %s: Failed to copy nElems() back to device on rank %d\n",
            name, comm_rank);
    ++fails;
  }
  if (device_structure->nPtcls() != structure->nPtcls()) {
    fprintf(stderr, "[ERROR] Test %s: Failed to copy nPtcls() back to device on rank %d\n",
            name, comm_rank);
    ++fails;
  }
  if (device_structure->capacity() != structure->capacity()) {
    fprintf(stderr, "[ERROR] Test %s: Failed to copy capacity() back to device on rank %d\n",
            name, comm_rank);
    ++fails;
  }
  if (device_structure->numRows() != structure->numRows()) {
    fprintf(stderr, "[ERROR] Test %s: Failed to copy numRows() back to device on rank %d\n",
            name, comm_rank);
    ++fails;
  }
  //Compare original and new particle structure on the device
  auto ids1 = structure->get<0>();
  auto ids2 = device_structure->get<0>();
  auto dbls1 = structure->get<1>();
  auto dbls2 = device_structure->get<1>();
  double EPSILON = .00001;
  kkLidView failure("failure", 1);
  int local_rank = comm_rank;
  auto testTypes = PS_LAMBDA(const int& eid, const int& pid, const bool mask) {
    if (mask) {
      if (ids1(pid) != ids2(pid)) {
        printf("[ERROR] Test %s: Particle ids do not match for particle %d "
               "[(old) %d != %d (copy)] on rank %d\n", name, pid, ids1(pid),
               ids2(pid), local_rank);
        failure(0) = 1;
      }
      for (int i = 0; i < 3; ++i)
        if (fabs(dbls1(pid,i) - dbls2(pid,i)) > EPSILON) {
          printf("[ERROR] Test %s: Particle's dbl %d does not match for particle %d"
                 "[(old) %.4f != %.4f (copy)] on rank %d\n", name, i, pid, dbls1(pid,i),
                 dbls2(pid,i), local_rank);
          failure(0) = 1;
      }
    }
  };
  ps::parallel_for(structure, testTypes, "testTypes on original structure");
  if (ps::getLastValue<lid_t>(failure)) {
    fprintf(stderr, "[ERROR] Test %s: Parallel for on original structure had failures\n",
            name);
    ++fails;
  }

  Kokkos::parallel_for(1, KOKKOS_LAMBDA(const int& i) {failure(i) = 0;});
  ps::parallel_for(device_structure, testTypes, "testTypes on copy of structure");
  if (ps::getLastValue<lid_t>(failure)) {
    fprintf(stderr, "[ERROR] Test %s: Parallel for on device structure had failures\n",
            name);
    ++fails;
  }
  return fails;
}

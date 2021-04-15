#include <particle_structs.hpp>
#include "read_particles.hpp"
#include <cmath>

int comm_rank, comm_size;

#ifdef PP_USE_CUDA
typedef Kokkos::CudaSpace DeviceSpace;
#else
typedef Kokkos::HostSpace DeviceSpace;
#endif
void finalize() {
  Kokkos::finalize();
  MPI_Finalize();
}

//Structure adding functions
int addSCSs(std::vector<PS*>& structures, std::vector<std::string>& names,
            lid_t num_elems, lid_t num_ptcls, kkLidView ppe,
            kkGidView element_gids, kkLidView particle_elements, PS::MTVs particle_info);
int addCSRs(std::vector<PS*>& structures, std::vector<std::string>& names,
            lid_t num_elems, lid_t num_ptcls, kkLidView ppe,
            kkGidView element_gids, kkLidView particle_elements, PS::MTVs particle_info);
int addCabMs(std::vector<PS*>& structures, std::vector<std::string>& names,
            lid_t num_elems, lid_t num_ptcls, kkLidView ppe,
            kkGidView element_gids, kkLidView particle_elements, PS::MTVs particle_info);

//Simple tests of constructors
int testCounts(const char* name, PS* structure, lid_t num_elems, lid_t num_ptcls);
int testParticleExistence(const char* name, PS* structure, lid_t num_ptcls);
int setValues(const char* name, PS* structure);
int pseudoPush(const char* name, PS* structure);

//Functionality tests
int testRebuild(const char* name, PS* structure);
int rebuildNoChanges(const char* name, PS* structure);
int rebuildNewElems(const char* name, PS* structure);
int rebuildNewPtcls(const char* name, PS* structure);
int rebuildPtclsDestroyed(const char* name, PS* structure);
int rebuildNewAndDestroyed(const char* name, PS* structure);

int testMigration(const char* name, PS* structure);
int migrateSendRight(const char* name, PS* structure);
int migrateSendToOne(const char* name, PS* structure);

int testMetrics(const char* name, PS* structure);
int testCopy(const char* name, PS* structure);
int testSegmentComp(const char* name, PS* structure);

//Edge Case tests
int migrateToEmptyAndRefill(const char* name, PS* structure);

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
    fails += addCSRs(structures, names, num_elems, num_ptcls, ppe, element_gids,
                     particle_elements, particle_info);
    //Add CabM
#ifdef PP_ENABLE_CABM
    fails += addCabMs(structures, names, num_elems, num_ptcls, ppe, element_gids,
                     particle_elements, particle_info);
#endif

    //Run each structure on every test
    for (int i = 0; i < structures.size(); ++i) {
      fails += testCounts(names[i].c_str(), structures[i], num_elems, num_ptcls);
      fails += testParticleExistence(names[i].c_str(), structures[i], num_ptcls);
      fails += setValues(names[i].c_str(), structures[i]);
      Kokkos::fence();
      fails += pseudoPush(names[i].c_str(), structures[i]);
      Kokkos::fence();
      fails += testMetrics(names[i].c_str(), structures[i]);
      fails += testRebuild(names[i].c_str(), structures[i]);
      fails += testMigration(names[i].c_str(), structures[i]);
      //fails += testCopy(names[i].c_str(), structures[i]);
      fails += testSegmentComp(names[i].c_str(), structures[i]);
      fails += migrateToEmptyAndRefill(names[i].c_str(), structures[i]);
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
    if (total_fails == 0)
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
  return fails;
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
    Kokkos::TeamPolicy<ExeSpace> policy(num_elems,32);
    PS* s = new ps::CSR<Types, MemSpace>(policy, num_elems, num_ptcls, ppe,
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

int addCabMs(std::vector<PS*>& structures, std::vector<std::string>& names,
            lid_t num_elems, lid_t num_ptcls, kkLidView ppe,
            kkGidView element_gids, kkLidView particle_elements, PS::MTVs particle_info) {
  int fails = 0;
  try {
    Kokkos::TeamPolicy<ExeSpace> policy(num_elems,32);
    PS* s = new ps::CabM<Types, MemSpace>(policy, num_elems, num_ptcls, ppe,
                                         element_gids, particle_elements, particle_info);
    structures.push_back(s);
    names.push_back("cabm");
  }
  catch(...) {
    fprintf(stderr, "[ERROR] Construction of CabM failed on rank %d\n", comm_rank);
    ++fails;
  }
  return fails;
}


//Functionality tests
int testRebuild(const char* name, PS* structure) {
  printf("testRebuild %s, rank %d\n", name, comm_rank);

  int fails = 0;
  fails += rebuildNoChanges(name, structure);
  fails += rebuildNewElems(name, structure);
  fails += rebuildNewPtcls(name, structure);
  fails += rebuildPtclsDestroyed(name, structure);
  fails += rebuildNewAndDestroyed(name, structure);

  return fails;
}

int testMigration(const char* name, PS* structure) {
  printf("testMigration %s, rank %d\n", name, comm_rank);

  int fails = 0;
  fails += migrateSendRight(name, structure);
  fails += migrateSendToOne(name, structure);

  return fails;
}

int testMetrics(const char* name, PS* structure) {
  printf("testMetrics %s, rank %d\n", name, comm_rank);

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
  printf("testCopy %s\n", name);

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
  PS* device_structure = ps::copy<DeviceSpace>(host_structure);
  delete host_structure;
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
  auto testTypes = PS_LAMBDA(const int& eid, const int& pid, const bool& mask) {
    if (mask) {
      if (ids1(pid) != ids2(pid)) {
        printf("[ERROR] Particle ids do not match for particle %d "
               "[(old) %d != %d (copy)] on rank %d\n", pid, ids1(pid),
               ids2(pid), local_rank);
        failure(0) = 1;
      }
      for (int i = 0; i < 3; ++i)
        if (fabs(dbls1(pid,i) - dbls2(pid,i)) > EPSILON) {
          printf("[ERROR] Particle's dbl %d does not match for particle %d"
                 "[(old) %.4f != %.4f (copy)] on rank %d\n", i, pid, dbls1(pid,i),
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
  delete device_structure;
  return fails;
}

int testSegmentComp(const char* name, PS* structure) {
  printf("testSegmentComp %s, rank %d\n", name, comm_rank);

  int fails = 0;
  kkLidView failures("fails", 1);

  auto dbls = structure->get<1>();
  auto setComponents = PS_LAMBDA(const lid_t& e, const lid_t& p, const bool& mask) {
    auto dbl_seg = dbls.getComponents(p);
    for (int i = 0; i < 3; ++i)
      dbl_seg(i) = e * (i + 1);
  };
  pumipic::parallel_for(structure, setComponents, "Set components");

  const double TOL = .00001;
  auto checkComponents = PS_LAMBDA(const lid_t& e, const lid_t& p, const bool& mask) {
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

#include "test_constructor.cpp"
#include "test_rebuild.cpp"
#include "test_migrate.cpp"


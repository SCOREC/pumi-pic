#include <particle_structs.hpp>
#include "read_particles.hpp"
#include <cmath>
#include "team_policy.hpp"
#include "ppMemUsage.hpp"

int comm_rank, comm_size;

void finalize() {
  Kokkos::finalize();
  MPI_Finalize();
}

//Copied and edited from: https://forums.developer.nvidia.com/t/best-way-to-report-memory-consumption-in-cuda/21042
double get_mem_usage() {
  //Barrier+fence to ensure readings are accurate when sharing GPUs in parallel.
  MPI_Barrier(MPI_COMM_WORLD);
  Kokkos::fence();
  size_t free_byte, total_byte;
  getMemUsage( &free_byte, &total_byte );
  double free_db = (double)free_byte;
  double total_db = (double)total_byte;
  double used_db = total_db - free_db;
  return used_db/1024/1024/1024;
}
//Structure adding functions
PS* buildNextStructure(int num, lid_t num_elems, lid_t num_ptcls, kkLidView ppe, kkGidView element_gids,
                       kkLidView particle_elements, PS::MTVs particle_info, std::string& name);

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
  bool check_memory = false;
#ifdef PP_USE_GPU
  check_memory = true;
#endif
  printf("CHECK: %d\n", check_memory);
  char filename[256];
  sprintf(filename, "%s_%d.ptl", argv[1], comm_rank);
  //Local count of fails
  int fails = 0;
  double initial_memory = get_mem_usage();
  {

    //General structure parameters
    lid_t num_elems;
    lid_t num_ptcls;
    kkLidView ppe;
    kkGidView element_gids;
    kkLidView particle_elements;
    PS::MTVs particle_info;
    readParticles(filename, num_elems, num_ptcls, ppe, element_gids,
                  particle_elements, particle_info);
    int num = 0;
    //Loops through each structure available in buildNextStructure(...) and execute tests on the structures
    while(true) {
      double mem_i = get_mem_usage();
      std::string name;
      PS* structure = buildNextStructure(num++, num_elems, num_ptcls, ppe, element_gids,
                                         particle_elements, particle_info, name);
      //Check if construction of structure failed
      if (name == "FAIL")
        ++fails;
      //structure is NULL on last/failed call to buildNextStructure
      if (!structure)
        break;
      //Run all tests on the structure
      double mem_s = get_mem_usage();
      fails += testCounts(name.c_str(), structure, num_elems, num_ptcls);
      fails += testParticleExistence(name.c_str(), structure, num_ptcls);
      fails += setValues(name.c_str(), structure);
      fails += pseudoPush(name.c_str(), structure);
      fails += testMetrics(name.c_str(), structure);
      double mem_pr = get_mem_usage();
      if (check_memory && fabs(mem_pr - mem_s) > .00001) {
        fprintf(stderr, "[ERROR] Structure %s memory usage changed in setup routines [%f]\n", name, mem_pr - mem_s);
        ++fails;
      }
      //Memory changes are expected in rebuild/migration (the structure check at the end will ensure no memory leaks)
      fails += testRebuild(name.c_str(), structure);
      fails += testMigration(name.c_str(), structure);
      double mem_pb = get_mem_usage();
      fails += testCopy(name.c_str(), structure);
      fails += testSegmentComp(name.c_str(), structure);
      double mem_pd = get_mem_usage();
      if (check_memory && fabs(mem_pd - mem_pb) > .00001) {
        fprintf(stderr, "[ERROR] Structure %s memory usage changed in later tests [%f]\n", name, mem_pd - mem_pb);
        ++fails;
      }
      fails += migrateToEmptyAndRefill(name.c_str(), structure);
      delete structure;
      double mem_f = get_mem_usage();
      if (check_memory && fabs(mem_f - mem_i) > .00001) {
        ++fails;
        fprintf(stderr, "[ERROR] Memory usage changed during structure %s"
                "| Initial: %f GB | Final: %f GB | Diff: %f GB\n", name.c_str(),
                mem_i, mem_f, mem_f - mem_i);
      }
      break;
    }
    ps::destroyViews<Types>(particle_info);
  }
  double final_memory = get_mem_usage();
  if (check_memory && fabs(final_memory - initial_memory) > .00001) {
    ++fails;
    fprintf(stderr, "[ERROR] Memory usage changed | Initial: %f GB | Final: %f GB | Diff: %f GB\n", initial_memory, final_memory, final_memory - initial_memory);
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

PS* buildNextStructure(int num, lid_t num_elems, lid_t num_ptcls, kkLidView ppe, kkGidView element_gids,
                       kkLidView particle_elements, PS::MTVs particle_info, std::string& name) {
  name="";
  std::string error_message;
  try {
    if (num == 0) {
      //Build SCS with C = 32, sigma = ne, V = 1024
      error_message = "SCS (C=32, sigma=ne, V=1024)";
      lid_t maxC = 32;
      lid_t sigma = num_elems;
      lid_t V = 1024;
      Kokkos::TeamPolicy<ExeSpace> policy = pumipic::TeamPolicyAuto(4, maxC);
      name = "scs_C32_SMAX_V1024";
      return new ps::SellCSigma<Types, MemSpace>(policy, sigma, V, num_elems, num_ptcls, ppe,
                                                 element_gids, particle_elements, particle_info);
    }
    else if (num == 1) {
      //Build SCS with C = 32, sigma = 1, V = 10
      error_message = "SCS (C=32, sigma=1, V=10)";
      lid_t maxC = 32;
      lid_t sigma = 1;
      lid_t V = 10;
      Kokkos::TeamPolicy<ExeSpace> policy = pumipic::TeamPolicyAuto(4, maxC);
      name = "scs_C32_S1_V10";
      return  new ps::SellCSigma<Types, MemSpace>(policy, sigma, V, num_elems, num_ptcls, ppe,
                                                  element_gids, particle_elements, particle_info);
    }
    else if (num == 2) {
      //CSR
      error_message = "CSR";
      name = "csr";
      Kokkos::TeamPolicy<ExeSpace> policy = pumipic::TeamPolicyAuto(num_elems, 32);
      return new ps::CSR<Types, MemSpace>(policy, num_elems, num_ptcls, ppe,
                                          element_gids, particle_elements, particle_info);
    }
#ifdef PP_ENABLE_CAB
    else if (num == 3) {
      //CabM
      error_message = "CabM";
      name = "cabm";
      Kokkos::TeamPolicy<ExeSpace> policy = pumipic::TeamPolicyAuto(num_elems,32);
      return new ps::CabM<Types, MemSpace>(policy, num_elems, num_ptcls, ppe,
                                           element_gids, particle_elements, particle_info);
    }
    else if (num == 4) {
      //DPS
      error_message = "DPS";
      name = "dps";
      Kokkos::TeamPolicy<ExeSpace> policy = pumipic::TeamPolicyAuto(num_elems,32);
      return new ps::DPS<Types, MemSpace>(policy, num_elems, num_ptcls, ppe,
                                          element_gids, particle_elements, particle_info);
    }
#endif
    return NULL;
  }
  catch(...) {
    fprintf(stderr, "[ERROR] Construction of %s failed on rank %d\n", error_message, comm_rank);
    name = "FAIL\n";
    return NULL;
  }
  return NULL;
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
  #ifndef PP_USE_GPU
    return 0;
  #endif

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


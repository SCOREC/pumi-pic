#include <particle_structs.hpp>
#include "read_particles.hpp"

int testCounts(const char* name, PS* structure, lid_t num_elems, lid_t num_ptcls) {
  printf("testCounts %s\n", name);

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
  printf("testParticleExistence %s, rank %d\n", name, comm_rank);

  int fails = 0;
  kkLidView count("count", 1);
  auto checkExistence = PS_LAMBDA(const lid_t& e, const lid_t& p, const bool& mask) {
    if (mask)
      Kokkos::atomic_increment<lid_t>(&(count(0)));
  };
  ps::parallel_for(structure, checkExistence, "check particle existence");
  lid_t c = ps::getLastValue<lid_t>(count);
  if (c != num_ptcls) {
    fprintf(stderr, "[ERROR] Test %s: Number of particles found in parallel_for "
            "does not match the number of particles on rank %d "
            "[(parallel_for)%d != %d(actual)]]\n", name, comm_rank,
            c, num_ptcls);
    ++fails;
  }
  return fails;
}

int setValues(const char* name, PS* structure) {
  printf("setValues %s, rank %d\n", name, comm_rank);

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
  Kokkos::fence();
  Kokkos::Timer timer;
  ps::parallel_for(structure, setValues, "setValues");
  Kokkos::fence();
  double time = timer.seconds();
  printf("Time to set values %s : %f\n", name, time);
  return fails;
}

int pseudoPush(const char* name, PS* structure) {
  printf("pseudoPush %s, rank %d\n", name, comm_rank);

  int fails = 0;

  int elements = structure->nElems();
  fprintf(stderr, "elements : %d\n", elements);
  Kokkos::View<double*> parentElmData("parentElmData", elements);
  fprintf(stderr, "parent elm data size : %d\n", parentElmData.size());
  Kokkos::parallel_for("parentElmData", parentElmData.size(),
      KOKKOS_LAMBDA(const lid_t& e){
    parentElmData(e) = 2+3*e;
  });
  //pumipic::printView(parentElmData);

  auto dbls = structure->get<1>();
  auto bools = structure->get<2>();
  auto nums = structure->get<3>();
  int local_rank = comm_rank;
  auto quickMaths = PS_LAMBDA(const lid_t& e, const lid_t& p, const bool& mask) {
    //printf("e: %d\tp: %d\tmask: %d\n", e, p, mask);
    if (mask) {
      const double p_fp = (double) p;
      const double e_fp = (double) e;
      dbls(p, 0) += 10;
      dbls(p, 1) += 10;
      dbls(p, 2) += 10;
      dbls(p, 0) = dbls(p,0) * dbls(p,0) * dbls(p,0) / std::sqrt(p_fp) / std::sqrt(e_fp) + parentElmData(e);
      dbls(p, 1) = dbls(p,1) * dbls(p,1) * dbls(p,1) / std::sqrt(p_fp) / std::sqrt(e_fp) + parentElmData(e);
      dbls(p, 2) = dbls(p,2) * dbls(p,2) * dbls(p,2) / std::sqrt(p_fp) / std::sqrt(e_fp) + parentElmData(e);
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

  Kokkos::fence();
  Kokkos::Timer timer;
  ps::parallel_for(structure, quickMaths, "setValues");
  Kokkos::fence();
  double time = timer.seconds();
  printf("Time for math Ops on %s : %f\n", name, time);

  return fails;
}

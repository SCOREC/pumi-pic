#include <particle_structs.hpp>
#include "read_particles.hpp"

int migrateSendRight(const char* name, PS* structure) {
  printf("migrateSendRight %s, rank %d\n", name, comm_rank);
  int fails = 0;
  kkLidView failures("fails", 1);

  kkLidView new_element("new_element", structure->capacity());
  kkLidView new_process("new_process", structure->capacity());

  int num_ptcls = structure->nPtcls();
  //Send even particles one process to the right
  auto pids = structure->get<0>();
  auto rnks = structure->get<3>();
  int local_rank = comm_rank;
  int local_csize = comm_size;
  int num_elems = structure->nElems();
  auto sendRight = PS_LAMBDA(const lid_t& e, const lid_t& p, const lid_t& mask) {
    if (mask) {
      new_element(p) = e;
      if (e == num_elems - 1)
        new_process(p) = (local_rank + 1) % local_csize;
      else
        new_process(p) = local_rank;
      rnks(p) = local_rank;
    }
    else {
      new_element(p) = e;
      new_process(p) = local_rank;
    }
  };
  ps::parallel_for(structure, sendRight, "sendRight");
  structure->migrate(new_element, new_process);

  pids = structure->get<0>();
  rnks = structure->get<3>();
  auto checkPostMigrate = PS_LAMBDA(const lid_t& e, const lid_t& p, const lid_t& mask) {
    if (mask) {
      const int pid = pids(p);
      const int rank = rnks(p);
      if (e == num_elems - 1 && rank == local_rank) {
        printf("[ERROR] Failed to send particle %d on rank %d\n",
               pid, local_rank);
        failures(0) = 1;
      }
      if (e != num_elems - 1 && rank != local_rank) {
        printf("[ERROR] Incorrectly received particle %d from rank %d to rank %d in element %d\n", pid, rank, local_rank, e);
        failures(0) = 1;
      }
    }
  };
  if (comm_size > 1)
    ps::parallel_for(structure, checkPostMigrate, "checkPostMigrate");
  fails += ps::getLastValue(failures);

  printf("migrateSendRight (Reverse) %s, rank %d\n", name, comm_rank);

  //Make a distributor
  int neighbors[3];
  neighbors[0] = comm_rank;
  neighbors[1] = (comm_rank - 1 + comm_size) % comm_size;
  neighbors[2] = (comm_rank + 1) % comm_size;
  ps::Distributor<typename PS::memory_space> dist(Kokkos::min(comm_size, 3), neighbors);

  new_element = kkLidView("new_element", structure->capacity());
  new_process = kkLidView("new_process", structure->capacity());
  auto sendBack = PS_LAMBDA(const lid_t& e, const lid_t& p, const lid_t& mask) {
    if (mask) {
      new_element(p) = e;
      new_process(p) = rnks(p);
    }
  };
  ps::parallel_for(structure, sendBack, "sendBack");
  structure->migrate(new_element, new_process, dist);

  failures = kkLidView("fails", 1);
  pids = structure->get<0>();
  rnks = structure->get<3>();
  auto checkPostBackMigrate = PS_LAMBDA(const lid_t& e, const lid_t& p, const lid_t& mask) {
    if (mask) {
      if (rnks(p) != local_rank) {
        printf("[ERROR] Test %s: Particle %d from rank %d was not sent back on rank %d\n",
               name, pids(p), rnks(p), local_rank);
        failures(0) = 1;
      }
    }
  };
  ps::parallel_for(structure, checkPostBackMigrate, "checkPostBackMigrate");

  fails += ps::getLastValue(failures);
  if (num_ptcls != structure->nPtcls()) {
    printf("[ERROR] Test %s: Structure does not have all of the particles it started with on rank %d\n", name, comm_rank);
    ++fails;
  }
  return fails;
}

int migrateSendToOne(const char* name, PS* structure) {
  printf("migrateSendToOne %s, rank %d\n", name, comm_rank);
  int fails = 0;
  kkLidView failures("fails", 1);

  int np = structure->nPtcls();
  kkLidView new_element("new_element", structure->capacity());
  kkLidView new_process("new_process", structure->capacity());

  auto int_slice = structure->get<0>();
  auto double_slice = structure->get<1>();
  kkLidView ptcls_set("ptcls_set", 1);
  auto comm_rank_local = comm_rank;
  auto setValues = PS_LAMBDA(const lid_t& elem_id, const lid_t& ptcl_id, const lid_t& mask) {
    int_slice(ptcl_id) = comm_rank_local;
    double_slice(ptcl_id,0) = comm_rank_local * 5;
    if (Kokkos::atomic_fetch_add(&ptcls_set(0), mask) < np*5/100)
      new_process[ptcl_id] = 0;
    else
      new_process[ptcl_id] = comm_rank_local;
    new_element[ptcl_id] = elem_id;
  };
  ps::parallel_for(structure, setValues, "setValues");
  structure->migrate(new_element, new_process);

  int np_send;
  if (comm_rank == 0)
    np_send = np;
  else
    np_send = np*5/100;
  int expected_np_0 = 0;
  MPI_Reduce(&np_send, &expected_np_0, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

  int nPtcls = structure->nPtcls();
  if (comm_rank == 0 && nPtcls != expected_np_0) {
    fprintf(stderr, "[ERROR] %s Rank 0 has incorrect number of particles (%d != %d)\n",
            name, nPtcls, expected_np_0);
    fails++;
  }
  else if (comm_rank != 0 && nPtcls != Kokkos::ceil(np*95.0/100)) {
    fprintf(stderr, "[ERROR] %s Rank %d has incorrect number of particles (%d != %d)\n",
            name, comm_rank, nPtcls, np*95/100);
    fails++;
  }

  int_slice = structure->get<0>();
  double_slice = structure->get<1>();
  auto checkValues = PS_LAMBDA(const lid_t& elm_id, const lid_t& ptcl_id, const lid_t& mask) {
    if (mask) {
      int rank = int_slice(ptcl_id);
      double val = double_slice(ptcl_id, 0);
      if (Kokkos::fabs(rank*5 - val) > .0005) {
        printf("[ERROR] %d Value fails on ptcl %d (%d %.2f) on %s", comm_rank_local, ptcl_id, rank*5, val, name);
        failures(0) = 1;
      }
    }
  };
  ps::parallel_for(structure, checkValues, "checkValues");
  fails += particle_structs::getLastValue(failures);
  return fails;
}


int migrateToEmptyAndRefill(const char* name, PS* structure) {
  printf("migrateToEmptyAndRefill %s, rank %d\n", name, comm_rank);

  int fails = 0;
  kkLidView failures("fails", 1);

  int originalPtcls = structure->nPtcls();

  kkLidView new_element("new_element", structure->capacity());
  kkLidView new_process("new_process", structure->capacity());

  int local_rank = comm_rank;
  int local_csize = comm_size;
  auto rnks = structure->get<3>();
  auto elem = structure->get<2>();
  int num_elems = structure->nElems();

  auto sendToOdd = PS_LAMBDA(const lid_t& e, const lid_t& p, const lid_t& mask) {
    rnks(p) = local_rank;
    elem(p) = e;
    if (mask) {
      if (local_rank % 2 == 0) {
        new_process(p) = (local_rank + 1) % local_csize;
        new_element(p) = num_elems - 1;
      }
      else {
        new_process(p) = local_rank;
        new_element(p) = e;
      }
    }
    else {
      new_element(p) = e;
      new_process(p) = local_rank;
    }
  };
  ps::parallel_for(structure, sendToOdd, "sendToOdd");
  structure->migrate(new_element, new_process);

  if (comm_rank % 2 == 0 && (comm_rank != 0 || comm_size % 2 == 0)) {
    if (structure->nPtcls() != 0) {
      ++fails;
      fprintf(stderr, "[ERROR] Particles remain on rank %d\n", comm_rank);
    }
    auto checkPtcls = PS_LAMBDA(const lid_t& e, const lid_t& p, const lid_t& mask) {
      if (mask) {
        failures(0) = 1;
        printf("[ERROR] Particle %d remains on rank %d\n", p, local_rank);
      }
    };
    ps::parallel_for(structure, checkPtcls, "checkPtcls");
  }
  else {
    if (structure->nPtcls() < originalPtcls) {
      ++fails;
      fprintf(stderr, "[ERROR] No particles on rank %d\n", comm_rank);
    }
    const int prev_rank = ((comm_rank - 1) + comm_size) % comm_size;
    auto new_rnks = structure->get<3>();
    auto checkPtcls = PS_LAMBDA(const lid_t& e, const lid_t& p, const lid_t& mask) {
      if (mask) {
        if (new_rnks(p) != local_rank && new_rnks(p) !=  prev_rank) {
          failures(0) = 1;
          printf("[ERROR] Particle %d is not from ranks %d or %d\n", p, local_rank, prev_rank);
        }
      }
    };
    ps::parallel_for(structure, checkPtcls, "checkPtcls");
  }

  //Send Particles back to original process
  rnks = structure->get<3>();
  new_element = kkLidView("new_element", structure->capacity());
  new_process = kkLidView("new_process", structure->capacity());
  auto sendToOrig = PS_LAMBDA(const lid_t& e, const lid_t& p, const lid_t& mask) {
    new_element(p) = e;
    if (mask) {
      new_process(p) = rnks(p);
    }
    else {
      new_process(p) = local_rank;
    }
  };
  ps::parallel_for(structure, sendToOrig, "sendToOrig");
  structure->migrate(new_element, new_process);

  if (structure->nPtcls() != originalPtcls) {
    ++fails;
    fprintf(stderr, "[ERROR] Number of particles does not match original on "
            "rank %d [%d != %d]\n", comm_rank, structure->nPtcls(), originalPtcls);
  }

  auto elems = structure->get<2>();
  new_element = kkLidView("new_element", structure->capacity());
  auto resetElements = PS_LAMBDA(const lid_t& e, const lid_t& p, const lid_t& mask) {
    if (mask) {
      new_element(p) = elems(p);
    }
    else {
      new_element(p) = e;
    }
  };
  structure->rebuild(new_element);

  fails += pumipic::getLastValue(failures);
  return fails;
}
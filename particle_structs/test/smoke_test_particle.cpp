#include <particle_structs.hpp>
#include "read_particles.hpp"
#include "team_policy.hpp"

void testStructure(PS* structure, int comm_rank, int comm_size, kkLidView failures);

int main(int argc, char* argv[]) {
  Kokkos::initialize(argc, argv);
  MPI_Init(&argc, &argv);
  int test_failed = 1;
  {
    int comm_rank, comm_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &comm_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &comm_size);


    const int num_elems = 5;
    const int num_ptcls = 5;
    const lid_t V = 1024;
    const lid_t sigma = INT_MAX;
    kkLidView ppe = kkLidView("particles_per_element", num_elems);
    kkGidView element_gids = kkGidView("elemnt_gids", num_elems);
    kkLidView particle_elements = kkLidView("element_of_particle", num_ptcls);
    PS::MTVs particle_info = ps::createMemberViews<Types>(num_ptcls);

    lid_t ppe_h[num_elems] = {1,1,1,1,1};
    lid_t element_gids_h[num_elems] = {0,1,2,3,4};
    lid_t particle_elements_h[num_ptcls] = {0,1,2,3,4};
    ps::hostToDevice(ppe, ppe_h);
    ps::hostToDevice(element_gids, element_gids_h);
    ps::hostToDevice(particle_elements, particle_elements_h);
    // Storage slots can change during construction, rebuild, and migration.
    // Carry the input identity in particle data instead of treating a slot as ID.
    ps::hostToDevice(ps::getMemberView<Types, 0>(particle_info), particle_elements_h);
    Kokkos::TeamPolicy<ExeSpace> policy = pumipic::TeamPolicyAuto(num_elems,32);
    kkLidView failures("fails", 1);

    PS* structure = new ps::SellCSigma<Types, MemSpace>(policy, sigma, V, num_elems, num_ptcls, ppe,
                                                 element_gids, particle_elements, particle_info);
    testStructure(structure, comm_rank, comm_size, failures);
    delete structure;

    structure = new ps::CSR<Types, MemSpace>(policy, num_elems, num_ptcls, ppe,
                                          element_gids, particle_elements, particle_info);
    testStructure(structure, comm_rank, comm_size, failures);
    delete structure;

    structure = new ps::DPS<Types, MemSpace>(policy, num_elems, num_ptcls, ppe,
                                      element_gids, particle_elements, particle_info);
    testStructure(structure, comm_rank, comm_size, failures);
    delete structure;

    test_failed = ps::getLastValue(failures);
    ps::destroyViews<Types>(particle_info);
  }
  Kokkos::finalize();
  MPI_Finalize();

  if (test_failed) exit(1);
}


void testStructure(PS* structure, int comm_rank, int comm_size, kkLidView failures) {
  //Init
  auto ids = structure->get<0>();
  auto checkInit = PS_LAMBDA(const lid_t& e, const lid_t& p, const bool& mask) {
    if (mask && e != ids(p)){
      printf("Structure not initalized at Particle: %d, Elm: %d\n", p, e);
      failures(0) = 1;
    }
  };
  ps::parallel_for(structure, checkInit, "checkInit");

  //Rebuild
  kkLidView new_element("new_element", structure->capacity());
  auto setNewElm = PS_LAMBDA(const lid_t& e, const lid_t& p, const bool& mask) {
    if (mask) new_element(p) = comm_rank;
    else new_element(p) = -1;
  };
  ps::parallel_for(structure, setNewElm, "setNewElm");
  structure->rebuild(new_element);

  auto checkRebuild = PS_LAMBDA(const lid_t& e, const lid_t& p, const bool& mask) {
    if (mask && e != comm_rank){
      printf("Structure failed to rebuild at Ptcl: %d, Elm: %d\n", p, e);
      failures(0) = 1;
    }
  };
  ps::parallel_for(structure, checkRebuild, "checkRebuild");

  //Migrate
  kkLidView new_process("new_process", structure->capacity());
  auto setNewProcess = PS_LAMBDA(const lid_t& e, const lid_t& p, const bool& mask) {
    if (mask) new_process(p) = (comm_rank + 1) % comm_size;
    else new_process(p) = comm_rank;
  };
  ps::parallel_for(structure, setNewProcess, "setNewProcess");
  structure->migrate(new_element, new_process);

  // Incoming particles retain the parent selected by their source rank.
  // With one rank, migration is a round trip and the parent remains zero.
  const int expected_parent = (comm_rank + comm_size - 1) % comm_size;
  ids = structure->get<0>();
  kkLidView seen("migrated_particle_ids", 5);
  auto checkMigrate = PS_LAMBDA(const lid_t& e, const lid_t& p, const bool& mask) {
    if (!mask) return;
    if (e != expected_parent || ids(p) < 0 || ids(p) >= 5){
      printf("Structure failed to migrate ranks at Ptcl: %d, Elm: %d\n", p, e);
      failures(0) = 1;
    } else {
      Kokkos::atomic_inc(&seen(ids(p)));
    }
  };
  ps::parallel_for(structure, checkMigrate, "checkMigrate");
  auto host_seen = ps::deviceToHost(seen);
  for (int id = 0; id < 5; ++id) {
    if (host_seen(id) != 1) {
      printf("Expected particle ID %d exactly once after migration, found %d\n",
             id, host_seen(id));
      Kokkos::deep_copy(failures, 1);
    }
  }
}

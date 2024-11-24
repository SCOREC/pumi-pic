#include <particle_structs.hpp>
#include "read_particles.hpp"
#include "team_policy.hpp"

int main(int argc, char* argv[]) {
  Kokkos::initialize(argc, argv);
  MPI_Init(&argc, &argv);
  int test_failed = 1;
  {
    int comm_rank, comm_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &comm_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &comm_size);


    int num_elems = 5;
    int num_ptcls = 5;
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
    kkLidView failures("fails", 1);

    //Init
    Kokkos::TeamPolicy<ExeSpace> policy = pumipic::TeamPolicyAuto(num_elems,32);
    PS* structure = new ps::DPS<Types, MemSpace>(policy, num_elems, num_ptcls, ppe,
                                      element_gids, particle_elements, particle_info);

    auto checkInit = PS_LAMBDA(const lid_t& e, const lid_t& p, const bool& mask) {
      if (mask && e != p){
        printf("Structure not initalized at Elm: %d, Particle: %d\n", e, p);
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
        printf("Particle not set to correct element at Ptcl: %d, Elm: %d\n", p, e);
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

    auto checkMigrate = PS_LAMBDA(const lid_t& e, const lid_t& p, const bool& mask) {
      if (mask && e == comm_rank){
        printf("Particle in correct rank at Ptcl: %d, Elm: %d\n", p, e);
        failures(0) = 1;
      }
    };
    ps::parallel_for(structure, checkMigrate, "checkMigrate");

    test_failed = ps::getLastValue(failures);
    delete structure;
    ps::destroyViews<Types>(particle_info);
  }
  Kokkos::finalize();
  MPI_Finalize();

  if (test_failed) exit(1);
}
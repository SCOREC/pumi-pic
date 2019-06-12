#include <stdio.h>
#include <Kokkos_Core.hpp>

#include <MemberTypes.h>
#include <SellCSigma.h>
#include <SCS_Macros.h>

#include <psAssert.h>
#include <Distribute.h>
#include <mpi.h>

using particle_structs::SellCSigma;
using particle_structs::MemberTypes;
using particle_structs::distribute_particles;
using particle_structs::distribute_elements;

typedef MemberTypes<int, double[3]> Type;
typedef Kokkos::DefaultExecutionSpace exe_space;
typedef SellCSigma<Type, exe_space> SCS;
int main(int argc, char* argv[]) {
  Kokkos::initialize(argc, argv);
  MPI_Init(&argc, &argv);

  int comm_rank;
  int comm_size;
  MPI_Comm_rank(MPI_COMM_WORLD, &comm_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &comm_size);


  int ne = 5;
  int np = 20;
  particle_structs::gid_t* gids = new particle_structs::gid_t[ne];
  distribute_elements(ne, 0, comm_rank, comm_size, gids);
  int* ptcls_per_elem = new int[ne];
  std::vector<int>* ids = new std::vector<int>[ne];
  distribute_particles(ne, np, 2, ptcls_per_elem, ids);
  delete [] ids;
  {
    SCS::kkLidView ptcls_per_elem_v("ptcls_per_elem_v", ne);
    SCS::kkGidView element_gids_v("element_gids_v", ne);
    particle_structs::hostToDevice(ptcls_per_elem_v, ptcls_per_elem);
    particle_structs::hostToDevice(element_gids_v, gids);
    delete [] ptcls_per_elem;
    delete [] gids;
    Kokkos::TeamPolicy<exe_space> po(4, 4);
    SCS* scs = new SCS(po, 5, 2, ne, np, ptcls_per_elem_v, element_gids_v);

    char rank_str[100];
    sprintf(rank_str,"Format for rank %d", comm_rank);
    scs->printFormatDevice(rank_str);
  
    typedef SCS::kkLidView kkLidView;
    kkLidView new_element("new_element", scs->capacity());
    kkLidView new_process("new_process", scs->capacity());

    auto int_slice = scs->get<0>();
    auto double3_slice = scs->get<1>();

    auto setValues = SCS_LAMBDA(int elm_id, int ptcl_id, int mask) {
      int_slice(ptcl_id) = comm_rank;
      double3_slice(ptcl_id, 0) = comm_rank;
      double3_slice(ptcl_id, 1) = (comm_rank + 1) * elm_id;
      double3_slice(ptcl_id, 2) = mask;
    };
    scs->parallel_for(setValues);
    //Send half the particles right one process except on rank 0
    if (comm_rank > 0) {
      auto setElmProcess = SCS_LAMBDA(int element_id, int particle_id, int mask) {
        new_element(particle_id) = element_id;
        new_process(particle_id) = (comm_rank + (element_id==4)) % comm_size;
      };
      scs->parallel_for(setElmProcess);
    }
    else {
      auto setElmProcess = SCS_LAMBDA(int element_id, int particle_id, int mask) {
        new_element(particle_id) = element_id;
        new_process(particle_id) = comm_rank;
      };
      scs->parallel_for(setElmProcess);
    }

    scs->migrate(new_element, new_process);


    int_slice = scs->get<0>();
    double3_slice = scs->get<1>();
    SCS::kkLidView fail("fail", 1);
    auto checkValues = SCS_LAMBDA(int elm_id, int ptcl_id, int mask) {
      if (mask != double3_slice(ptcl_id, 2)) {
        printf("mask failure on ptcl %d (%d, %f)\n", ptcl_id, mask, double3_slice(ptcl_id,2));
        fail(0) = 1;
      }
      if (mask && comm_rank != int_slice(ptcl_id) && elm_id != 0) {
        printf("rank failure on ptcl %d\n", ptcl_id);
        fail(0) = 1;
      }
      if (mask && int_slice(ptcl_id) != double3_slice(ptcl_id, 0)) {
        printf("int/double failure on ptcl %d\n", ptcl_id);
        fail(0) = 1;
      }
    };
    scs->parallel_for(checkValues);
    MPI_Barrier(MPI_COMM_WORLD);
    scs->printFormatDevice(rank_str);

    int f = particle_structs::getLastValue(fail);
    if (f == 1) {
      printf("Migration of values failed on rank %d\n", comm_rank);
      return EXIT_FAILURE;
    }
    delete scs;
  }
  Kokkos::finalize();
  MPI_Barrier(MPI_COMM_WORLD);
  MPI_Finalize();
  if (comm_rank == 0)
    printf("All tests passed\n");
  return 0;
}

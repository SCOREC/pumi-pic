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

typedef MemberTypes<int> Type;
typedef Kokkos::DefaultExecutionSpace exe_space;

int main(int argc, char* argv[]) {
  Kokkos::initialize(argc, argv);
  MPI_Init(&argc, &argv);

  int comm_rank;
  int comm_size;
  MPI_Comm_rank(MPI_COMM_WORLD, &comm_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &comm_size);

  int ne = 5;
  int np = 20;
  int* ptcls_per_elem = new int[ne];
  std::vector<int>* ids = new std::vector<int>[ne];
  distribute_particles(ne, np, 0, ptcls_per_elem, ids);
  Kokkos::TeamPolicy<exe_space> po(4, 4);
  typedef SellCSigma<Type, exe_space> SCS;
  SCS* scs =
    new SCS(po, 5, 2, ne, np, ptcls_per_elem, ids);

  scs->transferToDevice();
  
  {
    typedef SCS::kkLidView kkLidView;
    kkLidView new_element("new_element", scs->size());
    kkLidView new_process("new_process", scs->size());
    
    auto setElmProcess = SCS_LAMBDA(int element_id, int particle_id, int mask) {
      new_element[particle_id] = element_id;
      new_process[particle_id] = (particle_id>=10);
    };
    if (comm_rank == 1)
      scs->parallel_for(setElmProcess);

    scs->migrate(new_element, new_process);
  }

  delete scs;
  MPI_Finalize();
  Kokkos::finalize();
  printf("All tests passed\n");
  return 0;
}

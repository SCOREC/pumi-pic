#include <particle_structs.hpp>
#include "read_particles.hpp"
#include <stdlib.h>

#include "Distribute.h"

#ifdef PP_USE_CUDA
typedef Kokkos::CudaSpace DeviceSpace;
#else
typedef Kokkos::HostSpace DeviceSpace;
#endif
int comm_rank, comm_size;

using particle_structs::MemberTypes;
using particle_structs::getLastValue;
using particle_structs::lid_t;
typedef Kokkos::DefaultExecutionSpace exec_space;
typedef MemberTypes<int> Type;
typedef particle_structs::CabM<Type> CabM;


int main(int argc, char* argv[]) {
  Kokkos::initialize(argc,argv);
  MPI_Init(&argc,&argv);

  //count of fails
  int fails = 0;
  //begin Kokkos scope
  {
    MPI_Comm_rank(MPI_COMM_WORLD,&comm_rank);
    MPI_Comm_size(MPI_COMM_WORLD,&comm_size);
    
    if(argc != 4){
      printf("[ERROR] Too few command line arguments (%d supplied, 4 required)\n", argc);
      return 1;
    }
    int number_elements  = atoi(argv[1]);
    int number_particles = atoi(argv[2]);
    int distribution     = atoi(argv[3]);
    printf("Rebuild CabM Test: \nne = %d\nnp = %d\ndistribution = %d\n",
        number_elements, number_particles, distribution);
    
    ///////////////////////////////////////////////////////////////////////////
    //Tests to run go here
    ///////////////////////////////////////////////////////////////////////////


  }
  //end Kokkos scope
  Kokkos::finalize();
  MPI_Finalize();
  return fails;
}
#include <particle_structs.hpp>
#include "read_particles.hpp"

#ifdef PP_USE_CUDA
typedef Kokkos::CudaSpace DeviceSpace;
#else
typedef Kokkos::HostSpace DeviceSpace;
#endif
int comm_rank, comm_size;

bool rebuildNoChanges(ps::CSR<Types,MemSpace>* csr, kkLidView particle_elements);
bool rebuildNewElems(ps::CSR<Types,MemSpace>* csr);
bool rebuildNewPtcls(ps::CSR<Types,MemSpace>* csr);
bool rebuildPtclsDestroyed(ps::CSR<Types,MemSpace>* csr);
bool rebuildNewAndDestroyed(ps::CSR<Types,MemSpace>* csr);


int main(int argc, char* argv[]){
  Kokkos::initialize(argc,argv);
  MPI_Init(&argc,&argv);

  //count of fails
  int fails = 0;
  //begin Kokkos scope
  {
    MPI_Comm_rank(MPI_COMM_WORLD,&comm_rank);
    MPI_Comm_size(MPI_COMM_WORLD,&comm_size);

    if(argc != 2){
      if(!comm_rank){
        fprintf(stdout,"[ERROR] Format: %s <particle_file_prefix>\n",argv[0]);
      }
      Kokkos::finalize();
      MPI_Finalize();
      return 1;
    }

    char filename[256];
    sprintf(filename, "%s_%d.ptl", argv[1], comm_rank);
    //General structure parameters
    lid_t num_elems;
    lid_t num_ptcls;
    kkLidView ppe;
    kkGidView element_gids;
    kkLidView particle_elements;
    PS::MTVs particle_info;
    readParticles(filename, num_elems, num_ptcls, ppe, element_gids,
                particle_elements, particle_info);

    Kokkos::TeamPolicy<ExeSpace> policy(num_elems,32); //league_size, team_size
    ps::CSR<Types,MemSpace>* csr = new ps::CSR<Types, MemSpace>(policy, num_elems, num_ptcls, 
                                      ppe, element_gids, particle_elements, particle_info);

    ///////////////////////////////////////////////////////////////////////////
    //Tests to run go here
    ///////////////////////////////////////////////////////////////////////////

    //verify still correct elements and no changes otherwise
    fails += rebuildNoChanges(csr, particle_elements);

    //verify all elements are correctly assigned to new elements
    //fails += rebuildNewElems(csr);

    //check new elements are added properly and all elements assgined correct
    //fails += rebuildNewPtcls(csr);

    //check for particles that were removed and the rest in their correct loc
    //fails += rebuildPtclsDestroyed(csr);

    //complete check of rebuild functionality
    //fails += rebuildNewAndDestroyed(csr);

  }
  //end Kokkos scope
  Kokkos::finalize();
  MPI_Finalize();
  return fails;
}

//template <class DataTypes,typename MemSpace>
//CSR<DataTypes,MemSpace>::rebuild(kkLidView new_element,
//                                 kkLidView new_particle_elements,
//                                 MTVs new_particles);

//Rebuild test with no changes to structure
bool rebuildNoChanges(ps::CSR<Types,MemSpace>* csr, kkLidView particle_elements){
  Kokkos::Profiling::pushRegion("rebuildNoChanges");
  bool failed = false;

  kkLidView new_element = kkLidView("new_element", csr->getNumPtcls());
  kkLidView new_particle_elements = kkLidView("new_particle_elements", 0);
  PS::MTVs new_particles;

  Kokkos::parallel_for("new ptcl elements assignment", new_element.size(),
                          KOKKOS_LAMBDA(const int& i){
    new_element(i) = i/5;
  });


  csr->rebuild(new_element, new_particle_elements, new_particles);

  Kokkos::Profiling::popRegion();
  return failed;
}

//Rebuild test with no new particles, but reassigned particle elements
bool rebuildNewElems(ps::CSR<Types,MemSpace>* csr){
  Kokkos::Profiling::pushRegion("rebuildNewElems");
  bool passed = true;


  Kokkos::Profiling::popRegion();
  return passed;
}

//Rebuild test with new particles added only
bool rebuildNewPtcls(ps::CSR<Types,MemSpace>* csr){
  Kokkos::Profiling::pushRegion("rebuildNewPtcls");
  bool passed = true;


  Kokkos::Profiling::popRegion();
  return passed;
}

//Rebuild test with exsiting particles destroyed only
bool rebuildPtclsDestroyed(ps::CSR<Types,MemSpace>* csr){
  Kokkos::Profiling::pushRegion("rebuildPtclsDestroyed");
  bool passed = true;


  Kokkos::Profiling::popRegion();
  return passed;
}

//Rebuild test with particles added and destroyed
bool rebuildNewAndDestroyed(ps::CSR<Types,MemSpace>* csr){
  Kokkos::Profiling::pushRegion("rebuildNewAndDestroyed");
  bool passed = true;


  Kokkos::Profiling::popRegion();
  return passed;
}



#include <particle_structs.hpp>
#include "read_particles.hpp"

#include <CSR.hpp>
#include <MemberTypes.h>

#include "Distribute.h"

#ifdef PP_USE_CUDA
typedef Kokkos::CudaSpace DeviceSpace;
#else
typedef Kokkos::HostSpace DeviceSpace;
#endif
int comm_rank, comm_size;

//using particle_structs::CSR;
using particle_structs::MemberTypes;
using particle_structs::getLastValue;
using particle_structs::lid_t;
typedef Kokkos::DefaultExecutionSpace exe_space;
typedef MemberTypes<int> Type;
typedef particle_structs::CSR<Type> CSR;

int rebuildNoChanges();
int rebuildNewElems();
int rebuildNewPtcls();
int rebuildPtclsDestroyed();
int rebuildNewAndDestroyed();


int main(int argc, char* argv[]){
  Kokkos::initialize(argc,argv);
  MPI_Init(&argc,&argv);

  //count of fails
  int fails = 0;
  //begin Kokkos scope
  {
    MPI_Comm_rank(MPI_COMM_WORLD,&comm_rank);
    MPI_Comm_size(MPI_COMM_WORLD,&comm_size);

    ///////////////////////////////////////////////////////////////////////////
    //Tests to run go here
    ///////////////////////////////////////////////////////////////////////////

    //verify still correct elements and no changes otherwise
    fails += rebuildNoChanges();

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
int rebuildNoChanges(){
  Kokkos::Profiling::pushRegion("rebuildNoChanges");
  int fails = 0;

  //Test construction based on SCS testing
  int ne = 5;
  int np = 20;
  int* ptcls_per_elem = new int[ne];
  std::vector<int>* ids = new std::vector<int>[ne];
  distribute_particles(ne, np, 0, ptcls_per_elem, ids); 
  Kokkos::TeamPolicy<exe_space> po(32,Kokkos::AUTO);
  CSR::kkLidView ptcls_per_elem_v("ptcls_per_elem_v",ne);
  CSR::kkGidView element_gids_v("",0);
  particle_structs::hostToDevice(ptcls_per_elem_v,ptcls_per_elem);

  CSR* csr = new CSR(po, ne, np, ptcls_per_elem_v, element_gids_v);
  CSR* csr2 = new CSR(po, ne, np, ptcls_per_elem_v, element_gids_v);

  delete [] ptcls_per_elem;
  delete [] ids;

  CSR::kkLidView new_element("new_element", csr->getCapacity());
  
  auto values = csr->get<0>();
  auto values2 = csr->get<0>();
  auto offsets_cpy = csr->getOffsets();
  //"Send To Self"
  Kokkos::parallel_for("sendToSelf", np, 
      KOKKOS_LAMBDA (const int& i) {
    //assigns new_elements to be the current elements and fills values and
    //values2 for comparison after the rebuild

    lid_t elem = 0;
    while(offsets_cpy(elem+1) < i)
      elem++;

    new_element(i) = elem;
    values(i) = i;
    values2(i) = i;  

  });

  fprintf(stderr,"Rebuild next\n");
  //Rebuild with no changes
  csr->rebuild(new_element);
  values = csr->get<0>();

  kkLidView failed = kkLidView("failed", 1);
  Kokkos::parallel_for("check no changes", np,
      KOKKOS_LAMBDA (const int& i){
    lid_t elem1 = 0;
    lid_t elem2 = 0;
    while(offsets_cpy(elem1+1) < i)
      elem1++;
    while(offsets_cpy(elem2+1) < i)
      elem2++;

    if(elem1 != elem2)
      failed(0)+=1; 
  });


  fails+= getLastValue<lid_t>(failed);

  Kokkos::Profiling::popRegion();
  return fails;
}

//Rebuild test with no new particles, but reassigned particle elements
int rebuildNewElems(){
  Kokkos::Profiling::pushRegion("rebuildNewElems");
  int fails = 0;

  //kkLidView new_element = kkLidView("new_element", csr->getNumPtcls());
  //kkLidView new_particle_elements = kkLidView("new_particle_elements", 0);
  //PS::MTVs new_particles;

  //Kokkos::parallel_for("new ptcl elements assignment", new_element.size(),
  //                        KOKKOS_LAMBDA(const int& i){
  //  //for the case of 5 particles per element
  //  new_element(i) = i/5;
  //});

  //csr->rebuild(new_element, new_particle_elements, new_particles);



  Kokkos::Profiling::popRegion();
  return fails;
}

//Rebuild test with new particles added only
int rebuildNewPtcls(){
  Kokkos::Profiling::pushRegion("rebuildNewPtcls");
  int fails = 0;


  Kokkos::Profiling::popRegion();
  return fails;
}

//Rebuild test with exsiting particles destroyed only
int rebuildPtclsDestroyed(){
  Kokkos::Profiling::pushRegion("rebuildPtclsDestroyed");
  int fails = 0; 


  Kokkos::Profiling::popRegion();
  return fails;
}

//Rebuild test with particles added and destroyed
int rebuildNewAndDestroyed(){
  Kokkos::Profiling::pushRegion("rebuildNewAndDestroyed");
  int fails = 0;


  Kokkos::Profiling::popRegion();
  return fails;
}



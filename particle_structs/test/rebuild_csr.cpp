#include <particle_structs.hpp>
#include "read_particles.hpp"

#include <CSR.hpp>
#include <MemberTypes.h>
#include <MemberTypeLibraries.h>
#include <ppTypes.h>
#include <psMemberType.h>

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
    //fails += rebuildNewElems();

    //check new elements are added properly and all elements assgined correct
    fails += rebuildNewPtcls();

    //check for particles that were removed and the rest in their correct loc
    //fails += rebuildPtclsDestroyed();

    //complete check of rebuild functionality
    //fails += rebuildNewAndDestroyed();

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
//  fprintf(stderr,"csr inited\n");
//  printView(ps::getMemberView<Type,0>(csr->getPtcl_data()));
//  fprintf(stderr,"csr2 inited\n");
//  printView(ps::getMemberView<Type,0>(csr2->getPtcl_data()));
//
  CSR::kkLidView new_element("new_element", csr->getCapacity());
  
  auto values = csr->get<0>();
  auto values2 = csr2->get<0>();
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
    //set values to the element to check they end up in the correct one
    values(i) = elem;
    values2(i) = elem;  

  });
  //fprintf(stderr,"csr data check:\n");
  //printView(ps::getMemberView<Type,0>(csr->getPtcl_data()));

  //fprintf(stderr,"csr2 data check:\n");
  //printView(ps::getMemberView<Type,0>(csr2->getPtcl_data()));

  //fprintf(stderr,"Rebuild next\n");
  //Rebuild with no changes
  csr->rebuild(new_element);

  Kokkos::parallel_for("check Element", np,
      KOKKOS_LAMBDA (const int& i){

        lid_t elem = 0;
        while(offsets_cpy(elem+1)<i) elem++;

        values(i) = elem;
      });



  values = csr->get<0>();

  auto pIDs  = ps::getMemberView<Type,0>(csr->getPtcl_data());
  auto pIDs2  = ps::getMemberView<Type,0>(csr2->getPtcl_data());

  printView(pIDs);
  printView(pIDs2);

  //values holds the current element of each particle in csr
  //values2 holds the original element assignments

  kkLidView failed = kkLidView("failed", 1);
  Kokkos::parallel_for("check no changes", np,
      KOKKOS_LAMBDA (const int& i){
    if(values(i) != values2(i))
      failed(0)+=1; 
  });


  fails+= getLastValue<lid_t>(failed);
  fprintf(stderr,"end\n");
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
//  fprintf(stderr,"csr inited\n");
//  printView(ps::getMemberView<Type,0>(csr->getPtcl_data()));
//  fprintf(stderr,"csr2 inited\n");
//  printView(ps::getMemberView<Type,0>(csr2->getPtcl_data()));

  CSR::kkLidView new_element("new_element", csr->getCapacity());
  
  auto values = csr->get<0>();
  auto values2 = csr2->get<0>();
  auto offsets_cpy = csr->getOffsets();

  //"Send To Self"
  Kokkos::parallel_for("sendToSelf", np, 
      KOKKOS_LAMBDA (const int& i) {
    //assigns new_elements to be the current elements and fills values and
    //values2 for comparison after the rebuild

    lid_t elem = 0;
    while(offsets_cpy(elem+1) < i)
      elem++;

    new_element(i) = 4-elem;
    //set values to the element to check they end up in the correct one
    values(i) = elem;
    values2(i) = elem;  

  });
  //fprintf(stderr,"csr data check:\n");
  //printView(ps::getMemberView<Type,0>(csr->getPtcl_data()));

  //fprintf(stderr,"csr2 data check:\n");
  //printView(ps::getMemberView<Type,0>(csr2->getPtcl_data()));

  //fprintf(stderr,"Rebuild next\n");
  //Rebuild with no changes
  csr->rebuild(new_element);
  values = csr->get<0>();

  Kokkos::parallel_for("check Element", np,
      KOKKOS_LAMBDA (const int& i){

        lid_t elem = 0;
        while(offsets_cpy(elem+1)<i) elem++;

        values(i) = elem;
  });




  auto pIDs  = ps::getMemberView<Type,0>(csr->getPtcl_data());
  auto pIDs2  = ps::getMemberView<Type,0>(csr2->getPtcl_data());

  printView(pIDs);
  printView(pIDs2);

  //values holds the current element of each particle in csr
  //values2 holds the original element assignments

  kkLidView failed = kkLidView("failed", 1);
  Kokkos::parallel_for("check no changes", np,
      KOKKOS_LAMBDA (const int& i){
    if(values(i) != values2(np-i)){
      failed(0)+=1; 
    }
  });
  


  fails+= getLastValue<lid_t>(failed);
  fprintf(stderr,"end\n");

  Kokkos::Profiling::popRegion();
  return fails;
}

//Rebuild test with new particles added only
int rebuildNewPtcls(){
  Kokkos::Profiling::pushRegion("rebuildNewPtcls");
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
  //fprintf(stderr,"csr2 inited\n");
  //printView(ps::getMemberView<Type,0>(csr2->getPtcl_data()));

  CSR::kkLidView new_element("new_element", csr->getCapacity());
  
  auto values = csr->get<0>();
  auto values2 = csr2->get<0>();
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
    //give each particle an identifier
    values(i) = elem;
    values2(i) = elem;  
  });
  //fprintf(stderr,"csr2 data check:\n");
  //printView(ps::getMemberView<Type,0>(csr2->getPtcl_data()));

  
  //////////////////////////////////////////////////////////////
  //Introduce new particles
  int nnp = 5;
  CSR::kkLidView new_particle_elements("new_particle_elements", nnp);
  auto new_particles = particle_structs::createMemberViews<Type>(nnp);
  auto new_particle_access = particle_structs::getMemberView<Type,0>(new_particles);
  Kokkos::parallel_for("new_particle_elements", nnp, 
      KOKKOS_LAMBDA (const int& i){
      new_particle_elements(i) = i%5;
      new_particle_access(i) = i%5;
  });




//  //int* new_ptcls_per_elem = new int[ne];
//  delete[] new_ids;
//  std::vector<int>* new_ids = new std::vector<int>[ne];
//  distribute_particles(ne, nnp, 0, new_ptcls_per_elem, new_ids); 
//  CSR::kkLidView ptcls_per_elem_v("ptcls_per_elem_v",ne);
//  CSR::kkGidView element_gids_v("",0);
//  particle_structs::hostToDevice(new_ptcls_per_elem_v,new_ptcls_per_elem);
//
//  CSR::kkLidView new_particle_elements("new ptcl elems", nnp);
//  Kokkos::parallel_for("new ptcl elems assign", nnp,
//    KOKKOS_LAMBDA (const int& i){
//      new_particle_elements(i) = i%ne;
//  });
//
//  fprintf(stderr,"Rebuild next\n");
//  //Rebuild with new particles
//  MTVs new_particles;
//  CreateViews<device_type,DataTypes>(new_particles, new_particle_elements.size());
  fprintf(stderr,"new_particle_elements:\n");
  printView(new_particle_elements);
  fprintf(stderr,"new particle MTV data:\n");
  printView(new_particle_access);


  fprintf(stderr, "into rebuild");
  csr->rebuild(new_element, new_particle_elements, new_particles);
  values = csr->get<0>();

  auto pIDs  = ps::getMemberView<Type,0>(csr->getPtcl_data());
  auto pIDs2  = ps::getMemberView<Type,0>(csr2->getPtcl_data());

  printView(pIDs);
  printView(pIDs2);


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
  fprintf(stderr,"end\n");

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



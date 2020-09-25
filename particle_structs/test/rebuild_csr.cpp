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
    fails += rebuildNewElems();

    //check new elements are added properly and all elements assgined correct
    fails += rebuildNewPtcls();

    //check for particles that were removed and the rest in their correct loc
    fails += rebuildPtclsDestroyed();

    //complete check of rebuild functionality
    fails += rebuildNewAndDestroyed();

  }
  //end Kokkos scope
  Kokkos::finalize();
  MPI_Finalize();
  return fails;
}

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
  auto values2 = csr2->get<0>();
  auto offsets_cpy = csr->getOffsets();

  //Assign values to ptcls to track their movement
  Kokkos::parallel_for("sendToSelf", np, 
      KOKKOS_LAMBDA (const int& i) {
    lid_t elem = 0;
    while(offsets_cpy(elem+1) < i)
      elem++;

    new_element(i) = elem;
    //set values to the element to check they end up in the correct one
    values(i) = elem;
    values2(i) = elem;  
    });

  //Rebuild with no changes (function takes second and third args as optional
  csr->rebuild(new_element);

  values = csr->get<0>();

  //Setting current elements to changed MTV
  Kokkos::parallel_for("check Element", np,
      KOKKOS_LAMBDA (const int& i){
        lid_t elem = 0;
        while(offsets_cpy(elem+1)<i) elem++;

        values(i) = elem;
      });

  //values holds the current element of each particle in csr
  //values2 holds the original element assignments
  kkLidView failed = kkLidView("failed", 1);
  Kokkos::parallel_for("check no changes", np,
      KOKKOS_LAMBDA (const int& i){
    if(values(i) != values2(i))
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
  auto values2 = csr2->get<0>();
  auto offsets_cpy = csr->getOffsets();

  //Assign ptcl identifiers
  Kokkos::parallel_for("sendToSelf", np, 
      KOKKOS_LAMBDA (const int& i) {
    lid_t elem = 0;
    while(offsets_cpy(elem+1) < i)
      elem++;

    new_element(i) = (elem*3 + i)%ne; //change to assign to diff elems
    values(i) = i;
    values2(i) = i;  
  });

  //Rebuild with no changes
  csr->rebuild(new_element);
  values = csr->get<0>();
  offsets_cpy = csr->getOffsets();

  kkLidView failed = kkLidView("failed", 1);
  Kokkos::parallel_for("check no changes", np,
      KOKKOS_LAMBDA (const int& i){
    lid_t id = values2(i);
    lid_t dest_elem = new_element(i);
    lid_t row_start = offsets_cpy(dest_elem);
    lid_t row_end =offsets_cpy(dest_elem+1);
    bool found = false;
    for(lid_t i = row_start; i < row_end; ++i){
      if(values(i) == id){
        found = true;
        break;
      } 
    }
    if(!found) failed(0)+=1;
  });
  
  fails += getLastValue<lid_t>(failed);

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

  CSR::kkLidView new_element("new_element", csr->getCapacity());
  
  auto valuesOrig = csr->get<0>();
  auto valuesOrig2 = csr2->get<0>();
  auto offsets_cpy = csr->getOffsets();

  //Assign identifiers
  Kokkos::parallel_for("sendToSelf", np, 
      KOKKOS_LAMBDA (const int& i) {
    lid_t elem = 0;
    while(offsets_cpy(elem+1) < i)
      elem++;

    new_element(i) = (3*i+2)%ne;
    //give each particle an identifier
    valuesOrig(i) = i;
    valuesOrig2(i) = i;  
  });

  //////////////////////////////////////////////////////////////
  //Introduce new particles
  int nnp = 5; //number new ptcls
  CSR::kkLidView new_particle_elements("new_particle_elements", nnp);
  auto new_particles = particle_structs::createMemberViews<Type>(nnp);
  auto new_particle_access = particle_structs::getMemberView<Type,0>(new_particles);
   
  //Assign new ptcl elements and identifiers
  Kokkos::parallel_for("new_particle_elements", nnp, 
      KOKKOS_LAMBDA (const int& i){
      new_particle_elements(i) = i%5;
      new_particle_access(i) = i+np;
  });

  //Rebuild with new ptcls
  csr->rebuild(new_element, new_particle_elements, new_particles);
  
  auto valuesNew = csr->get<0>(); //new size
  offsets_cpy = csr->getOffsets();

  kkLidView failed = kkLidView("failed", 1);

  //Check old value & new_element --> expected new element 

  //Check existing ptcls placed correctly
  Kokkos::parallel_for("check orig particles", np,
      KOKKOS_LAMBDA (const int& i){
      //Find row and element info
      lid_t id = valuesOrig2(i);
      lid_t dest_elem = new_element(i);
      lid_t row_start = offsets_cpy(dest_elem);
      lid_t row_end = offsets_cpy(dest_elem+1);
      bool found = false;
      for(lid_t i = row_start; i < row_end; ++i){
        if(valuesNew(i) == id){
          found = true;
          break;
        } 
      }
      if(!found) failed(0)+=1;
  });

  //Check new ptcls placed correctly
  Kokkos::parallel_for("check new particles", nnp,
      KOKKOS_LAMBDA (const int& i){
      //insert checks here
      lid_t id = new_particle_access(i);
      lid_t dest_elem = new_particle_elements(i);
      lid_t row_start = offsets_cpy(dest_elem);
      lid_t row_end = offsets_cpy(dest_elem+1);
      bool found = false;
      for(lid_t i = row_start; i < row_end; i++){
        if(valuesNew(i) == id){
          found = true;
          break;
        }
      }
      if(!found) failed(0) += 1;
  });


  fails+= getLastValue<lid_t>(failed);

  Kokkos::Profiling::popRegion();
  return fails;
}

//Rebuild test with exsiting particles destroyed only
int rebuildPtclsDestroyed(){
  Kokkos::Profiling::pushRegion("rebuildPtclsDestroyed");
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
  auto values2 = csr2->get<0>();
  auto offsets_cpy = csr->getOffsets();

  //Assign identifiers
  Kokkos::parallel_for("sendToSelf", np, 
      KOKKOS_LAMBDA (const int& i) {

    lid_t elem = 0;
    while(offsets_cpy(elem+1) < i)
      elem++;

    new_element(i) = elem; //change to assign to diff elems
    if(i%7 == 0) new_element(i) = -1;
    values(i) = i;
    values2(i) = i;  

  });

  //Rebuild with elements removed 
  csr->rebuild(new_element);
  values = csr->get<0>();
  offsets_cpy = csr->getOffsets();

  lid_t particles = csr->getCapacity();
  kkLidView failed = kkLidView("failed", 1);
  Kokkos::parallel_for("check no changes", np,
      KOKKOS_LAMBDA (const int& i){
    lid_t id = values2(i);
    lid_t dest_elem = new_element(i);
    bool found = false;
    if(dest_elem == -1){
      for(lid_t i = 0; i < particles; ++i){
        assert(values(i) != id);
      }
      found = true;
    }
    else{
      lid_t row_start = offsets_cpy(dest_elem);
      lid_t row_end =offsets_cpy(dest_elem+1);
      for(lid_t i = row_start; i < row_end; ++i){
        if(values(i) == id){
          found = true;
          break;
        } 
      }
    }
    if(!found) failed(0)+=1;
  });
  
  fails += getLastValue<lid_t>(failed);

  Kokkos::Profiling::popRegion();
  return fails;
}

//Rebuild test with particles added and destroyed
int rebuildNewAndDestroyed(){
  Kokkos::Profiling::pushRegion("rebuildNewAndDestroyed");
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
  auto values2 = csr2->get<0>();
  auto offsets_cpy = csr->getOffsets();

  //Assign ptcl elements
  Kokkos::parallel_for("sendToSelf", np, 
      KOKKOS_LAMBDA (const int& i) {

    lid_t elem = 0;
    while(offsets_cpy(elem+1) < i)
      elem++;

    new_element(i) = (3*elem+7)%ne; //change to assign to diff elems
    if(i%7 == 0) new_element(i) = -1;
    values(i) = i;
    values2(i) = i;  

  });

  //////////////////////////////////////////////////////////////
  //Introduce new particles
  int nnp = 5;
  CSR::kkLidView new_particle_elements("new_particle_elements", nnp);
  auto new_particles = particle_structs::createMemberViews<Type>(nnp);
  auto new_particle_access = particle_structs::getMemberView<Type,0>(new_particles);
   
  Kokkos::parallel_for("new_particle_elements", nnp, 
      KOKKOS_LAMBDA (const int& i){
      new_particle_elements(i) = i%5;
      new_particle_access(i) = i%5+np;
  });

  //Rebuild with elements removed 
  csr->rebuild(new_element,new_particle_elements,new_particles);
  values = csr->get<0>();
  offsets_cpy = csr->getOffsets();

  lid_t particles = csr->getCapacity();

  kkLidView failed = kkLidView("failed", 1);
  Kokkos::parallel_for("check no changes", np,
      KOKKOS_LAMBDA (const int& i){
    lid_t id = values2(i);
    lid_t dest_elem = new_element(i);
    bool found = false;
    if(dest_elem == -1){
      for(lid_t i = 0; i < particles; ++i){
        assert(values(i) != id);
      }
      found = true;
    }
    else{
      lid_t row_start = offsets_cpy(dest_elem);
      lid_t row_end =offsets_cpy(dest_elem+1);
      for(lid_t i = row_start; i < row_end; ++i){
        if(values(i) == id){
          found = true;
          break;
        } 
      }
    }
    if(!found) failed(0)+=1;
  });
  
  fails += getLastValue<lid_t>(failed);

  Kokkos::Profiling::popRegion();
  return fails;
}


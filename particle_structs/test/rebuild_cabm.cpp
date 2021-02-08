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
using AoSoA = Cabana::AoSoA<Cabana::MemberTypes<int>,DeviceSpace::device_type>;
typedef Kokkos::DefaultExecutionSpace exe_space;
typedef MemberTypes<int> Type;
typedef particle_structs::CabM<Type> CabM;

bool rebuildNoChanges(int ne_in, int np_in,int distribution);
bool rebuildNewElems(int ne_in, int np_in,int distribution);
bool rebuildNewPtcls(int ne_in, int np_in,int distribution);
bool rebuildPtclsDestroyed(int ne_in, int np_in,int distribution);
bool rebuildNewAndDestroyed(int ne_in, int np_in,int distribution);


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

    //verify still correct elements and no changes otherwise
    fprintf(stderr,"RebuildNoChanges\n");
    fails += rebuildNoChanges(number_elements,number_particles,distribution);

    
    //verify all elements are correctly assigned to new elements
    fprintf(stderr,"RebuildNewElems\n");
    fails += rebuildNewElems(number_elements,number_particles,distribution);
    /*
    //check new elements are added properly and all elements assgined correct
    fprintf(stderr,"RebuildNewPtcls\n");
    fails += rebuildNewPtcls(number_elements,number_particles,distribution);

    //check for particles that were removed and the rest in their correct loc
    fprintf(stderr,"RebuildPtclsDestroyed\n");
    fails += rebuildPtclsDestroyed(number_elements,number_particles,distribution);

    //complete check of rebuild functionality
    fprintf(stderr,"RebuildNewAndDestroyed\n");
    fails += rebuildNewAndDestroyed(number_elements,number_particles,distribution);
    */
  }
  //end Kokkos scope
  Kokkos::finalize();
  MPI_Finalize();
  return fails;
}

int rebuild_test_counts(CabM* cabm, CabM* cabm_ref, int num_add_ptcl) {
  int fails = 0;
  if (cabm->nElems() != cabm_ref->nElems()) {
    fprintf(stderr, "[ERROR] Element count mismatch on rank %d "
            "[(structure)%d != %d(reference)]\n",
            comm_rank, cabm->nElems(), cabm_ref->nElems());
    ++fails;
  }
  if (cabm->nPtcls() + num_add_ptcl != cabm_ref->nPtcls()) {
    fprintf(stderr, "[ERROR] Particle count mismatch on rank %d "
            "[(structure)%d != %d(reference)]\n",
            comm_rank, cabm->nPtcls(), cabm_ref->nPtcls());
    ++fails;
  }
  if (cabm->numRows() < cabm_ref->nElems()) {
    fprintf(stderr, "[ERROR] Number of rows is too small to fit elements on rank %d "
            "[(structure)%d < %d(reference)]\n", comm_rank,
            cabm->numRows(), cabm_ref->nElems());
    ++fails;
  }
  if (cabm->capacity() < cabm_ref->nPtcls() + num_add_ptcl) {
    fprintf(stderr, "[ERROR] Capcity is too small to fit particles on rank %d "
            "[(structure)%d < %d(reference)]\n", comm_rank,
            cabm->capacity(), cabm_ref->nPtcls());
    ++fails;
  }
  return fails;
}

//Rebuild test with no changes to structure
bool rebuildNoChanges(int ne_in, int np_in,int distribution){
  Kokkos::Profiling::pushRegion("rebuildNoChanges");
  int fails = 0;

  //Test construction based on SCS testing
  int ne = ne_in;
  int np = np_in;
  int* ptcls_per_elem = new int[ne];
  std::vector<int>* ids = new std::vector<int>[ne];
  distribute_particles(ne, np, distribution, ptcls_per_elem, ids); 
  Kokkos::TeamPolicy<exe_space> po(32,Kokkos::AUTO);
  CabM::kkLidView ptcls_per_elem_v("ptcls_per_elem_v",ne);
  CabM::kkGidView element_gids_v("",0);
  particle_structs::hostToDevice(ptcls_per_elem_v,ptcls_per_elem);

  //Structure to perform rebuild on
  CabM* cabm = new CabM(po, ne, np, ptcls_per_elem_v, element_gids_v);
  //Structure to compare rebuild result to 
  CabM* cabm_ref = new CabM(po, ne, np, ptcls_per_elem_v, element_gids_v);

  delete [] ptcls_per_elem;
  delete [] ids;

  CabM::kkLidView new_element("new_element", cabm->capacity());

  //pID = particle ID
  //Gives access to MTV data to set an identifier for each ptcl
  auto pID = cabm->get<0>();
  auto pID_ref = cabm_ref->get<0>();

  //Assign values to ptcls to track their movement
  auto sendToSelf = PS_LAMBDA(const lid_t& elm, const lid_t& ptcl, const bool& active) {
    if (active) {
      new_element(ptcl) = elm;
      pID(ptcl) = ptcl;
      pID_ref(ptcl) = ptcl;
    }
    else {
      new_element(ptcl) = -1;
      pID(ptcl) = -1;
      pID_ref(ptcl) = -1;
    }
  };
  cabm->parallel_for(sendToSelf, "sendToSelf");

  //Rebuild with no changes (function takes second and third args as optional)
  cabm->rebuild(new_element);

  int count_fails =  rebuild_test_counts(cabm, cabm_ref, 0);
  if ( count_fails > 0 ) {
    return count_fails;
  }

  //(Necessary) update of pID
  pID = cabm->get<0>();

  kkLidView failed = kkLidView("failed", 1);
  Kokkos::parallel_for("check no changes", cabm->capacity(),
      KOKKOS_LAMBDA (const int& ptcl) {
        const lid_t id = pID_ref(ptcl);
        if (id != -1) { // only looking at active particles
          const lid_t dest_elem = new_element(ptcl);

          bool found = false;
          for (lid_t i = dest_elem*AoSoA::vector_length; i < (dest_elem+1)*AoSoA::vector_length; ++i) {
            //printf("looking for: %d, index: %d, id: %d\n", id, i, pID(i));
            if (pID(i) == id) {
              found = true;
              break;
            } 
          }
          if (!found) {
            //printf("[ERROR] particle %d not found in element %d\n", id, dest_elem);
            failed(0) = 1;
          }
        }
      });

  fails += getLastValue<lid_t>(failed);

  Kokkos::Profiling::popRegion();
  return fails;
}

//Rebuild test with no new particles, but reassigned particle elements
bool rebuildNewElems(int ne_in, int np_in,int distribution){
  Kokkos::Profiling::pushRegion("rebuildNewElems");
  int fails = 0;

  //Test construction based on SCS testing
  int ne = ne_in;
  int np = np_in;
  int* ptcls_per_elem = new int[ne];
  std::vector<int>* ids = new std::vector<int>[ne];
  distribute_particles(ne, np, distribution, ptcls_per_elem, ids); 
  Kokkos::TeamPolicy<exe_space> po(32,Kokkos::AUTO);
  CabM::kkLidView ptcls_per_elem_v("ptcls_per_elem_v",ne);
  CabM::kkGidView element_gids_v("",0);
  particle_structs::hostToDevice(ptcls_per_elem_v,ptcls_per_elem);

  //Structure to perform rebuild on
  CabM* cabm = new CabM(po, ne, np, ptcls_per_elem_v, element_gids_v);
  //Structure to compare rebuild result to 
  CabM* cabm_ref = new CabM(po, ne, np, ptcls_per_elem_v, element_gids_v);

  delete [] ptcls_per_elem;
  delete [] ids;

  CabM::kkLidView new_element("new_element", cabm->capacity(), 0);

  //pID = particle ID
  //Gives access to MTV data to set an identifier for each ptcl
  auto pID = cabm->get<0>();
  auto pID_ref = cabm_ref->get<0>();

  //Assign values to ptcls to track their movement
  auto sendToSelf = PS_LAMBDA(const lid_t& elm, const lid_t& ptcl, const bool& active) {
    if (active) {
      new_element(ptcl) = (elm*3 + ptcl)%ne; //change to assign to diff elems
      pID(ptcl) = ptcl;
      pID_ref(ptcl) = ptcl;
    }
    else {
      new_element(ptcl) = -1;
      pID(ptcl) = -1;
      pID_ref(ptcl) = -1;
    }
  };
  cabm->parallel_for(sendToSelf, "sendToSelf");

  //Rebuild with no changes (function takes second and third args as optional)
  cabm->rebuild(new_element);

  int count_fails =  rebuild_test_counts(cabm, cabm_ref, 0);
  if ( count_fails > 0 ) {
    return count_fails;
  }

  //(Necessary) update of pID
  pID = cabm->get<0>();

  kkLidView failed = kkLidView("failed", 1);
  Kokkos::parallel_for("check no changes", cabm->capacity(),
      KOKKOS_LAMBDA (const int& ptcl) {
        const lid_t id = pID_ref(ptcl);
        if (id != -1) { // only looking at active particles
          const lid_t dest_elem = new_element(ptcl);

          bool found = false;
          for (lid_t i = dest_elem*AoSoA::vector_length; i < (dest_elem+1)*AoSoA::vector_length; ++i) {
            //printf("looking for: %d, index: %d, id: %d\n", id, i, pID(i));
            if (pID(i) == id) {
              found = true;
              break;
            } 
          }
          if (!found) {
            //printf("[ERROR] particle %d not found in element %d\n", id, dest_elem);
            failed(0) = 1;
          }
        }
      });

  fails += getLastValue<lid_t>(failed);

  Kokkos::Profiling::popRegion();
  return fails;
}
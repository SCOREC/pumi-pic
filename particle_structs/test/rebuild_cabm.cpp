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

  delete [] ptcls_per_elem;
  delete [] ids;

  CabM::kkLidView new_element("new_element", cabm->capacity());

  //pID = particle ID
  //Gives access to MTV data to set an identifier for each ptcl
  auto pID = cabm->get<0>();

  //Sum of the particles in each element
  CabM::kkLidView element_sums("element_sums", cabm->nElems());

  //Assign values to ptcls to track their movement
  auto setSameElement = PS_LAMBDA(const lid_t& e, const lid_t& p, const bool& mask) {
    if (mask) {
      new_element(p) = e;
      Kokkos::atomic_add(&(element_sums(e)), p);
    }
    else {
      new_element(p) = -1;
    }
    pID(p) = p;
  };
  ps::parallel_for(cabm, setSameElement, "setSameElement");

  //Rebuild with no changes (function takes second and third args as optional)
  cabm->rebuild(new_element);

  //(Necessary) update of pID
  pID = cabm->get<0>();

  if (cabm->nPtcls() != np) {
    fprintf(stderr, "[ERROR] CabM does not have the correct number of particles after "
        "rebuild %d (should be %d)\n", cabm->nPtcls(), np);
    ++fails;
  }
  //Sum of the particles in each element
  CabM::kkLidView new_element_sums("new_element_sums", cabm->nElems());

  kkLidView failed = kkLidView("failed", 1);
  auto checkSameElement = PS_LAMBDA(const lid_t e, const lid_t p, const bool mask) {
    const lid_t id = pID(p);
    const lid_t dest_elem = new_element(id);
    if (mask) {
      Kokkos::atomic_add(&(new_element_sums[e]), id);
      if (dest_elem != e) {
        printf("[ERROR] Particle %d was moved to incorrect element %d "
               "(should be in element %d)\n", id, e, dest_elem);
        failed(0) = 1;
      }
    }
  };
  ps::parallel_for(cabm, checkSameElement, "checkSameElement");
  fails += getLastValue<lid_t>(failed);

  CabM::kkLidView failed2("failed2", 1);
  auto checkElementSums = KOKKOS_LAMBDA(const int i) {
    const lid_t old_sum = element_sums(i);
    const lid_t new_sum = new_element_sums(i);
    if (old_sum != new_sum) {
      printf("Sum of particle ids on element %d do not match. Old: %d New: %d\n",
             i, old_sum, new_sum);
      failed2(0) = 1;
    }
  };
  Kokkos::parallel_for(cabm->nElems(), checkElementSums, "checkElementSums");
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

  CabM* cabm = new CabM(po, ne, np, ptcls_per_elem_v, element_gids_v);

  delete [] ptcls_per_elem;
  delete [] ids;

  CabM::kkLidView new_element("new_element", cabm->capacity());

  auto pID = cabm->get<0>();

  //Sum of the particles in each element
  CabM::kkLidView element_sums("element_sums", cabm->nElems());

  //Assign ptcl identifiers
  auto setElement = PS_LAMBDA(const lid_t e, const lid_t p, const bool mask) {
    if (mask) {
      new_element(p) = (e*3 + p) % ne; //assign to diff elems
      Kokkos::atomic_add(&(element_sums(new_element(p))), p);
    }
    else
      new_element(p) = -1;

    pID(p) = p;
  };
  ps::parallel_for(cabm, setElement, "setElement");

  //Rebuild with no changes
  cabm->rebuild(new_element);

  if (cabm->nPtcls() != np) {
    fprintf(stderr, "[ERROR] CabM does not have the correct number of particles after "
            "rebuild %d (should be %d)\n", cabm->nPtcls(), np);
    ++fails;
  }

  //(Necessary) update of pID
  pID = cabm->get<0>();

  //Sum of the particles in each element
  CabM::kkLidView new_element_sums("new_element_sums", cabm->nElems());

  kkLidView failed = kkLidView("failed", 1);
  auto checkElement = PS_LAMBDA(const lid_t e, const lid_t p, const bool mask) {
    const lid_t id = pID(p);
    const lid_t dest_elem = new_element(id);
    if (mask) {
      Kokkos::atomic_add(&(new_element_sums[e]), id);
      if (dest_elem != e) {
        printf("[ERROR] Particle %d was moved to incorrect element %d "
               "(should be in element %d)\n", id, e, dest_elem);
        failed(0) = 1;
      }
    }
  };
  ps::parallel_for(cabm, checkElement, "checkElement");
  fails += getLastValue<lid_t>(failed);

    CabM::kkLidView failed2("failed2", 1);
    auto checkElementSums = KOKKOS_LAMBDA(const int i) {
      const lid_t old_sum = element_sums(i);
      const lid_t new_sum = new_element_sums(i);
      if (old_sum != new_sum) {
        printf("Sum of particle ids on element %d do not match. Old: %d New: %d\n",
               i, old_sum, new_sum);
        failed2(0) = 1;
      }
    };
    Kokkos::parallel_for(cabm->nElems(), checkElementSums, "checkElementSums");
    fails += getLastValue<lid_t>(failed);

  Kokkos::Profiling::popRegion();
  return fails;
}
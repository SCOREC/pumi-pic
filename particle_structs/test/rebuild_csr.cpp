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

//using particle_structs::CSR;
using particle_structs::MemberTypes;
using particle_structs::getLastValue;
using particle_structs::lid_t;
typedef Kokkos::DefaultExecutionSpace exe_space;
typedef MemberTypes<int> Type;
typedef particle_structs::CSR<Type> CSR;

bool rebuildNoChanges(int ne_in, int np_in,int distribution);
bool rebuildNewElems(int ne_in, int np_in,int distribution);
bool rebuildNewPtcls(int ne_in, int np_in,int distribution);
bool rebuildPtclsDestroyed(int ne_in, int np_in,int distribution);
bool rebuildNewAndDestroyed(int ne_in, int np_in,int distribution);


int main(int argc, char* argv[]){
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
    printf("Rebuild CSR Test: \nne = %d\nnp = %d\ndistribtuion = %d\n",
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

    //check new elements are added properly and all elements assgined correct
    fprintf(stderr,"RebuildNewPtcls\n");
    fails += rebuildNewPtcls(number_elements,number_particles,distribution);

    //check for particles that were removed and the rest in their correct loc
    fprintf(stderr,"RebuildPtclsDestroyed\n");
    fails += rebuildPtclsDestroyed(number_elements,number_particles,distribution);

    //complete check of rebuild functionality
    fprintf(stderr,"RebuildNewAndDestroyed\n");
    fails += rebuildNewAndDestroyed(number_elements,number_particles,distribution);

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
  CSR::kkLidView ptcls_per_elem_v("ptcls_per_elem_v",ne);
  CSR::kkGidView element_gids_v("",0);
  particle_structs::hostToDevice(ptcls_per_elem_v,ptcls_per_elem);

  //Structure to perform rebuild on
  CSR* csr = new CSR(po, ne, np, ptcls_per_elem_v, element_gids_v);

  delete [] ptcls_per_elem;
  delete [] ids;

  CSR::kkLidView new_element("new_element", csr->capacity());

  //pID = particle ID
  //Gives access to MTV data to set an identifier for each ptcl
  auto pID = csr->get<0>();

  //Sum of the particles in each element
  CSR::kkLidView element_sums("element_sums", csr->nElems());

  //Assign values to ptcls to track their movement
  auto setSameElement = PS_LAMBDA(const lid_t e, const lid_t p, const bool mask) {
    if (mask) {
      new_element(p) = e;
      Kokkos::atomic_add(&(element_sums(e)), p);
    }
    else
      new_element(p) = -1;

    pID(p) = p;
  };
  ps::parallel_for(csr, setSameElement, "setSameElement");

  //Rebuild with no changes (function takes second and third args as optional)
  csr->rebuild(new_element);

  //(Necessary) updates of pID
  pID = csr->get<0>();

  if (csr->nPtcls() != np) {
    fprintf(stderr, "[ERROR] CSR does not have the correct number of particles after "
            "rebuild %d (should be %d)\n", csr->nPtcls(), np);
    ++fails;
  }
  //Sum of the particles in each element
  CSR::kkLidView new_element_sums("new_element_sums", csr->nElems());

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
  ps::parallel_for(csr, checkSameElement, "checkSameElement");
  fails += getLastValue<lid_t>(failed);

  CSR::kkLidView failed2("failed2", 1);
  auto checkElementSums = KOKKOS_LAMBDA(const int i) {
    const lid_t old_sum = element_sums(i);
    const lid_t new_sum = new_element_sums(i);
    if (old_sum != new_sum) {
      printf("Sum of particle ids on element %d do not match. Old: %d New: %d\n",
             i, old_sum, new_sum);
      failed2(0) = 1;
    }
  };
  Kokkos::parallel_for(csr->nElems(), checkElementSums, "checkElementSums");
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
  CSR::kkLidView ptcls_per_elem_v("ptcls_per_elem_v",ne);
  CSR::kkGidView element_gids_v("",0);
  particle_structs::hostToDevice(ptcls_per_elem_v,ptcls_per_elem);

  CSR* csr = new CSR(po, ne, np, ptcls_per_elem_v, element_gids_v);

  delete [] ptcls_per_elem;
  delete [] ids;

  CSR::kkLidView new_element("new_element", csr->capacity());

  auto pID = csr->get<0>();

  //Sum of the particles in each element
  CSR::kkLidView element_sums("element_sums", csr->nElems());

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
  ps::parallel_for(csr, setElement, "setElement");

  //Rebuild with no changes
  csr->rebuild(new_element);

  if (csr->nPtcls() != np) {
    fprintf(stderr, "[ERROR] CSR does not have the correct number of particles after "
            "rebuild %d (should be %d)\n", csr->nPtcls(), np);
    ++fails;
  }

  //(Necessary) update of pID
  pID = csr->get<0>();

  //Sum of the particles in each element
  CSR::kkLidView new_element_sums("new_element_sums", csr->nElems());

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
  ps::parallel_for(csr, checkElement, "checkElement");
  fails += getLastValue<lid_t>(failed);

    CSR::kkLidView failed2("failed2", 1);
    auto checkElementSums = KOKKOS_LAMBDA(const int i) {
      const lid_t old_sum = element_sums(i);
      const lid_t new_sum = new_element_sums(i);
      if (old_sum != new_sum) {
        printf("Sum of particle ids on element %d do not match. Old: %d New: %d\n",
               i, old_sum, new_sum);
        failed2(0) = 1;
      }
    };
    Kokkos::parallel_for(csr->nElems(), checkElementSums, "checkElementSums");
    fails += getLastValue<lid_t>(failed);

  Kokkos::Profiling::popRegion();
  return fails;
}

//Rebuild test with new particles added only
bool rebuildNewPtcls(int ne_in, int np_in,int distribution){
  Kokkos::Profiling::pushRegion("rebuildNewPtcls");
  int fails = 0;

  //Test construction based on SCS testing
  int ne = ne_in;
  int np = np_in;
  int* ptcls_per_elem = new int[ne];
  std::vector<int>* ids = new std::vector<int>[ne];
  distribute_particles(ne, np, distribution, ptcls_per_elem, ids);
  Kokkos::TeamPolicy<exe_space> po(32,Kokkos::AUTO);
  CSR::kkLidView ptcls_per_elem_v("ptcls_per_elem_v",ne);
  CSR::kkGidView element_gids_v("",0);
  particle_structs::hostToDevice(ptcls_per_elem_v,ptcls_per_elem);

  CSR* csr = new CSR(po, ne, np, ptcls_per_elem_v, element_gids_v);

  delete [] ptcls_per_elem;
  delete [] ids;

  CSR::kkLidView new_element("new_element", csr->capacity());

  auto pID = csr->get<0>();

  //Assign ptcl identifiers
  auto setElement = PS_LAMBDA(const lid_t e, const lid_t p, const bool mask) {
    if (mask)
      new_element(p) = (e*3 + p + 2) % ne; //assign to diff elems
    else
      new_element(p) = -1;

    pID(p) = p;
  };
  ps::parallel_for(csr, setElement, "setElement");

  //////////////////////////////////////////////////////////////
  //Introduce new particles
  int nnp = 5; //number new ptcls
  CSR::kkLidView new_particle_elements("new_particle_elements", nnp);
  auto new_particles = particle_structs::createMemberViews<Type>(nnp);
  auto new_particle_access = particle_structs::getMemberView<Type,0>(new_particles);
  lid_t cap = csr->capacity();
  //Assign new ptcl elements and identifiers
  Kokkos::parallel_for("new_particle_elements", nnp,
      KOKKOS_LAMBDA (const int& i){
      new_particle_elements(i) = i%ne;
      new_particle_access(i) = i+cap;
  });

  //Rebuild with new ptcls
  csr->rebuild(new_element, new_particle_elements, new_particles);

  if (csr->nPtcls() != np + nnp) {
    fprintf(stderr, "[ERROR] CSR does not have the correct number of particles after "
            "rebuild with new particles %d (should be %d)\n", csr->nPtcls(), np + nnp);
    ++fails;
  }

  pID = csr->get<0>(); //new size

  kkLidView failed = kkLidView("failed", 1);
  auto checkElement = PS_LAMBDA(const lid_t e, const lid_t p, const bool mask) {
    const lid_t id = pID(p);
    if (mask) {
      if (id < cap) { //Check old particles
        const lid_t dest_elem = new_element(id);

        if (dest_elem != e) {
          printf("[ERROR] Particle %d was moved to incorrect element %d "
                 "(should be in element %d)\n", id, e, dest_elem);
          failed(0) = 1;
        }
      }
      else { //Check new particles
        const lid_t i = id - cap;
        const lid_t dest_elem = new_particle_elements(i);
        if (e != dest_elem) {
          printf("[ERROR] New particle %d was added to incorrect element %d "
                 "(should be in element %d)\n", id, e, dest_elem);
          failed(0) = 1;
        }
      }
    }
  };

  fails += getLastValue<lid_t>(failed);

  Kokkos::Profiling::popRegion();
  return fails;
}

//Rebuild test with existing particles destroyed only
bool rebuildPtclsDestroyed(int ne_in, int np_in,int distribution){
  Kokkos::Profiling::pushRegion("rebuildPtclsDestroyed");
  int fails = 0;

  //Test construction based on SCS testing
  int ne = ne_in;
  int np = np_in;
  int* ptcls_per_elem = new int[ne];
  std::vector<int>* ids = new std::vector<int>[ne];
  distribute_particles(ne, np, distribution, ptcls_per_elem, ids);
  Kokkos::TeamPolicy<exe_space> po(32,Kokkos::AUTO);
  CSR::kkLidView ptcls_per_elem_v("ptcls_per_elem_v",ne);
  CSR::kkGidView element_gids_v("",0);
  particle_structs::hostToDevice(ptcls_per_elem_v,ptcls_per_elem);

  CSR* csr = new CSR(po, ne, np, ptcls_per_elem_v, element_gids_v);

  delete [] ptcls_per_elem;
  delete [] ids;

  CSR::kkLidView new_element("new_element", csr->capacity());

  auto pID = csr->get<0>();

  CSR::kkLidView num_removed("num_removed", 1);
  //Remove every 7th particle, keep other particles in same element
  auto assign_ptcl_elems = PS_LAMBDA(const int& e, const int& p, const bool mask){
    new_element(p) = e;
    if(p%7 == 0) {
      new_element(p) = -1;
      Kokkos::atomic_add(&(num_removed(0)), 1);
    }
    pID(p) = p;
  };
  csr->parallel_for(assign_ptcl_elems, "assing ptcl elems");
  int nremoved = pumipic::getLastValue(num_removed);

  //Rebuild with particles removed
  csr->rebuild(new_element);

  if (csr->nPtcls() != np - nremoved) {
    fprintf(stderr, "[ERROR] CSR does not have the correct number of particles after "
            "rebuild after removing particles %d (should be %d)\n",
            csr->nPtcls(), np - nremoved);
    ++fails;
  }

  pID = csr->get<0>();

  lid_t particles = csr->capacity();
  kkLidView failed = kkLidView("failed", 1);
  auto checkElement = PS_LAMBDA(const lid_t e, const lid_t p, const bool mask) {
    if (mask) {
      const lid_t id = pID(p);
      const lid_t dest_elem = new_element(id);
      if (dest_elem != e) {
        printf("[ERROR] Particle %d was moved to incorrect element %d "
               "(should be in element %d)\n", id, e, dest_elem);
        failed(0) = 1;
      }
      if (id % 7 == 0) {
        printf("[ERROR] Particle %d was not removed during rebuild\n", id);
        failed(0) = 1;
      }
    }
  };
  ps::parallel_for(csr, checkElement, "checkElement");

  fails += getLastValue<lid_t>(failed);

  Kokkos::Profiling::popRegion();
  return fails;
}

//Rebuild test with particles added and destroyed
bool rebuildNewAndDestroyed(int ne_in, int np_in,int distribution){
  Kokkos::Profiling::pushRegion("rebuildNewAndDestroyed");
  int fails = 0;

  //Test construction based on SCS testing
  int ne = ne_in;
  int np = np_in;
  int* ptcls_per_elem = new int[ne];
  std::vector<int>* ids = new std::vector<int>[ne];
  distribute_particles(ne, np, distribution, ptcls_per_elem, ids);
  Kokkos::TeamPolicy<exe_space> po(32,Kokkos::AUTO);
  CSR::kkLidView ptcls_per_elem_v("ptcls_per_elem_v",ne);
  CSR::kkGidView element_gids_v("",0);
  particle_structs::hostToDevice(ptcls_per_elem_v,ptcls_per_elem);

  CSR* csr = new CSR(po, ne, np, ptcls_per_elem_v, element_gids_v);

  delete [] ptcls_per_elem;
  delete [] ids;

  CSR::kkLidView new_element("new_element", csr->capacity());

  auto pID = csr->get<0>();

  CSR::kkLidView num_removed("num_removed", 1);
  //Remove every 7th particle and move others to new element
  auto assign_ptcl_elems = PS_LAMBDA(const int& e, const int& p, const bool mask){
    if (mask) {
      new_element(p) = (3*e+7)%ne;
      if(p%7 == 0) {
        new_element(p) = -1;
        Kokkos::atomic_add(&(num_removed(0)), 1);
      }
      pID(p) = p;
    }
  };
  csr->parallel_for(assign_ptcl_elems, "assing ptcl elems");
  int nremoved = pumipic::getLastValue(num_removed);

  //////////////////////////////////////////////////////////////
  //Introduce new particles
  int nnp = 5;
  CSR::kkLidView new_particle_elements("new_particle_elements", nnp);
  auto new_particles = particle_structs::createMemberViews<Type>(nnp);
  auto new_particle_access = particle_structs::getMemberView<Type,0>(new_particles);
  lid_t cap = csr->capacity();

  Kokkos::parallel_for("new_particle_elements", nnp,
      KOKKOS_LAMBDA (const int& i){
      new_particle_elements(i) = i%ne;
      new_particle_access(i) = i+cap;
  });

  //Rebuild with elements removed
  csr->rebuild(new_element,new_particle_elements,new_particles);

  if (csr->nPtcls() != np + nnp - nremoved) {
    fprintf(stderr, "[ERROR] CSR does not have the correct number of particles after "
            "rebuild with new and removed particles %d (should be %d)\n",
            csr->nPtcls(), np + nnp - nremoved);
    ++fails;
  }

  pID = csr->get<0>();

  //need variable here bc can't access csr on device
  const lid_t new_cap = csr->capacity();
  kkLidView failed = kkLidView("failed", 1);

  auto checkElement = PS_LAMBDA(const lid_t e, const lid_t p, const bool mask) {
    const lid_t id = pID(p);
    if (mask) {
      if (id < cap) { //Check old particles
        const lid_t dest_elem = new_element(id);
        if (dest_elem != e) {
          printf("[ERROR] Particle %d was moved to incorrect element %d "
                 "(should be in element %d)\n", id, e, dest_elem);
          failed(0) = 1;
        }
        if (id % 7 == 0) { //Check removed particles
          printf("[ERROR] Particle %d was not removed during rebuild\n", id);
          failed(0) = 1;
        }
      }
      else { //Check new particles
        const lid_t i = id - cap;
        const lid_t dest_elem = new_particle_elements(i);
        if (e != dest_elem) {
          printf("[ERROR] New particle %d was added to incorrect element %d "
                 "(should be in element %d)\n", id, e, dest_elem);
          failed(0) = 1;
        }
      }

    }
  };
  ps::parallel_for(csr, checkElement, "checkElement");

  fails += getLastValue<lid_t>(failed);

  Kokkos::Profiling::popRegion();
  return fails;
}

#include <stdio.h>
#include <Kokkos_Core.hpp>

#include <MemberTypes.h>
#include <SellCSigma.h>

#include <psAssert.h>
#include <Distribute.h>

using particle_structs::SellCSigma;
using particle_structs::MemberTypes;
using particle_structs::distribute_particles;
using particle_structs::distribute_elements;
using particle_structs::getLastValue;
using particle_structs::lid_t;

typedef MemberTypes<int> Type;
typedef Kokkos::DefaultExecutionSpace exe_space;
typedef SellCSigma<Type,exe_space> SCS;

bool shuffleParticlesTests();
bool resortElementsTest();
bool reshuffleTests();

int main(int argc, char* argv[]) {
  MPI_Init(&argc, &argv);
  Kokkos::initialize(argc, argv);
  
  bool passed = true;
  if (!shuffleParticlesTests()) {
    passed = false;
    printf("[ERROR] shuffleParticlesTests() failed\n");
  }
  if (!resortElementsTest()) {
    passed = false;
    printf("[ERROR] resortElementsTest() failed\n");
  }
  if (!reshuffleTests()) {
    passed = false;
    printf("[ERROR] shuffleParticlesTests() failed\n");
  }

  Kokkos::finalize();
  MPI_Finalize();
  if (passed)
    printf("All tests passed\n");
  return 0;
}


bool shuffleParticlesTests() {
  int ne = 5;
  int np = 20;
  int* ptcls_per_elem = new int[ne];
  std::vector<int>* ids = new std::vector<int>[ne];
  distribute_particles(ne, np, 0, ptcls_per_elem, ids);
  int C = 4;
  Kokkos::TeamPolicy<exe_space> po(128, C);

  SCS::kkLidView ptcls_per_elem_v("ptcls_per_elem_v", ne);
  SCS::kkGidView element_gids_v("", 0);
  particle_structs::hostToDevice(ptcls_per_elem_v, ptcls_per_elem);

  SCS* scs = new SCS(po, 5, 2, ne, np, ptcls_per_elem_v, element_gids_v);
  SCS* scs2 = new SCS(po, 5, 2, ne, np, ptcls_per_elem_v,  element_gids_v);
  delete [] ptcls_per_elem;
  delete [] ids;

  scs->printFormat();
  scs->printMetrics();
  SCS::kkLidView new_element("new_element", scs->capacity());

  auto values = scs->get<0>();
  auto values2 = scs2->get<0>();
  auto sendToSelf = SCS_LAMBDA(const int& element_id, const int& particle_id, const bool mask) {
    new_element(particle_id) = element_id;
    values(particle_id) = particle_id;
    values2(particle_id) = particle_id;
  };
  scs->parallel_for(sendToSelf);

  //Rebuild with no changes  
  scs->rebuild(new_element);

  scs->printFormat();
  scs->printMetrics();
  values = scs->get<0>();

  SCS::kkLidView fail("fail",1);
  auto checkParticles = SCS_LAMBDA(int elm_id, int ptcl_id, bool mask) {
    if (mask) {
      if (values(ptcl_id) % C != values2(ptcl_id) % C) {
        printf("Particle mismatch at %d (%d != %d)\n", ptcl_id, values(ptcl_id), values2(ptcl_id));
        fail(0) = 1;
      }
    }
  };
  scs->parallel_for(checkParticles);
  if (getLastValue<lid_t>(fail) == 1) {
    printf("Value mismatch on at least one particle\n");
    return false;
  }
  auto moveParticles = SCS_LAMBDA(int elm_id, int ptcl_id, bool mask) {
    new_element(ptcl_id) = (elm_id + 2) % ne;
  };
  scs->parallel_for(moveParticles);

  scs->rebuild(new_element);

  values = scs->get<0>();

  auto printParticles = SCS_LAMBDA(int elm_id, int ptcl_id, bool mask) {
    if (mask)
      printf("Particle %d has value %d\n", ptcl_id, values(ptcl_id));
  };
  scs->parallel_for(printParticles);
  scs->printFormat();
  scs->printMetrics();
  delete scs;
  delete scs2;
  return true;
}


bool resortElementsTest() {
  int ne = 5;
  int np = 20;
  particle_structs::gid_t* gids = new particle_structs::gid_t[ne];
  distribute_elements(ne, 0, 0, 1, gids);
  int* ptcls_per_elem = new int[ne];
  std::vector<int>* ids = new std::vector<int>[ne];
  distribute_particles(ne, np, 0, ptcls_per_elem, ids);
  
  Kokkos::TeamPolicy<exe_space> po(128, 4);
  SCS::kkLidView ptcls_per_elem_v("ptcls_per_elem_v", ne);
  SCS::kkGidView element_gids_v("element_gids_v", ne);
  particle_structs::hostToDevice(ptcls_per_elem_v, ptcls_per_elem);
  particle_structs::hostToDevice(element_gids_v, gids);

  SCS* scs = new SCS(po, 5, 2, ne, np, ptcls_per_elem_v, element_gids_v);
  delete [] ptcls_per_elem;
  delete [] ids;
  delete [] gids;

  scs->printFormat();
  scs->printMetrics();
  auto values = scs->get<0>();

  SCS::kkLidView new_element("new_element", scs->capacity());
  //Remove all particles from first element
  auto moveParticles = SCS_LAMBDA(int elm_id, int ptcl_id, bool mask) {
    if (mask) {
      values(ptcl_id) = elm_id;
      if (ptcl_id % 4 == 0 && ptcl_id < 8)
        new_element(ptcl_id) = -1;
      else
        new_element(ptcl_id) = elm_id;
    }
  };
  scs->parallel_for(moveParticles);
  scs->rebuild(new_element);

  scs->printFormat();
  scs->printMetrics();
  values = scs->get<0>();
  SCS::kkLidView fail("", 1);
  auto checkParticles = SCS_LAMBDA(int elm_id, int ptcl_id, bool mask) {
    if (mask) {
      if (values(ptcl_id) != elm_id) {
        fail(0) = 1;
      }
    }
  };
  scs->parallel_for(checkParticles);

  if (getLastValue<lid_t>(fail) == 1) {
    printf("Value mismatch on some particles\n");
    return false;
  }
  delete scs;
  return true;
}


bool reshuffleTests() {

  //Move nothing (should only use reshuffle)
  printf("\n\nReshuffle Tests\n");

  int ne = 5;
  int np = 1;
  int* ptcls_per_elem = new int[ne];
  std::vector<int>* ids = new std::vector<int>[ne];
  distribute_particles(ne, np, 0, ptcls_per_elem, ids);
  
  Kokkos::TeamPolicy<exe_space> po(128, 4);
  SCS::kkLidView ptcls_per_elem_v("ptcls_per_elem_v", ne);
  SCS::kkGidView element_gids_v("element_gids_v", 0);
  particle_structs::hostToDevice(ptcls_per_elem_v, ptcls_per_elem);

  SCS* scs = new SCS(po, ne, np, ne, np, ptcls_per_elem_v, element_gids_v);
  delete [] ptcls_per_elem;
  delete [] ids;

  scs->printFormat();
  scs->printMetrics();
  SCS::kkLidView new_element("new_element", scs->capacity());

  //Shuffle
  auto sendToSelf = SCS_LAMBDA(const int& element_id, const int& particle_id, const bool mask) {
    new_element(particle_id) = element_id;
  };
  scs->parallel_for(sendToSelf);

  scs->rebuild(new_element);

  scs->printFormat();
  scs->printMetrics();
  //Shuffle
  auto sendToChunk = SCS_LAMBDA(const int& element_id, const int& particle_id, const bool mask) {
    new_element(particle_id) = 2;
  };
  scs->parallel_for(sendToChunk);

  scs->rebuild(new_element);

  scs->printFormat();
  scs->printMetrics();
  //Needs Rebuild
  auto sendOffChunk = SCS_LAMBDA(const int& element_id, const int& particle_id, const bool mask) {
    new_element(particle_id) = 4;
  };
  scs->parallel_for(sendOffChunk);

  scs->rebuild(new_element);

  scs->printFormat();
  scs->printMetrics();
  return true;
}

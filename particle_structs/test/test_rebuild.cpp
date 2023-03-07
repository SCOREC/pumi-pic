#include <particle_structs.hpp>
#include "read_particles.hpp"

//Rebuild test with no changes to structure
int rebuildNoChanges(const char* name, PS* structure) {
  printf("rebuildNoChanges %s, rank %d\n", name, comm_rank);
  int fails = 0;
  int np = structure->nPtcls();

  auto pID = structure->get<0>();
  kkLidView new_element("new_element", structure->capacity());
  kkLidView element_sums("element_sums", structure->nElems());
  auto setSameElement = PS_LAMBDA(const lid_t& e, const lid_t& p, const bool& mask) {
    if (mask) {
      new_element(p) = e;
      Kokkos::atomic_add(&(element_sums(e)), p);
    }
    else
      new_element(p) = -1;
    pID(p) = p;
  };
  ps::parallel_for(structure, setSameElement, "setSameElement");
  //Rebuild with no changes
  structure->rebuild(new_element);

  pID = structure->get<0>();

  if (structure->nPtcls() != np) {
    fprintf(stderr, "[ERROR] %s does not have the correct number of particles after "
        "rebuild %d (should be %d)\n", name, structure->nPtcls(), np);
    ++fails;
  }

  kkLidView new_element_sums("new_element_sums", structure->nElems());
  kkLidView failed = kkLidView("failed", 1);
  auto checkSameElement = PS_LAMBDA(const lid_t& e, const lid_t& p, const bool& mask) {
    const lid_t id = pID(p);
    const lid_t dest_elem = new_element(id);
    if (mask) {
      Kokkos::atomic_add(&(new_element_sums[e]), id);
      if (dest_elem != e) {
        printf("[ERROR] Particle %d was moved to incorrect element %d on %s "
               "(should be in element %d)\n", id, e, name, dest_elem);
        failed(0) = 1;
      }
    }
  };
  ps::parallel_for(structure, checkSameElement, "checkSameElement");
  fails += ps::getLastValue<lid_t>(failed);

  failed = kkLidView("failed", 1);
  auto checkElementSums = KOKKOS_LAMBDA(const int i) {
    const lid_t old_sum = element_sums(i);
    const lid_t new_sum = new_element_sums(i);
    if (old_sum != new_sum) {
      printf("[ERROR] Sum of particle ids on element %d do not match on %s. Old: %d New: %d\n",
             i, name, old_sum, new_sum);
      failed(0) = 1;
    }
  };
  Kokkos::parallel_for("checkElementSums", structure->nElems(), checkElementSums);
  fails += ps::getLastValue<lid_t>(failed);

  return fails;
}

//Rebuild test with no new particles, but reassigned particle elements
int rebuildNewElems(const char* name, PS* structure) {
  printf("rebuildNewElems %s, rank %d\n", name, comm_rank);
  int fails = 0;
  int np = structure->nPtcls();
  int ne = structure->nElems();

  auto pID = structure->get<0>();
  kkLidView new_element("new_element", structure->capacity());
  kkLidView element_sums("element_sums", structure->nElems());
  auto setElement = PS_LAMBDA(const lid_t& e, const lid_t& p, const bool& mask) {
    if (mask) {
      new_element(p) = (e*3 + p) % ne; //assign to diff elems
      Kokkos::atomic_add(&(element_sums(new_element(p))), p);
    }
    else
      new_element(p) = -1;
    pID(p) = p;
  };
  ps::parallel_for(structure, setElement, "setElement");
  //Rebuild with moving particles
  structure->rebuild(new_element);

  if (structure->nPtcls() != np) {
    fprintf(stderr, "[ERROR] %s does not have the correct number of particles after "
            "rebuild %d (should be %d)\n", name, structure->nPtcls(), np);
    ++fails;
  }

  pID = structure->get<0>();

  kkLidView new_element_sums("new_element_sums", structure->nElems());
  kkLidView failed = kkLidView("failed", 1);
  auto checkElement = PS_LAMBDA(const lid_t& e, const lid_t& p, const bool& mask) {
    if (mask) {
      const lid_t id = pID(p);
      const lid_t dest_elem = new_element(id);
      Kokkos::atomic_add(&(new_element_sums[e]), id);
      if (dest_elem != e) {
        printf("[ERROR] Particle %d was moved to incorrect element %d on %s "
               "(should be in element %d)\n", id, e, name, dest_elem);
        failed(0) = 1;
      }
    }
  };
  ps::parallel_for(structure, checkElement, "checkElement");
  fails += ps::getLastValue<lid_t>(failed);

  failed == kkLidView("failed", 1);
  auto checkElementSums = KOKKOS_LAMBDA(const int i) {
    const lid_t old_sum = element_sums(i);
    const lid_t new_sum = new_element_sums(i);
    if (old_sum != new_sum) {
      printf("[ERROR] Sum of particle ids on element %d do not match on %s. Old: %d New: %d\n",
              i, name, old_sum, new_sum);
      failed(0) = 1;
    }
  };
  Kokkos::parallel_for("checkElementSums", structure->nElems(), checkElementSums);
  fails += ps::getLastValue<lid_t>(failed);

  return fails;
}

//Rebuild test with new particles added only
int rebuildNewPtcls(const char* name, PS* structure) {
  printf("rebuildNewPtcls %s, rank %d\n", name, comm_rank);
  int fails = 0;
  int np = structure->nPtcls();
  int nnp = structure->capacity()/2;
  int ne = structure->nElems();

  auto pID = structure->get<0>();
  kkLidView new_element("new_element", structure->capacity());
  auto setElement = PS_LAMBDA(const lid_t& e, const lid_t& p, const bool& mask) {
    if (mask)
      new_element(p) = (e*3 + p + 2) % ne; //assign to diff elems
    else
      new_element(p) = -1;
    pID(p) = p;
  };
  ps::parallel_for(structure, setElement, "setElement");

  kkLidView new_particle_elements("new_particle_elements", nnp);
  auto new_particles = particle_structs::createMemberViews<Types>(nnp);
  auto new_particle_access = particle_structs::getMemberView<Types,0>(new_particles);
  lid_t cap = structure->capacity();
  //Assign new ptcl elements and identifiers
  Kokkos::parallel_for("new_particle_elements", nnp,
      KOKKOS_LAMBDA (const int& i){
      new_particle_elements(i) = i%ne;
      new_particle_access(i) = i+cap;
  });
  //Rebuild with new ptcls
  structure->rebuild(new_element, new_particle_elements, new_particles);

  if (structure->nPtcls() != np + nnp) {
    fprintf(stderr, "[ERROR] %s does not have the correct number of particles after "
            "rebuild with new particles %d (should be %d)\n", name, structure->nPtcls(), np + nnp);
    ++fails;
  }

  pID = structure->get<0>();

  kkLidView failed = kkLidView("failed", 1);
  auto checkElement = PS_LAMBDA(const lid_t& e, const lid_t& p, const bool& mask) {
    if (mask) {
      const lid_t id = pID(p);
      if (id < cap) { //Check old particles
        const lid_t dest_elem = new_element(id);
        if (dest_elem != e) {
          printf("[ERROR] Particle %d was moved to incorrect element %d on %s "
                 "(should be in element %d)\n", id, e, name, dest_elem);
          failed(0) = 1;
        }
      }
      else { //Check new particles
        const lid_t i = id - cap;
        const lid_t dest_elem = new_particle_elements(i);
        if (e != dest_elem) {
          printf("[ERROR] New particle %d was added to incorrect element %d on %s "
                 "(should be in element %d)\n", id, e, name, dest_elem);
          failed(0) = 1;
        }
      }
    }
  };

  fails += ps::getLastValue<lid_t>(failed);

  ps::destroyViews<Types>(new_particles);
  return fails;
}

//Rebuild test with existing particles destroyed only
int rebuildPtclsDestroyed(const char* name, PS* structure) {
  printf("rebuildPtclsDestroyed %s, rank %d\n", name, comm_rank);
  int fails = 0;
  int np = structure->nPtcls();

  auto pID = structure->get<0>();
  kkLidView new_element("new_element", structure->capacity());
  kkLidView num_removed("num_removed", 1);
  //Remove every 7th particle, keep other particles in same element
  auto assign_ptcl_elems = PS_LAMBDA(const int& e, const int& p, const bool& mask){
    if (mask) {
      new_element(p) = e;
      if (p%7 == 0) {
        new_element(p) = -1;
        Kokkos::atomic_add(&(num_removed(0)), 1);
      }
      pID(p) = p;
    }
  };
  ps::parallel_for(structure, assign_ptcl_elems, "assign ptcl elems");
  int nremoved = ps::getLastValue(num_removed);
  structure->rebuild(new_element);

  if (structure->nPtcls() != np - nremoved) {
    fprintf(stderr, "[ERROR] %s does not have the correct number of particles after "
            "rebuild after removing particles %d (should be %d)\n",
            name, structure->nPtcls(), np - nremoved);
    ++fails;
  }

  pID = structure->get<0>();

  kkLidView failed = kkLidView("failed", 1);
  auto checkElement = PS_LAMBDA(const lid_t& e, const lid_t& p, const bool& mask) {
    if (mask) {
      const lid_t id = pID(p);
      const lid_t dest_elem = new_element(id);
      if (id%7 == 0) {
        printf("[ERROR] Particle %d (%d, %d) was not removed during rebuild on %s\n", id, e, p, name);
        failed(0) = 1;
      }
      else if (dest_elem != e) {
        printf("[ERROR] Particle %d was moved to incorrect element %d on %s "
               "(should be in element %d)\n", id, e, name, dest_elem);
        failed(0) = 1;
      }
    }
  };
  ps::parallel_for(structure, checkElement, "checkElement");

  fails += ps::getLastValue<lid_t>(failed);

  return fails;
}

//Rebuild test with particles added and destroyed
int rebuildNewAndDestroyed(const char* name, PS* structure) {
  printf("rebuildNewAndDestroyed %s, rank %d\n", name, comm_rank);
  int fails = 0;
  int np = structure->nPtcls();
  int nnp = structure->capacity()/2;
  int ne = structure->nElems();

  auto pID = structure->get<0>();
  kkLidView new_element("new_element", structure->capacity());
  kkLidView num_removed("num_removed", 1);
  //Remove every 7th particle and move others to new element
  auto assign_ptcl_elems = PS_LAMBDA(const int& e, const int& p, const bool& mask){
    if (mask) {
      new_element(p) = (3*e+7)%ne;
      if (p%7 == 0) {
        new_element(p) = -1;
        Kokkos::atomic_add(&(num_removed(0)), 1);
      }
      pID(p) = p;
    }
  };
  parallel_for(structure, assign_ptcl_elems, "assing ptcl elems");
  int nremoved = ps::getLastValue(num_removed);

  kkLidView new_particle_elements("new_particle_elements", nnp);
  auto new_particles = particle_structs::createMemberViews<Types>(nnp);
  auto new_particle_access = particle_structs::getMemberView<Types,0>(new_particles);
  lid_t cap = structure->capacity();
  Kokkos::parallel_for("new_particle_elements", nnp,
      KOKKOS_LAMBDA (const int& i){
      new_particle_elements(i) = i%ne;
      new_particle_access(i) = i+cap;
  });
  //Rebuild with elements removed
  structure->rebuild(new_element, new_particle_elements, new_particles);

  if (structure->nPtcls() != np + nnp - nremoved) {
    fprintf(stderr, "[ERROR] %s does not have the correct number of particles after "
            "rebuild with new and removed particles %d (should be %d)\n",
            name, structure->nPtcls(), np + nnp - nremoved);
    ++fails;
  }

  pID = structure->get<0>();

  kkLidView failed = kkLidView("failed", 1);
  auto checkElement = PS_LAMBDA(const lid_t& e, const lid_t& p, const bool& mask) {
    if (mask) {
      const lid_t id = pID(p);
      if (id < cap) { //Check old particles
        const lid_t dest_elem = new_element(id);
        if (id % 7 == 0) { //Check removed particles
          printf("[ERROR] Particle %d in element %d was not removed during rebuild on %s\n", id, e, name);
          failed(0) = 1;
        }
        else if (dest_elem != e) {
          printf("[ERROR] Particle %d was moved to incorrect element %d on %s "
                 "(should be in element %d)\n", id, e, name, dest_elem);
          failed(0) = 1;
        }
      }
      else { //Check new particles
        const lid_t i = id - cap;
        const lid_t dest_elem = new_particle_elements(i);
        if (e != dest_elem) {
          printf("[ERROR] New particle %d was added to incorrect element %d on %s "
                 "(should be in element %d)\n", id, e, name, dest_elem);
          failed(0) = 1;
        }
      }
    }
  };
  ps::parallel_for(structure, checkElement, "checkElement");

  fails += ps::getLastValue<lid_t>(failed);

  ps::destroyViews<Types>(new_particles);
  return fails;
}

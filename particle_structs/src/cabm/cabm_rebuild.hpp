#pragma once

#include <ppTiming.hpp>

namespace pumipic {
  /**
   * Fully rebuild the AoSoA with these new parent SoAs and particles
   *     by copying into a new AoSoA and overwriting the old one.
   *     Delete particles with new_element(ptcl) < 0
   * @param[in] new_element view of ints with new elements for each particle
   * @param[in] new_particle_elements view of ints, representing which elements
   *    particle reside in
   * @param[in] new_particles array of views filled with particle data
   * @exception new_particle_elements(ptcl) < 0,
   *    undefined behavior for new_particle_elements.size() != sizeof(new_particles),
   *    undefined behavior for numberoftypes(new_particles) != numberoftypes(DataTypes)
  */
  template <class DataTypes, typename MemSpace>
  void CabM<DataTypes, MemSpace>::rebuild(kkLidView new_element,
                                         kkLidView new_particle_elements,
                                         MTVs new_particles) {
    Kokkos::Profiling::pushRegion("CabM Rebuild");
    Kokkos::Timer overall_timer; // timer for rebuild

    const auto soa_len = AoSoA_t::vector_length;
    kkLidView elmDegree("elmDegree", num_elems);
    kkLidView elmOffsets("elmOffsets", num_elems);
    const auto activeSliceIdx = aosoa_.number_of_members-1;
    auto active = Cabana::slice<activeSliceIdx>(aosoa_);
    auto elmDegree_d = Kokkos::create_mirror_view_and_copy(memory_space(), elmDegree);

    //first loop to count number of particles per new element (atomic)
    assert(new_element.size() == capacity_);
    kkLidView num_removed_d("num_removed_d", 1); // for counting particles to be removed
    auto atomic = KOKKOS_LAMBDA(const int& soa, const int& tuple) {
      if (active.access(soa,tuple) == 1){
        auto parent = new_element((soa*soa_len)+tuple);
        if (parent >= 0) { // count particles to be deleted
          Kokkos::atomic_increment<int>(&elmDegree_d(parent));
        } else {
          Kokkos::atomic_increment<int>(&num_removed_d(0));
        }
      }
    };
    Cabana::SimdPolicy<soa_len,execution_space> simd_policy( 0, capacity_ );
    Cabana::simd_parallel_for( simd_policy, atomic, "atomic" );
    lid_t num_removed = getLastValue(num_removed_d);

    // also add the number of particles in new_particle_elements for later (atomic)
    if (new_particle_elements.size() > 0 && new_particles != NULL) {
      Kokkos::parallel_for("add_particle_setup", new_particle_elements.size(),
        KOKKOS_LAMBDA(const lid_t ptcl_id) {
          auto parent = new_particle_elements(ptcl_id);
          assert(parent >= 0); // new particles should have a destination element
          Kokkos::atomic_increment<int>(&elmDegree_d(parent));
        });
    }

    RecordTime("CabM count active particles", overall_timer.seconds());
    Kokkos::Timer existing_timer; // timer for moving/deleting particles

    auto elmDegree_h = Kokkos::create_mirror_view_and_copy(host_space(), elmDegree_d);
    //prepare a new aosoa to store the shuffled particles
    kkLidView newOffset_d = buildOffset(elmDegree_d);
    const auto newNumSoa = getLastValue(newOffset_d);
    const auto newCapacity = newNumSoa*soa_len;
    auto newAosoa = makeAoSoA(newCapacity, newNumSoa);
    Kokkos::View<lid_t*, host_space> elmPtclCounter_h("elmPtclCounter_device", num_elems);
    auto elmPtclCounter_d = Kokkos::create_mirror_view_and_copy(memory_space(), elmPtclCounter_h);
    auto newActive = Cabana::slice<activeSliceIdx>(newAosoa);
    auto aosoa_cpy = aosoa_; // copy of member variable aosoa_ (necessary, Kokkos doesn't like member variables)
    auto copyPtcls = KOKKOS_LAMBDA(const int& soa,const int& tuple){
        if (active.access(soa,tuple) == 1){
          //Compute the destSoa based on the destParent and an array of
          // counters for each destParent tracking which particle is the next
          // free position. Use atomic fetch and incriment with the
          // 'elmPtclCounter_d' array.
          auto destParent = new_element(soa*soa_len + tuple);
          if ( destParent >= 0 ) { // delete particles with negative destination element
            auto occupiedTuples = Kokkos::atomic_fetch_add<int>(&elmPtclCounter_d(destParent), 1);
            auto oldTuple = aosoa_cpy.getTuple(soa*soa_len + tuple);
            auto firstSoa = newOffset_d(destParent);
            // use newOffset_d to figure out which soa is the first for destParent
            newAosoa.setTuple(firstSoa*soa_len + occupiedTuples, oldTuple);
          }
        }
      };
    Cabana::simd_parallel_for(simd_policy, copyPtcls, "copyPtcls");

    // update member variables
    num_soa_ = newNumSoa;
    capacity_ = newCapacity;
    offsets = newOffset_d;
    num_ptcls = num_ptcls - num_removed;
    parentElms_ = getParentElms(num_elems, num_soa_, offsets);

    RecordTime("CabM move/destroy existing particles", existing_timer.seconds());
    Kokkos::Timer add_timer; // timer for adding particles

    // add new particles
    if (new_particle_elements.size() > 0 && new_particles != NULL) {
      auto new_num_ptcls = new_particle_elements.size();

      // View for tracking particle indices in elements
      kkLidView ptcl_elm_indices("ptcl_elm_indices", num_elems);
      Kokkos::parallel_for("fill_zeros", num_elems,
        KOKKOS_LAMBDA(const lid_t& elm) {
          ptcl_elm_indices(elm) = 0;
        });
      
      // count all CURRENTLY active particles
      auto fill_elm_indices = PS_LAMBDA(const lid_t& elm, const lid_t& ptcl, const bool& mask) {
        ptcl_elm_indices(elm) = Kokkos::atomic_fetch_add(&ptcl_elm_indices(elm),mask);
      };
      parallel_for( fill_elm_indices, "fill_elm_indices" );

      kkLidView particle_indices("particle_indices", new_num_ptcls);
      // increment through free spaces in elements for NEW particle indices (atomic)
      Kokkos::parallel_for("fill_ptcl_indices", new_num_ptcls,
        KOKKOS_LAMBDA(const lid_t ptcl_id) {
          particle_indices(ptcl_id) = Kokkos::atomic_fetch_add(&ptcl_elm_indices(new_particle_elements(ptcl_id)),1);
        });

      auto offsets_cpy = offsets; // copy of offsets (necessary, Kokkos doesn't like member variables)
      // calculate SoA and ptcl in SoA indices for next CopyMTVsToAoSoA
      kkLidView soa_indices("soa_indices", new_num_ptcls);
      kkLidView soa_ptcl_indices("soa_ptcl_indices", new_num_ptcls);
      Kokkos::parallel_for("soa_and_ptcl", new_num_ptcls,
        KOKKOS_LAMBDA(const lid_t ptcl_id) {
          soa_indices(ptcl_id) = offsets_cpy(new_particle_elements(ptcl_id))
            + (particle_indices(ptcl_id)/soa_len);
          soa_ptcl_indices(ptcl_id) = particle_indices(ptcl_id)%soa_len;
        });
      CopyMTVsToAoSoA<device_type, DataTypes>(aosoa_, new_particles, soa_indices,
        soa_ptcl_indices); // copy data over
      num_ptcls = num_ptcls + new_particle_elements.size(); // update particle number
    }

    RecordTime("CabM add particles", add_timer.seconds());

    //destroy the old aosoa and use the new one in the CabanaM object
    aosoa_ = newAosoa;
    setActive(aosoa_, elmDegree_d, parentElms_, offsets);

    RecordTime("CSR rebuild", overall_timer.seconds());
    Kokkos::Profiling::popRegion();
  }

}
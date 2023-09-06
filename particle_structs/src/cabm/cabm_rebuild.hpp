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
   *    undefined behavior for new_element(ptcl) >= num_elms or new_particle_elements(ptcl) >= num_elems
  */
  template <class DataTypes, typename MemSpace>
  void CabM<DataTypes, MemSpace>::rebuild(kkLidView new_element,
                                         kkLidView new_particle_elements,
                                         MTVs new_particles) {
    const auto btime = prebarrier();

    Kokkos::Profiling::pushRegion("CabM Rebuild");
    Kokkos::Timer overall_timer; // timer for rebuild

    const auto num_new_ptcls = new_particle_elements.size();
    const auto soa_len = AoSoA_t::vector_length;
    kkLidView elmDegree_d("elmDegree", num_elems);
    const auto activeSliceIdx = aosoa_->number_of_members-1;
    auto active = Cabana::slice<activeSliceIdx>(*aosoa_);

    // first loop to count number of particles per new element (atomic)
    assert(new_element.size() == capacity_);
    kkLidView num_removed_d("num_removed_d", 1); // for counting particles to be removed
    auto atomic = KOKKOS_LAMBDA(const lid_t& soa, const lid_t& tuple) {
      if (active.access(soa,tuple)) {
        lid_t parent = new_element(soa*soa_len + tuple);
        if (parent > -1) // count particles to be kept
          Kokkos::atomic_increment<lid_t>(&elmDegree_d(parent));
        else // count particles to be deleted
          Kokkos::atomic_increment<lid_t>(&num_removed_d(0));
        }
    };
    Cabana::SimdPolicy<soa_len,execution_space> simd_policy(0, capacity_);
    Cabana::simd_parallel_for(simd_policy, atomic, "atomic");
    lid_t num_removed = getLastValue(num_removed_d);

    // count and index new particles (atomic)
    kkLidView particle_indices(Kokkos::ViewAllocateWithoutInitializing("particle_indices"), num_new_ptcls);
    Kokkos::parallel_for("fill_ptcl_indices", num_new_ptcls,
      KOKKOS_LAMBDA(const lid_t ptcl) {
        lid_t parent = new_particle_elements(ptcl);
        assert(parent > -1); // new particles should have a destination element
        particle_indices(ptcl) = Kokkos::atomic_fetch_add(&elmDegree_d(parent),1);
      });

    RecordTime(name + " count active particles", overall_timer.seconds());
    Kokkos::Timer setup_timer; // timer for aosoa setup

    // prepare a new aosoa to store the shuffled particles
    kkLidView newOffset_d = buildOffset(elmDegree_d, num_ptcls-num_removed+num_new_ptcls, -1, padding_start); // -1 signifies to fill to num_soa
    lid_t newNumSoa = getLastValue(newOffset_d);
    bool swap;
    AoSoA_t* newAosoa;
    lid_t newCapacity;
    if (newNumSoa > num_soa_) { // if need extra space, update
    swap = false;
    newOffset_d = buildOffset(elmDegree_d, num_ptcls-num_removed+num_new_ptcls, extra_padding, padding_start);
    newNumSoa = getLastValue(newOffset_d);
    newCapacity = newNumSoa*soa_len;
    if (use_swap) delete aosoa_swap;
    newAosoa = makeAoSoA(newCapacity, newNumSoa);
    } else { // if we don't need extra space
    swap = true;
    newCapacity = capacity_;
    if (use_swap) newAosoa = aosoa_swap;
    else newAosoa = makeAoSoA(newCapacity, newNumSoa);
    }

    RecordTime(name + " move/destroy setup", setup_timer.seconds());
    Kokkos::Timer existing_timer; // timer for moving/deleting particles

    kkLidView elmPtclCounter_d("elmPtclCounter_device", num_elems);
    AoSoA_t aosoa_copy = *aosoa_; // copy of member variable aosoa_ (necessary, Kokkos doesn't like member variables)
    AoSoA_t newAosoa_copy = *newAosoa;
    auto copyPtcls = KOKKOS_LAMBDA(const lid_t& soa, const lid_t& tuple) {
    const lid_t destParent = new_element(soa*soa_len + tuple);
    if (active.access(soa,tuple) && destParent != -1) {
    // Compute the destSoa based on the destParent and an array of
    //   counters for each destParent tracking which particle is the next
    //   free position. Use atomic fetch and incriment with the
    //   'elmPtclCounter_d' array.
    const lid_t occupiedTuples = Kokkos::atomic_fetch_add(&elmPtclCounter_d(destParent), 1);
    // use newOffset_d to figure out which soa is the first for destParent
    const lid_t firstSoa = newOffset_d(destParent);
    const lid_t destSoa = firstSoa + occupiedTuples/soa_len;
    const lid_t destTuple = occupiedTuples%soa_len;
    Cabana::Impl::tupleCopy(
    newAosoa_copy.access(destSoa), destTuple, // dest
    aosoa_copy.access(soa), tuple); // src
    }
      };
    Cabana::simd_parallel_for(simd_policy, copyPtcls, "copyPtcls");

    // swap the old aosoa and use the new one in the CabanaM object
    if (swap) {
    auto temp = aosoa_;
    aosoa_ = newAosoa;
    if (use_swap) aosoa_swap = temp;
    else delete temp;
    } else { // destroy old aosoas and make new ones
    delete aosoa_;
    aosoa_ = newAosoa;
    if (use_swap) aosoa_swap = makeAoSoA(newCapacity, newNumSoa);
    }
    // update member variables (note that these are set before particle addition)
    num_soa_ = newNumSoa;
    capacity_ = newCapacity;
    offsets = newOffset_d;
    num_ptcls = num_ptcls-num_removed+num_new_ptcls;
    parentElms_ = getParentElms(num_elems, num_soa_, offsets);
    setActive(elmDegree_d);

    RecordTime(name + " move/destroy existing particles", existing_timer.seconds());
    Kokkos::Timer add_timer; // timer for adding particles

    // add new particles
    if (num_new_ptcls > 0 && new_particles != NULL) {
    kkLidView offsets_cpy = offsets; // copy of offsets (necessary, Kokkos doesn't like member variables)
    // calculate SoA and ptcl in SoA indices for next CopyMTVsToAoSoA
    kkLidView soa_indices(Kokkos::ViewAllocateWithoutInitializing("soa_indices"), num_new_ptcls);
    kkLidView soa_ptcl_indices(Kokkos::ViewAllocateWithoutInitializing("soa_ptcl_indices"), num_new_ptcls);
    Kokkos::parallel_for("soa_and_ptcl", num_new_ptcls,
    KOKKOS_LAMBDA(const lid_t ptcl) {
    soa_indices(ptcl) = offsets_cpy(new_particle_elements(ptcl))
    + (particle_indices(ptcl)/soa_len);
    soa_ptcl_indices(ptcl) = particle_indices(ptcl)%soa_len;
    });
    CopyMTVsToAoSoA<CabM<DataTypes, MemSpace>, DataTypes>(*aosoa_, new_particles,
    soa_indices, soa_ptcl_indices); // copy data over
    }

    RecordTime(name + " add particles", add_timer.seconds());
    RecordTime(name + " rebuild", overall_timer.seconds(), btime);
    Kokkos::Profiling::popRegion();
  }

}

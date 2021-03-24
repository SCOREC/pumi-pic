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
    /// @todo add prebarrier to main ParticleStructure files
    //const auto btime = prebarrier();
    Kokkos::Timer barrier_timer;
    MPI_Barrier(MPI_COMM_WORLD);
    const auto btime = barrier_timer.seconds();

    Kokkos::Profiling::pushRegion("CabM Rebuild");
    Kokkos::Timer overall_timer; // timer for rebuild

    const auto num_new_ptcls = new_particle_elements.size();
    const auto soa_len = AoSoA_t::vector_length;
    kkLidView elmDegree_d("elmDegree", num_elems);
    const auto activeSliceIdx = aosoa_.number_of_members-1;
    auto active = Cabana::slice<activeSliceIdx>(aosoa_);

    // first loop to count number of particles per new element (atomic)
    assert(new_element.size() == capacity_);
    kkLidView num_removed_d("num_removed_d", 1); // for counting particles to be removed
    auto atomic = KOKKOS_LAMBDA(const lid_t& soa, const lid_t& tuple) {
      if (active.access(soa,tuple) == 1) {
        lid_t parent = new_element((soa*soa_len)+tuple);
        if (parent >= 0) { // count particles to be deleted
          Kokkos::atomic_increment<lid_t>(&elmDegree_d(parent));
        } else {
          Kokkos::atomic_increment<lid_t>(&num_removed_d(0));
        }
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

    RecordTime("CabM count active particles", overall_timer.seconds());
    Kokkos::Timer existing_timer; // timer for moving/deleting particles

    // prepare a new aosoa to store the shuffled particles
    kkLidView newOffset_d = buildOffset(elmDegree_d);
    const lid_t newNumSoa = getLastValue(newOffset_d);
    const lid_t newCapacity = newNumSoa*soa_len;
    auto newAosoa = makeAoSoA(newCapacity, newNumSoa);
    kkLidView elmPtclCounter_d("elmPtclCounter_device", num_elems);
    auto aosoa_copy = aosoa_; // copy of member variable aosoa_ (necessary, Kokkos doesn't like member variables)
    auto copyPtcls = KOKKOS_LAMBDA(const lid_t& soa, const lid_t& tuple) {
        if (active.access(soa,tuple) == 1) {
          // Compute the destSoa based on the destParent and an array of
          //   counters for each destParent tracking which particle is the next
          //   free position. Use atomic fetch and incriment with the
          //   'elmPtclCounter_d' array.
          lid_t destParent = new_element(soa*soa_len + tuple);
          if ( destParent >= 0 ) { // delete particles with negative destination element
            lid_t occupiedTuples = Kokkos::atomic_fetch_add<lid_t>(&elmPtclCounter_d(destParent), 1);
            // use newOffset_d to figure out which soa is the first for destParent
            const lid_t firstSoa = newOffset_d(destParent);
            const lid_t destIdx = occupiedTuples%soa_len;
            Cabana::Impl::tupleCopy(
              newAosoa.access(firstSoa + occupiedTuples/soa_len), destIdx, // dest
              aosoa_copy.access(soa), tuple); // src
          }
        }
      };
    Cabana::simd_parallel_for(simd_policy, copyPtcls, "copyPtcls");

    // destroy the old aosoa and use the new one in the CabanaM object
    aosoa_ = newAosoa;
    // update member variables (note that these are set before particle addition)
    num_soa_ = newNumSoa;
    capacity_ = newCapacity;
    offsets = newOffset_d;
    num_ptcls = num_ptcls - num_removed;
    parentElms_ = getParentElms(num_elems, num_soa_, offsets);
    setActive(aosoa_, elmDegree_d, parentElms_, offsets);

    RecordTime("CabM move/destroy existing particles", existing_timer.seconds());
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
      CopyMTVsToAoSoA<device_type, DataTypes>(aosoa_, new_particles, soa_indices,
        soa_ptcl_indices); // copy data over
      num_ptcls = num_ptcls + num_new_ptcls; // update particle number
    }

    RecordTime("CabM add particles", add_timer.seconds());
    RecordTime("CabM rebuild", overall_timer.seconds(), btime);
    Kokkos::Profiling::popRegion();
  }

}
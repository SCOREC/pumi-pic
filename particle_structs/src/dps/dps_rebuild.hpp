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
  void DPS<DataTypes, MemSpace>::rebuild(kkLidView new_element,
                                         kkLidView new_particle_elements,
                                         MTVs new_particles) {
    const auto btime = prebarrier();

    Kokkos::Profiling::pushRegion("DPS Rebuild");
    Kokkos::Timer overall_timer; // timer for rebuild

    const auto num_new_ptcls = new_particle_elements.size();
    const auto soa_len = AoSoA_t::vector_length;
    kkLidView elmDegree_d("elmDegree", num_elems);
    auto active = Cabana::slice<DPS_DT::size-1>(*aosoa_);
    auto parentElms_cpy = parentElms_;

    // first loop to count removed particles and move/remove them (move)
    assert(new_element.size() == capacity_);
    kkLidView num_removed_d("num_removed_d", 1); // for counting particles to be removed
    auto move = KOKKOS_LAMBDA(const lid_t& soa, const lid_t& tuple) {
      if (active.access(soa,tuple)) {
        lid_t parent = new_element(soa*soa_len + tuple);
        if (parent > -1) // count particles kept and move
          parentElms_cpy(soa*soa_len + tuple) = parent;
        else { // count particles deleted and delete
          Kokkos::atomic_inc<lid_t>(&num_removed_d(0));
          active.access(soa,tuple) = false; // delete particles
        }
      }
    };
    Cabana::SimdPolicy<soa_len,execution_space> simd_policy(0, capacity_);
    Cabana::simd_parallel_for(simd_policy, move, "move");
    lid_t num_removed = getLastValue(num_removed_d);
    parentElms_ = parentElms_cpy;
    
    RecordTime("DPS count/move/delete active particles", overall_timer.seconds());
    Kokkos::Timer copy_timer; // timer for copying to new structure

    // if capacity reached, allocate new structure and copy into
    if ( num_ptcls-num_removed+num_new_ptcls > capacity_ ) {
      lid_t new_num_soa = Kokkos::ceil(Kokkos::ceil(double(num_ptcls-num_removed+num_new_ptcls)/soa_len)*(1+extra_padding));
      lid_t new_capacity = new_num_soa*soa_len;
      AoSoA_t* newAosoa = makeAoSoA(new_capacity, new_num_soa);

      auto newActive = Cabana::slice<DPS_DT::size-1>(*newAosoa);
      auto setInactive = KOKKOS_LAMBDA(const lid_t& soa, const lid_t& tuple) { 
        newActive.access(soa,tuple) = false; // fill new active mask with false
      };
      Cabana::SimdPolicy<soa_len,execution_space> new_simd_policy(0, new_capacity);
      Cabana::simd_parallel_for(new_simd_policy, setInactive, "setInactive");
      
      AoSoA_t aosoa_copy = *aosoa_;
      AoSoA_t newAosoa_copy = *newAosoa;
      kkLidView new_parentElms(Kokkos::ViewAllocateWithoutInitializing("parentElms"), new_capacity);
      kkLidView parentElms_copy = parentElms_;
      kkLidView copy_index("copy_index", 1);
      auto copyPtcls = KOKKOS_LAMBDA(const lid_t& soa, const lid_t& tuple) {
        if (active.access(soa,tuple)) { // copy all active particles in
          // map particles in this structure to new structure
          lid_t index = Kokkos::atomic_fetch_add(&copy_index(0),1);
          lid_t soa_index = index/soa_len;
          lid_t tuple_index = index%soa_len;
          // copy particle aosoa_(soa,tuple) into newAosoa(index)
          new_parentElms(index) = parentElms_copy(soa*soa_len + tuple);
          Cabana::Impl::tupleCopy(
            newAosoa_copy.access(soa_index), tuple_index, // dest
            aosoa_copy.access(soa), tuple); // src
        }
      };
      Cabana::simd_parallel_for(simd_policy, copyPtcls, "copyPtcls");
      delete aosoa_;
      aosoa_ = newAosoa; // assign new aosoa
      num_soa_ = new_num_soa;
      capacity_ = new_capacity;
      parentElms_ = new_parentElms;
      active = Cabana::slice<DPS_DT::size-1>(*aosoa_);
    }

    num_ptcls = num_ptcls-num_removed+num_new_ptcls;

    RecordTime("DPS copy particles", copy_timer.seconds());
    Kokkos::Timer add_timer; // timer for adding particles

    if (num_new_ptcls > 0 && new_particles != NULL) {
      // calculate new particle indices by filling holes
      kkLidView parentElms_copy = parentElms_;
      kkLidView new_index("new_index", 1);
      kkLidView soa_indices(Kokkos::ViewAllocateWithoutInitializing("soa_indices"), num_new_ptcls);
      kkLidView soa_ptcl_indices(Kokkos::ViewAllocateWithoutInitializing("soa_ptcl_indices"), num_new_ptcls);
      auto addPtcls = KOKKOS_LAMBDA(const lid_t& soa, const lid_t& tuple) {
        if (!active.access(soa,tuple)) { // if inactive
          lid_t index = Kokkos::atomic_fetch_add(&new_index(0),1); // attempt to fill hole
          if (index < num_new_ptcls) { // if new particles not yet assigned
            active.access(soa,tuple) = true;
            soa_indices(index) = soa;
            soa_ptcl_indices(index) = tuple;
            parentElms_copy(soa*soa_len + tuple) = new_particle_elements(index);
          }
        }
      };
      Cabana::SimdPolicy<soa_len,execution_space> new_simd_policy(0,capacity_);
      Cabana::simd_parallel_for(new_simd_policy, addPtcls, "addPtcls");
      parentElms_ = parentElms_copy;
      // populate with new particle data
      CopyMTVsToAoSoA<DPS<DataTypes, MemSpace>, DataTypes>(*aosoa_, new_particles,
        soa_indices, soa_ptcl_indices); // copy data over
    }

    RecordTime("DPS add particles", add_timer.seconds());
    RecordTime("DPS rebuild", overall_timer.seconds(), btime);
    Kokkos::Profiling::popRegion();
  }

}
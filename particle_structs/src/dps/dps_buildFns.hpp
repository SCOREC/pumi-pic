#pragma once

namespace pumipic {

  /**
   * helper function: initialize an AoSoA (including hidden active mask)
   * @param[in] capacity maximum capacity (number of particles) of the AoSoA to be created
   * @param[in] num_soa total number of SoAs
   * @return AoSoA of max capacity, capacity, and total number of SoAs, numSoa
   * @exception num_soa != aosoa.numSoA()
  */
  template<class DataTypes, typename MemSpace>
  DPS<DataTypes, MemSpace>::AoSoA_t*
  DPS<DataTypes, MemSpace>::makeAoSoA(const lid_t capacity, const lid_t num_soa) {
    AoSoA_t* aosoa = new AoSoA_t;
    *aosoa = AoSoA_t();
    aosoa->resize(capacity);
    assert(num_soa == aosoa->numSoA());
    return aosoa;
  }

  /**
   * helper function: Builds the offset array for the CSR structure and sets sidealong tracking arrays
   * @param[in] particles_per_element View representing the number of active particles in each element
   * @param[in] capacity int representing total capacity
   * @param[out] particleIds View representing indexes of particles, grouped by element
   * @param[out] parentElms View representing parent element for each particle
   * @return offset view (each element is the first index of each element block in particleIds)
  */
  template<class DataTypes, typename MemSpace>
  typename ParticleStructure<DataTypes, MemSpace>::kkLidView
  DPS<DataTypes, MemSpace>::buildIndices(const kkLidView particles_per_element, const lid_t capacity,
  kkLidView &particleIds, kkLidView &parentElms) {
    kkLidHostMirror particles_per_element_h = deviceToHost(particles_per_element);
    Kokkos::View<lid_t*,host_space> offsets_h(Kokkos::ViewAllocateWithoutInitializing("offsets_host"), particles_per_element.size()+1);
    Kokkos::View<lid_t*,host_space> parentElms_h(Kokkos::ViewAllocateWithoutInitializing("parentElms_host"), capacity);
    // fill offsets
    offsets_h(0) = 0;
    for (lid_t i = 0; i < particles_per_element.size(); i++) {
      offsets_h(i+1) = particles_per_element_h(i) + offsets_h(i);
      for (lid_t j = offsets_h(i); j < offsets_h(i+1); j++)
        parentElms_h(j) = i; // set particle elements
    }
    lid_t num_particles = offsets_h(offsets_h.size()-1);
    for (lid_t j = num_particles; j < capacity; j++)
      parentElms_h(j) = particles_per_element.size()-1; // set all inactive particles to last element
    offsets_h(offsets_h.size()-1) = capacity; // add inactive particles to last element
    // move offsets and parentElms to device
    kkLidView offsets_d(Kokkos::ViewAllocateWithoutInitializing("offsets_device"), offsets_h.size());
    hostToDevice(offsets_d, offsets_h.data());
    parentElms = kkLidView(Kokkos::ViewAllocateWithoutInitializing("parentElms_device"), parentElms_h.size());
    hostToDevice(parentElms, parentElms_h.data());

    // add base particle ids
    particleIds = kkLidView(Kokkos::ViewAllocateWithoutInitializing("particleIds"), num_particles);
    Kokkos::parallel_for(num_particles, KOKKOS_LAMBDA(const lid_t& i) {
      particleIds(i) = i;
    });

    return offsets_d;
  }

  /**
   * helper function: initializes last type in AoSoA as active mask
   *   where 1 denotes an active particle and 0 denotes an inactive particle.
   *   Fills with 1s up to num_particles
   * @param[in] num_particles number of particles to fill up to
  */
  template<class DataTypes, typename MemSpace>
  void DPS<DataTypes, MemSpace>::setNewActive(const lid_t num_particles) {
    const auto soa_len = AoSoA_t::vector_length;
    const auto activeSliceIdx = aosoa_->number_of_members-1;
    auto active = Cabana::slice<activeSliceIdx>(*aosoa_);

    Cabana::SimdPolicy<soa_len,execution_space> simd_policy(0, capacity_);
    Cabana::simd_parallel_for(simd_policy,
      KOKKOS_LAMBDA( const lid_t soa, const lid_t ptcl ) {
        bool isActive = false;
        if (soa*soa_len+ptcl < num_particles)
          isActive = true;
        active.access(soa,ptcl) = isActive;
      }, "set_active");
  }

  /**
   * helper function: copies element_gids and creates a map for converting in the opposite direction
   * @param[in] element_gids view of global ids for each element
   * @param[out] lid_to_gid view to copy elmGid to
   * @param[out] gid_to_lid unordered map with elements global ids as keys and local ids as values
  */
  template<class DataTypes, typename MemSpace>
  void DPS<DataTypes, MemSpace>::createGlobalMapping(kkGidView element_gids, kkGidView& lid_to_gid, GID_Mapping& gid_to_lid) {
    lid_to_gid = kkGidView(Kokkos::ViewAllocateWithoutInitializing("row_to_element_gid"), num_elems);
    Kokkos::parallel_for(num_elems, KOKKOS_LAMBDA(const lid_t& i) {
      const gid_t gid = element_gids(i);
      lid_to_gid(i) = gid; // deep copy
      gid_to_lid.insert(gid, i);
    });
  }

  /**
   * helper function: find indices for particle_info for fillAoSoA
   * @param[in] particle_elements - particle_elements[i] contains the id (index)
   *                          of the parent element * of particle i
   * @param[in] particle_info - 'member type views' containing the user's data to be
   *                      associated with each particle
   * @exception particle_elements.size() != num_ptcls
  */
  template<class DataTypes, typename MemSpace>
  void DPS<DataTypes, MemSpace>::fillAoSoA(kkLidView particle_elements, MTVs particle_info) {
    assert(particle_elements.size() == num_ptcls);
    const auto soa_len = AoSoA_t::vector_length;

    kkLidView ptcl_elm_indices("ptcl_elm_indices", num_elems);
    kkLidView soa_indices("soa_indices", particle_elements.size());
    kkLidView soa_ptcl_indices("soa_ptcl_indices", particle_elements.size());
    kkLidView offsets_cpy = offsets;
    Kokkos::parallel_for(particle_elements.size(), KOKKOS_LAMBDA(const lid_t& ptcl_id) {
      lid_t index_in_element = Kokkos::atomic_fetch_add(&ptcl_elm_indices(particle_elements(ptcl_id)),1);
      lid_t index_in_aosoa = offsets_cpy(particle_elements(ptcl_id)) + index_in_element;
      soa_indices(ptcl_id) = index_in_aosoa/soa_len;
      soa_ptcl_indices(ptcl_id) = index_in_aosoa%soa_len;
    });

    CopyMTVsToAoSoA<DPS<DataTypes, MemSpace>, DataTypes>(*aosoa_, particle_info,
      soa_indices, soa_ptcl_indices);
  }

}
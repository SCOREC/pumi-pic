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
  typename DPS<DataTypes, MemSpace>::AoSoA_t*
  DPS<DataTypes, MemSpace>::makeAoSoA(const lid_t capacity, const lid_t num_soa) {
    AoSoA_t* aosoa = new AoSoA_t;
    *aosoa = AoSoA_t();
    aosoa->resize(capacity);
    assert(num_soa == aosoa->numSoA());
    return aosoa;
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
    const auto activeSliceIdx = DataTypes::size-1;
    auto active = Cabana::slice<activeSliceIdx>(*aosoa_);

    Cabana::SimdPolicy<soa_len,execution_space> simd_policy(0, capacity_);
    Cabana::simd_parallel_for(simd_policy,
      KOKKOS_LAMBDA(const lid_t& soa, const lid_t& ptcl) {
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
  void DPS<DataTypes, MemSpace>::createGlobalMapping(const kkGidView element_gids, kkGidView& lid_to_gid, GID_Mapping& gid_to_lid) {
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
   * @param[out] parentElms - member View of elements for particle i in AoSoA
   * @exception particle_elements.size() != num_ptcls
  */
  template<class DataTypes, typename MemSpace>
  void DPS<DataTypes, MemSpace>::fillAoSoA(const kkLidView particle_elements, const MTVs particle_info, kkLidView& parentElms) {
    assert(particle_elements.size() == num_ptcls);
    const auto soa_len = AoSoA_t::vector_length;

    parentElms = kkLidView(Kokkos::ViewAllocateWithoutInitializing("parentElms"), capacity_);

    kkLidView ptcl_elm_indices("ptcl_elm_indices", num_elems);
    kkLidView soa_indices("soa_indices", particle_elements.size());
    kkLidView soa_ptcl_indices("soa_ptcl_indices", particle_elements.size());
    kkLidView index("index", 1);
    Kokkos::parallel_for(particle_elements.size(), KOKKOS_LAMBDA(const lid_t& ptcl_id) {
      // calculate indices of particles
      lid_t index_in_aosoa = Kokkos::atomic_fetch_add(&index(0),1);
      soa_indices(ptcl_id) = index_in_aosoa/soa_len;
      soa_ptcl_indices(ptcl_id) = index_in_aosoa%soa_len;
      // set element
      parentElms(index_in_aosoa) = particle_elements(ptcl_id);
    });

    CopyMTVsToAoSoA<DPS<DataTypes, MemSpace>, DataTypes>(*aosoa_, particle_info,
      soa_indices, soa_ptcl_indices);
  }

  /**
    * helper function: sets elements in event of no data provided
    * @param[in] element_gids view of global ids for each element
    * @param[out] lid_to_gid view to copy elmGid to
  */
  template<class DataTypes, typename MemSpace>
  void DPS<DataTypes, MemSpace>:: setParentElms(const kkLidView particles_per_element, kkLidView& parentElms) {
    kkLidHostMirror particles_per_element_h = deviceToHost(particles_per_element);
    Kokkos::View<lid_t*,host_space> parentElms_h(Kokkos::ViewAllocateWithoutInitializing("parentElms_host"), capacity_);
    int index = 0;
    for (int i = 0; i < particles_per_element_h.size(); i++)
      for (int j = 0; j < particles_per_element_h(i); j++)
        parentElms_h(index++) = i;
    parentElms = kkLidView(Kokkos::ViewAllocateWithoutInitializing("parentElms"), capacity_);
    hostToDevice(parentElms, parentElms_h.data());
  }

}
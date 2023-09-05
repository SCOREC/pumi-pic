#pragma once

namespace pumipic {

  /**
   * helper function: Builds the offset array for the CSR structure (element i owns SoA offsets[i] to offsets[i+1])
   * @param[in] particles_per_element View representing the number of active elements in each SoA
   * @param[in] num_ptcls number of particles
   * @param[in] padding double representing percentage of extra soas to add to last element
   *                    set to -1 to fill offsets array to current number of soa
   * @param[out] padding_start saved int representing start of padding
   * @return offset View (each element is the first index of each SoA block)
  */
  template<class DataTypes, typename MemSpace>
  typename ParticleStructure<DataTypes, MemSpace>::kkLidView
  CabM<DataTypes, MemSpace>::buildOffset(const kkLidView particles_per_element, const lid_t num_ptcls, const double padding, lid_t &padding_start) {
    kkLidHostMirror particles_per_element_h = deviceToHost(particles_per_element);
    Kokkos::View<lid_t*,host_space> offsets_h(Kokkos::ViewAllocateWithoutInitializing("offsets_host"), particles_per_element.size()+1);
    // setup offsets
    const auto soa_len = AoSoA_t::vector_length;
    offsets_h(0) = 0;
    for ( lid_t i=0; i<particles_per_element.size(); i++ ) {
      const lid_t SoA_count = ceil( double(particles_per_element_h(i))/soa_len );
      offsets_h(i+1) = SoA_count + offsets_h(i);
    }
    padding_start = offsets_h(offsets_h.size()-1);

    if (padding == -1) { // add any remaining soa to end
      lid_t remaining_soa = num_soa_ - offsets_h(offsets_h.size()-1);
      if (remaining_soa > 0)
        offsets_h(offsets_h.size()-1) += remaining_soa;
    }
    else // add extra padding
      offsets_h(offsets_h.size()-1) += ceil( (num_ptcls*padding)/soa_len );

    kkLidView offsets_d(Kokkos::ViewAllocateWithoutInitializing("offsets_device"), offsets_h.size());
    hostToDevice(offsets_d, offsets_h.data());
    return offsets_d;
  }

  /**
   * helper function: initialize an AoSoA (including hidden active SoA)
   * @param[in] capacity maximum capacity (number of particles) of the AoSoA to be created
   * @param[in] num_soa total number of SoAs (can be greater than elem_count if
   * any element of particles_per_element is vector_length)
   * @return AoSoA of max capacity, capacity, and total number of SoAs, numSoa
   * @exception num_soa != aosoa.numSoA()
  */
  template<class DataTypes, typename MemSpace>
  typename CabM<DataTypes, MemSpace>::AoSoA_t*
  CabM<DataTypes, MemSpace>::makeAoSoA(const lid_t capacity, const lid_t num_soa) {
    AoSoA_t* aosoa = new AoSoA_t;
    *aosoa = AoSoA_t();
    aosoa->resize(capacity);
    assert(num_soa == aosoa->numSoA());
    return aosoa;
  }

  /**
   * helper function: builds the parent view for tracking particle position
   * @param[in] num_elements total number of elements in AoSoA
   * @param[in] num_soa total number of SoAs (can be greater than elem_count if
   * any element of deg is _vector_length)
   * @param[in] offsets offset view for AoSoA, built by buildOffset
   * @return parentElms view, each element is an lid_t representing the parent element each SoA resides in
  */
  template<class DataTypes, typename MemSpace>
  typename ParticleStructure<DataTypes, MemSpace>::kkLidView
  CabM<DataTypes, MemSpace>::getParentElms(const lid_t num_elements, const lid_t num_soa, const kkLidView offsets) {
    Kokkos::View<lid_t*,host_space> elms_h(Kokkos::ViewAllocateWithoutInitializing("parentElms_host"), num_soa);
    kkLidHostMirror offsets_h = create_mirror_view_and_copy(host_space(), offsets);
    for ( lid_t elm=0; elm<num_elements; elm++ )
      for ( lid_t soa=offsets_h(elm); soa<offsets_h(elm+1); soa++ )
          elms_h(soa)=elm;
    kkLidView elms_d("elements_device", elms_h.size());
    hostToDevice(elms_d, elms_h.data());
    return elms_d;
  }

  /**
   * helper function: initializes last type in AoSoA as active mask
   *   where 1 denotes an active particle and 0 denotes an inactive particle.
   * @param[in] particles_per_element view representing the number of active elements in each SoA
  */
  template<class DataTypes, typename MemSpace>
  void CabM<DataTypes, MemSpace>::setActive(const kkLidView particles_per_element) {
    // get copies of member variables for device
    auto parentElms_cpy = parentElms_;
    auto offsets_cpy = offsets;
    auto padding_start_cpy = padding_start;

    const lid_t num_elements = particles_per_element.size();
    const auto soa_len = AoSoA_t::vector_length;
    auto active = Cabana::slice<CM_DT::size-1>(*aosoa_);
    Cabana::SimdPolicy<soa_len,execution_space> simd_policy(0, capacity_);
    Cabana::simd_parallel_for(simd_policy,
      KOKKOS_LAMBDA( const lid_t soa, const lid_t ptcl ) {
        const lid_t elm = parentElms_cpy(soa);
        lid_t num_soa = offsets_cpy(elm+1)-offsets_cpy(elm);
        lid_t last_soa = offsets_cpy(elm+1)-1;
        if (last_soa >= padding_start_cpy) {
          last_soa = padding_start_cpy-1;
          num_soa = padding_start_cpy-offsets_cpy(elm);
        }
        const lid_t elm_ppe = particles_per_element(elm);
        const lid_t last_soa_ppe = soa_len - ((num_soa * soa_len) - elm_ppe);
        bool isActive = false;
        if (soa < last_soa)
          isActive = true;
        else if (soa == last_soa && ptcl < last_soa_ppe)
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
  void CabM<DataTypes, MemSpace>::createGlobalMapping(const kkGidView element_gids, kkGidView& lid_to_gid, GID_Mapping& gid_to_lid) {
    lid_to_gid = kkGidView(Kokkos::ViewAllocateWithoutInitializing("row to element gid"), num_elems);
    Kokkos::parallel_for(num_elems, KOKKOS_LAMBDA(const lid_t& i) {
      const gid_t gid = element_gids(i);
      lid_to_gid(i) = gid; // deep copy
      gid_to_lid.insert(gid, i);
    });
  }

  /**
   * helper function: find indices for particle_info and fills aosoa_ with particle data
   * @param[in] particle_elements - particle_elements[i] contains the id (index)
   *                          of the parent element * of particle i
   * @param[in] particle_info - 'member type views' containing the user's data to be
   *                      associated with each particle
   * @exception particle_elements.size() != num_ptcls
  */
  template<class DataTypes, typename MemSpace>
  void CabM<DataTypes, MemSpace>::fillAoSoA(const kkLidView particle_elements, const MTVs particle_info) {
    assert(particle_elements.size() == num_ptcls);

    const auto soa_len = AoSoA_t::vector_length;
    kkLidView offsets_cpy = offsets; // copy of offsets for device
    kkLidView ptcl_elm_indices("ptcl_elm_indices", num_elems);
    kkLidView soa_indices(Kokkos::ViewAllocateWithoutInitializing("soa_indices"), particle_elements.size());
    kkLidView soa_ptcl_indices(Kokkos::ViewAllocateWithoutInitializing("soa_ptcl_indices"), particle_elements.size());
    // fill each element left to right
    Kokkos::parallel_for("fill_ptcl_indices", num_ptcls,
      KOKKOS_LAMBDA(const lid_t ptcl_id) {
        lid_t index_in_element = Kokkos::atomic_fetch_add(&ptcl_elm_indices(particle_elements(ptcl_id)),1);
        soa_indices(ptcl_id) = offsets_cpy(particle_elements(ptcl_id)) + index_in_element/soa_len; // index of soa
        soa_ptcl_indices(ptcl_id) = index_in_element%soa_len; // index of particle in soa
      });

    CopyMTVsToAoSoA<CabM<DataTypes, MemSpace>, DataTypes>(*aosoa_, particle_info,
      soa_indices, soa_ptcl_indices);
  }

}
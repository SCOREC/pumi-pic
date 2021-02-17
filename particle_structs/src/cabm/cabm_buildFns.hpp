#pragma once

namespace pumipic {

  /**
   * helper function: Builds the offset array for the CSR structure
   * @param[in] particles_per_element View representing the number of active elements in each SoA
   * @return offset array (each element is the first index of each SoA block)
  */
  template<class DataTypes, typename MemSpace>
  typename ParticleStructure<DataTypes, MemSpace>::kkLidView
  CabM<DataTypes, MemSpace>::buildOffset(const kkLidView particles_per_element) {
    auto particles_per_element_h = Kokkos::create_mirror_view_and_copy(host_space(), particles_per_element);
    Kokkos::View<lid_t*,host_space> offsets_h(Kokkos::ViewAllocateWithoutInitializing("offsets_host"), particles_per_element.size()+1);
    // elem at i owns SoA offsets[i+1] - offsets[i]
    auto soa_len = AoSoA_t::vector_length;
    offsets_h(0) = 0;
    for ( lid_t i=0; i<particles_per_element.size(); i++ ) {
    const lid_t SoA_count = (particles_per_element_h(i)/soa_len) + 1;
    offsets_h(i+1) = SoA_count + offsets_h(i);
    }
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
  CabM<DataTypes, MemSpace>::AoSoA_t
  CabM<DataTypes, MemSpace>::makeAoSoA(const lid_t capacity, const lid_t num_soa) {
    auto aosoa = AoSoA_t();
    aosoa.resize(capacity);
    assert(num_soa == aosoa.numSoA());
    return aosoa;
  }

  /**
   * helper function: builds the parent view for tracking particle position
   * @param[in] num_elements total number of element SoAs in AoSoA
   * @param[in] num_soa total number of SoAs (can be greater than elem_count if
   * any element of deg is _vector_length)
   * @param[in] offsets offset array for AoSoA, built by buildOffset
   * @return parent array, each element is an lid_t representing the parent element each SoA resides in
  */
  template<class DataTypes, typename MemSpace>
  typename ParticleStructure<DataTypes, MemSpace>::kkLidView
  CabM<DataTypes, MemSpace>::getParentElms( const lid_t num_elements, const lid_t num_soa, const kkLidView offsets ) {
    Kokkos::View<lid_t*,host_space> elms_h(Kokkos::ViewAllocateWithoutInitializing("parentElms_host"), num_soa);
    kkLidHostMirror offsets_h = create_mirror_view_and_copy(host_space(), offsets);
    for ( lid_t elm=0; elm<num_elements; elm++ )
    for ( lid_t soa=offsets_h(elm); soa<offsets_h(elm+1); soa++)
        elms_h(soa)=elm;
    kkLidView elms_d("elements_device", elms_h.size());
    hostToDevice(elms_d, elms_h.data());
    return elms_d;
  }

  /**
   * helper function: initializes last SoAs in AoSoA as active mask
   * where 1 denotes an active particle and 0 denotes an inactive particle.
   * @param[out] aosoa the AoSoA to be edited
   * @param[in] particles_per_element pointer to an array of ints, representing the number of active elements in each SoA
   * @param[in] parentElms parent view for AoSoA, built by getParentElms
   * @param[in] offsets offset array for AoSoA, built by buildOffset
  */
  template<class DataTypes, typename MemSpace>
  void CabM<DataTypes, MemSpace>::setActive(AoSoA_t &aosoa, const kkLidView particles_per_element,
  const kkLidView parentElms, const kkLidView offsets) {

    const lid_t num_elements = particles_per_element.size();
    const auto soa_len = AoSoA_t::vector_length;

    const auto activeSliceIdx = aosoa.number_of_members-1;
    auto active = Cabana::slice<activeSliceIdx>(aosoa);
    Cabana::SimdPolicy<soa_len,execution_space> simd_policy(0, capacity_);
    Cabana::simd_parallel_for(simd_policy,
      KOKKOS_LAMBDA( const lid_t soa, const lid_t ptcl ) {
        const lid_t elm = parentElms(soa);
        const lid_t num_soa = offsets(elm+1)-offsets(elm);
        const lid_t last_soa = offsets(elm+1)-1;
        const lid_t elm_ppe = particles_per_element(elm);
        const lid_t last_soa_ppe = soa_len - ((num_soa * soa_len) - elm_ppe);
        lid_t isActive = 0;
        if (soa < last_soa) {
          isActive = 1;
        }
        if (soa == last_soa && ptcl < last_soa_ppe) {
          isActive = 1;
        }
        active.access(soa,ptcl) = isActive;
      }, "set_active");
  }

  /**
   * helper function: fills aosoa_ with particle data
   * @param[in] particle_indices - particle_elements[i] contains the index of particle i
   *                          in its parent element
   * @param[in] particle_elements - particle_elements[i] contains the id (index)
   *                          of the parent element * of particle i
   * @param[in] particle_info - 'member type views' containing the user's data to be
   *                      associated with each particle
  */
  template<class DataTypes, typename MemSpace>
  void CabM<DataTypes, MemSpace>::fillAoSoA(kkLidView particle_indices, kkLidView particle_elements, MTVs particle_info) {
    const auto soa_len = AoSoA_t::vector_length;

    // calculate SoA and ptcl in SoA indices for next CopyMTVsToAoSoA
    kkLidView soa_indices(Kokkos::ViewAllocateWithoutInitializing("soa_indices"), particle_elements.size());
    kkLidView soa_ptcl_indices(Kokkos::ViewAllocateWithoutInitializing("soa_ptcl_indices"), particle_elements.size());
    kkLidView offsets_copy = offsets; // copy of offsets since GPUs don't like member variables
    Kokkos::parallel_for("soa_and_ptcl", particle_elements.size(),
      KOKKOS_LAMBDA(const lid_t ptcl_id) {
        soa_indices(ptcl_id) = offsets_copy(particle_elements(ptcl_id))
          + (particle_indices(ptcl_id)/soa_len);
        soa_ptcl_indices(ptcl_id) = particle_indices(ptcl_id)%soa_len;
      });
    CopyMTVsToAoSoA<device_type, DataTypes>(aosoa_, particle_info, soa_indices,
      soa_ptcl_indices);
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
  void CabM<DataTypes, MemSpace>::initCabMData(kkLidView particle_elements, MTVs particle_info) {
    assert(particle_elements.size() == num_ptcls);

    // create a pointer to the offsets array that we can access in a kokkos parallel_for
    kkLidView offset_copy = offsets;
    kkLidView particle_indices(Kokkos::ViewAllocateWithoutInitializing("particle_indices"), num_ptcls);
    // View for tracking particle index in elements
    kkLidView ptcl_elm_indices("ptcl_elm_indices", num_elems);
    // atomic_fetch_add to increment from the beginning of each element
    Kokkos::parallel_for("fill_ptcl_indices", num_ptcls,
      KOKKOS_LAMBDA(const lid_t ptcl_id) {
        particle_indices(ptcl_id) = Kokkos::atomic_fetch_add(&ptcl_elm_indices(particle_elements(ptcl_id)),1);
      });
    fillAoSoA(particle_indices, particle_elements, particle_info);
  }

}
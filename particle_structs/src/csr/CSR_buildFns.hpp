#pragma once

namespace pumipic {

  /**
   * helper function: copies element_gids and creates a map for converting in the opposite direction
   * @param[in] element_gids view of global ids for each element
   * @param[out] lid_to_gid view to copy elmGid to
   * @param[out] gid_to_lid unordered map with elements global ids as keys and local ids as values
  */
  template<class DataTypes, typename MemSpace>
  void CSR<DataTypes, MemSpace>::createGlobalMapping(kkGidView element_gids, kkGidView& lid_to_gid, GID_Mapping& gid_to_lid) {
    lid_to_gid = kkGidView(Kokkos::ViewAllocateWithoutInitializing("row to element gid"), num_elems);
    Kokkos::parallel_for(num_elems, KOKKOS_LAMBDA(const lid_t& i) {
      const gid_t gid = element_gids(i);
      lid_to_gid(i) = gid;
      gid_to_lid.insert(gid, i);
    });
  }

  /**
   * helper funcion: fills the structure with particle data
   * @param[in] particle_elements particle_elements[i] contains the id (index)
   *    of the parent element * of particle i
   * @param[in] particle_info 'member type views' containing the user's data to be
   *    associated with each particle
  */
  template<class DataTypes, typename MemSpace>
  void CSR<DataTypes, MemSpace>::initCsrData(kkLidView particle_elements, MTVs particle_info) {
    // Create the 'particle_indices' array.  particle_indices[i] stores the
    // location in the 'ptcl_data' where  particle i is stored.  Use the
    // CSR offsets array and an atomic_fetch_add to compute these entries.
    lid_t given_particles = particle_elements.size();
    assert(given_particles == num_ptcls);

    // create a pointer to the offsets array that we can access in a kokkos parallel_for
    auto offset_cpy = offsets;
    kkLidView particle_indices(Kokkos::ViewAllocateWithoutInitializing("particle_indices"), num_ptcls);
    // SS3 insert code to set the entries of particle_indices>
    kkLidView row_indices(Kokkos::ViewAllocateWithoutInitializing("row_indices"), num_elems+1);
    Kokkos::deep_copy(row_indices, offset_cpy);

    // atomic_fetch_add to increment from the beginning of each element
    // when filling (offset[element] is start of element)
    auto fill_ptcl_indices = PS_LAMBDA(const lid_t& elm_id, const lid_t& ptcl_id, bool& mask){
      particle_indices(ptcl_id) = Kokkos::atomic_fetch_add(&row_indices(particle_elements(ptcl_id)),1);
    };
    parallel_for(fill_ptcl_indices);

    // populate ptcl_data with input data and particle_indices mapping
    CopyViewsToViews<kkLidView, DataTypes>(ptcl_data, particle_info, particle_indices);
  }

  template<class DataTypes, typename MemSpace>
  void CSR<DataTypes,MemSpace>::construct(kkLidView ptcls_per_elem, kkGidView element_gids,
                                          kkLidView particle_elements, MTVs particle_info){
    Kokkos::Profiling::pushRegion("csr_construction");

    int comm_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &comm_rank);

    if(!comm_rank)
      fprintf(stderr, "Building CSR\n");

    // SS1 allocate the offsets array and use an exclusive_scan (aka prefix sum)
    // to fill the entries of the offsets array.
    // see pumi-pic/support/SupportKK.h for the exclusive_scan helper function
    offsets = kkLidView(Kokkos::ViewAllocateWithoutInitializing("offsets"), num_elems+1);
    Kokkos::resize(ptcls_per_elem, ptcls_per_elem.size()+1);
    exclusive_scan(ptcls_per_elem, offsets);

    // get global ids
    if (element_gids.size() > 0) {
      createGlobalMapping(element_gids, element_to_gid, element_gid_to_lid);
    }

    // SS2 set the 'capacity_' of the CSR storage from the last entry of offsets
    // pumi-pic/support/SupportKK.h has a helper function for this
    capacity_ = getLastValue(offsets)*padding_amount;
    // allocate storage for user particle data
    CreateViews<device_type, DataTypes>(ptcl_data, capacity_);
    CreateViews<device_type, DataTypes>(ptcl_data_swap,capacity_);
    swap_capacity_ = capacity_;

    // If particle info is provided then enter the information
    lid_t given_particles = particle_elements.size();
    if (given_particles > 0 && particle_info != NULL) {
      if(!comm_rank) fprintf(stderr, "initializing CSR data\n");
      initCsrData(particle_elements, particle_info);
    }

    Kokkos::Profiling::popRegion();
  }
    
}

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
    lid_to_gid = kkGidView("row to element gid", num_elems);
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
    kkLidView particle_indices("particle_indices", num_ptcls);
    // SS3 insert code to set the entries of particle_indices>
    kkLidView row_indices("row indces", num_elems+1);
    Kokkos::deep_copy(row_indices, offset_cpy);

    // atomic_fetch_add to increment from the beginning of each element
    // when filling (offset[element] is start of element)
    auto fill_ptcl_indices = PS_LAMBDA(const lid_t elm_id, const lid_t ptcl_id, bool mask){
      particle_indices(ptcl_id) = Kokkos::atomic_fetch_add(&row_indices(particle_elements(ptcl_id)),1);
    };
    parallel_for(fill_ptcl_indices);

    // populate ptcl_data with input data and particle_indices mapping
    CopyViewsToViews<kkLidView, DataTypes>(ptcl_data, particle_info, particle_indices);
  }
    
}
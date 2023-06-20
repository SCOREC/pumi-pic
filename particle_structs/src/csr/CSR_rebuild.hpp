#pragma once

namespace pumipic {

  /**
   * Fully rebuild the structure by replacing the old one
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
  template<class DataTypes,typename MemSpace>
  void CSR<DataTypes,MemSpace>::rebuild(kkLidView new_element,
                                        kkLidView new_particle_elements,
                                        MTVs new_particles) {
    const auto btime = prebarrier();
    
    Kokkos::Profiling::pushRegion("CSR Rebuild");
    Kokkos::Timer timer;

    Kokkos::Timer time_ppe;
    // fresh filling of particles_per_element
    kkLidView particles_per_element = kkLidView("particlesPerElement", num_elems+1);
    kkLidView num_removed_d("num_removed_d",1);
    // Fill ptcls per elem for existing ptcls
    auto count_existing = PS_LAMBDA(const lid_t& elm_id, const lid_t& ptcl_id, const bool& mask) {
      if (new_element[ptcl_id] > -1)
        Kokkos::atomic_increment(&particles_per_element[new_element[ptcl_id]]);
      else
        Kokkos::atomic_increment(&num_removed_d(0));
    };
    parallel_for(count_existing,"fill particle Per Element existing");
    lid_t num_removed = getLastValue(num_removed_d); // save number of removed particles for later

    Kokkos::parallel_for("fill particlesPerElementNew", new_particle_elements.size(),
        KOKKOS_LAMBDA(const int& i) {
          assert(new_particle_elements[i] > -1);
          Kokkos::atomic_increment(&particles_per_element[new_particle_elements[i]]);
        });
    RecordTime("CSR calc ppe", time_ppe.seconds());

    // time offsets and indices calc
    Kokkos::Timer time_off_ind;
    // refill offset here
    auto offsets_new = kkLidView("offsets", num_elems+1); // CopyPSToPS uses orig offsets
    Kokkos::deep_copy(offsets_new, offsets);
    exclusive_scan(particles_per_element, offsets_new, execution_space());

    // Determine new_indices for all of the existing particles
    kkLidView row_indices(Kokkos::ViewAllocateWithoutInitializing("row indices"), num_elems+1);
    Kokkos::deep_copy(row_indices,offsets_new);
    kkLidView new_indices(Kokkos::ViewAllocateWithoutInitializing("new indices"), new_element.size());

    auto existing_ptcl_new_indices = PS_LAMBDA(const lid_t& elm_id, const lid_t& ptcl_id, const bool& mask) {
      const lid_t new_elem = new_element[ptcl_id];
      if (new_elem != -1)
        new_indices[ptcl_id] = Kokkos::atomic_fetch_add(&row_indices(new_elem),1);
      else
        new_indices[ptcl_id] = -1;
    };
    parallel_for(existing_ptcl_new_indices,"calc row indices");
    RecordTime("CSR offsets and indices", time_off_ind.seconds());

    lid_t num_new_ptcls = new_particle_elements.size();
    lid_t particles_on_process = num_ptcls - num_removed + num_new_ptcls;

    //Determine if realloc appropriate based on variables
    if (always_realloc || particles_on_process > swap_capacity_) {
      destroyViews<DataTypes>(ptcl_data_swap);
      CreateViews<device_type,DataTypes>(ptcl_data_swap, padding_amount*particles_on_process);
      swap_capacity_ = padding_amount*particles_on_process;
    }
    else if (particles_on_process < minimize_size*swap_capacity_){
      destroyViews<DataTypes>(ptcl_data_swap);
      CreateViews<device_type,DataTypes>(ptcl_data_swap, padding_amount*particles_on_process);
      swap_capacity_ = padding_amount*particles_on_process;
    }
    
    Kokkos::Timer time_pstops;
    // Copy existing particles to their new location in the temp MTV
    CopyPSToPS< CSR<DataTypes,MemSpace>, DataTypes >(this, ptcl_data_swap, ptcl_data, new_element, new_indices);
    RecordTime("CSR PSToPS", time_pstops.seconds());

    Kokkos::Timer time_newPtcls;
    // If there are new particles
    kkLidView new_particle_indices(Kokkos::ViewAllocateWithoutInitializing("new_particle_indices"), num_new_ptcls);
    // Determine new particle indices in the MTVs
    Kokkos::parallel_for("new_patricles_indices", num_new_ptcls,
                            KOKKOS_LAMBDA(const int& i) {
      lid_t new_elem = new_particle_elements(i);
      new_particle_indices(i) = Kokkos::atomic_fetch_add(&row_indices(new_elem),1);
    });
    if (num_new_ptcls > 0 && new_particles != NULL) {
      CopyViewsToViews<kkLidView,DataTypes>(ptcl_data_swap, new_particles,
                                                          new_particle_indices);
    }
    RecordTime("CSR ViewsToViews", time_newPtcls.seconds());

    // Reassign all member variables
    MTVs tmp_data = ptcl_data;
    ptcl_data = ptcl_data_swap;
    ptcl_data_swap = tmp_data;

    lid_t tmp_cap = capacity_;
    capacity_ = swap_capacity_;
    swap_capacity_ = tmp_cap;

    num_ptcls = particles_on_process;
    offsets   = offsets_new;

    RecordTime("CSR rebuild", timer.seconds(), btime);
    Kokkos::Profiling::popRegion();
  }

}

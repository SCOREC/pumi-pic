#pragma once

namespace {

  //helper function for rebuild to determine how much space to allocate
  template <typename ppView>
  int countParticlesOnProcess(ppView particle_elements){
    int count = 0;
    Kokkos::parallel_reduce("particle on process",
        particle_elements.size(), KOKKOS_LAMBDA (const int& i, int& lsum){
      if(particle_elements(i) > -1){
        lsum += 1;
      }
    }, count);
    return count;
  }
}

namespace pumipic{
  template<class DataTypes,typename MemSpace>
  void CSR<DataTypes,MemSpace>::rebuild(kkLidView new_element,
                                        kkLidView new_particle_elements,
                                        MTVs new_particles){
    Kokkos::Profiling::pushRegion("CSR Rebuild");
    Kokkos::Timer timer;

    //Counting of particles on process
    lid_t particles_on_process = countParticlesOnProcess(new_element) +
                                 countParticlesOnProcess(new_particle_elements);
    //Needs to be assigned here for later methods used
    num_ptcls = particles_on_process;

    RecordTime("CSR count active particles", timer.seconds());

    /*
    //Alocate new (temp) MTV
    MTVs particle_info;
    CreateViews<device_type,DataTypes>(particle_info, particles_on_process);
    */
    //ResizeViews(ptcl_data_swap,particles_on_process);

    if(particles_on_process > capacity_){
      destroyViews<DataTypes>(ptcl_data_swap);
      CreateViews<device_type,DataTypes>(ptcl_data_swap,particles_on_process);
    }

    //fresh filling of particles_per_element
    kkLidView particles_per_element = kkLidView("particlesPerElement", num_elems+1);

    Kokkos::fence();
    Kokkos::Timer time_ppe;
    //Fill ptcls per elem for exisitng ptcls
    auto count_existing = PS_LAMBDA(lid_t elm_id, lid_t ptcl_id, bool mask){
      if(new_element[ptcl_id] > -1)
        Kokkos::atomic_increment(&particles_per_element[new_element[ptcl_id]]);
    };
    parallel_for(count_existing,"fill particle Per Element existing");

    Kokkos::parallel_for("fill particlesPerElementNew", new_particle_elements.size(),
        KOKKOS_LAMBDA(const int& i){
          assert(new_particle_elements[i] > -1);
          Kokkos::atomic_increment(&particles_per_element[new_particle_elements[i]]);
        });
    Kokkos::fence();
    RecordTime("CSR calc ppe", time_ppe.seconds());

    //time offsets and indices calc
    Kokkos::Timer time_off_ind;
    //refill offset here
    auto offsets_new = kkLidView("offsets", num_elems+1); //CopyPSToPS uses orig offsets
    Kokkos::deep_copy(offsets_new, offsets);
    exclusive_scan(particles_per_element, offsets_new);

    //Determine new_indices for all of the exisitng particles
    kkLidView row_indices("row indices", num_elems+1);
    Kokkos::deep_copy(row_indices,offsets_new);
    kkLidView new_indices("new indices", new_element.size());

    auto existing_ptcl_new_indices = PS_LAMBDA(const lid_t elm_id, lid_t ptcl_id, bool mask){
      const lid_t new_elem = new_element[ptcl_id];
      if(new_elem != -1)
        new_indices[ptcl_id] = Kokkos::atomic_fetch_add(&row_indices(new_elem),1);
      else
        new_indices[ptcl_id] = -1;
    };
    parallel_for(existing_ptcl_new_indices,"calc row indices");
    Kokkos::fence();

    RecordTime("CSR offsets and indices", time_off_ind.seconds());

    Kokkos::Timer time_pstops;
    //Copy existing particles to their new location in the temp MTV
    CopyPSToPS< CSR<DataTypes,MemSpace> , DataTypes >(this, ptcl_data_swap, ptcl_data, new_element, new_indices);
    Kokkos::fence();
    RecordTime("CSR PSToPS", time_pstops.seconds());

    /*
    //Deallocate ptcl_data
    destroyViews<DataTypes>(ptcl_data);
    */

    Kokkos::Timer time_newPtcls;

    //If there are new particles
    lid_t num_new_ptcls = new_particle_elements.size();
    kkLidView new_particle_indices("new_particle_indices", num_new_ptcls);

    //Determine new particle indices in the MTVs
    Kokkos::parallel_for("new_patricles_indices", num_new_ptcls,
                            KOKKOS_LAMBDA(const int& i){
      lid_t new_elem = new_particle_elements(i);
      new_particle_indices(i) = Kokkos::atomic_fetch_add(&row_indices(new_elem),1);
    });

    if(num_new_ptcls > 0){
      CopyViewsToViews<kkLidView,DataTypes>(ptcl_data_swap, new_particles,
                                                          new_particle_indices);
    }
    Kokkos::fence();
    RecordTime("CSR ViewsToViews", time_newPtcls.seconds());

    //Resassign all member variables
    ptcl_data = ptcl_data_swap;
    capacity_ = particles_on_process;

    //capacity_ = getLastValue<lid_t>(offsets_new);
    num_ptcls = capacity_;
    offsets   = offsets_new;

    RecordTime("CSR rebuild", timer.seconds());
    Kokkos::Profiling::popRegion();
  }
}

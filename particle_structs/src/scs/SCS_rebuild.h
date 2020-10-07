#pragma once
namespace pumipic {
  template<class DataTypes, typename MemSpace>
    bool SellCSigma<DataTypes,MemSpace>::reshuffle(kkLidView new_element,
                                                   kkLidView new_particle_elements,
                                                   MTVs new_particles) {
    //Count current/new particles per row
    kkLidView new_particles_per_row("new_particles_per_row", numRows()+1);
    kkLidView num_holes_per_row("num_holes_per_row", numRows());
    kkLidView element_to_row_local = element_to_row;
    auto particle_mask_local = particle_mask;
    auto countNewParticles = PS_LAMBDA(lid_t element_id,lid_t particle_id, bool mask){
      const lid_t new_elem = new_element(particle_id);

      const lid_t row = element_to_row_local(element_id);
      const bool is_particle = mask & new_elem != -1;
      const bool is_moving = is_particle & new_elem != element_id;
      if (is_moving) {
        const lid_t new_row = element_to_row_local(new_elem);
        Kokkos::atomic_fetch_add(&(new_particles_per_row(new_row)), mask);
      }
      particle_mask_local(particle_id) = is_particle;
      Kokkos::atomic_fetch_add(&(num_holes_per_row(row)), !is_particle);
    };
    parallel_for(countNewParticles, "countNewParticles");
    // Add new particles to counts
    Kokkos::parallel_for("reshuffle_count", new_particle_elements.size(), KOKKOS_LAMBDA(const lid_t& i) {
        const lid_t new_elem = new_particle_elements(i);
        const lid_t new_row = element_to_row_local(new_elem);
        Kokkos::atomic_fetch_add(&(new_particles_per_row(new_row)), 1);
      });

    //Check if the particles will fit in current structure
    kkLidView fail("fail",1);
    Kokkos::parallel_for(numRows(), KOKKOS_LAMBDA(const lid_t& i) {
        if( new_particles_per_row(i) > num_holes_per_row(i))
          fail(0) = 1;
      });

    if (getLastValue<lid_t>(fail)) {
      //Reshuffle fails
      return false;
    }

    //Offset moving particles
    kkLidView offset_new_particles("offset_new_particles", numRows() + 1);
    kkLidView counting_offset_index("counting_offset_index", numRows() + 1);
    exclusive_scan(new_particles_per_row, offset_new_particles);
    Kokkos::deep_copy(counting_offset_index, offset_new_particles);

    int num_moving_ptcls = getLastValue<lid_t>(offset_new_particles);
    if (num_moving_ptcls == 0) {
      Kokkos::parallel_reduce(capacity(), KOKKOS_LAMBDA(const lid_t& i, lid_t& sum) {
          sum += particle_mask_local(i);
        }, num_ptcls);
      return true;
    }
    kkLidView movingPtclIndices("movingPtclIndices", num_moving_ptcls);
    kkLidView isFromSCS("isFromSCS", num_moving_ptcls);
    //Gather moving particle list
    auto gatherMovingPtcls = PS_LAMBDA(const lid_t& element_id,const lid_t& particle_id, const bool& mask){
      const lid_t new_elem = new_element(particle_id);

      const lid_t row = element_to_row_local(element_id);
      const bool is_moving = new_elem != -1 & new_elem != element_id & mask;
      if (is_moving) {
        const lid_t new_row = element_to_row_local(new_elem);
        const lid_t index = Kokkos::atomic_fetch_add(&(counting_offset_index(new_row)), 1);
        movingPtclIndices(index) = particle_id;
        isFromSCS(index) = 1;
      }
    };
    parallel_for(gatherMovingPtcls, "gatherMovingPtcls");

    //Gather new particles in list
    Kokkos::parallel_for("reshuffle_count", new_particle_elements.size(), KOKKOS_LAMBDA(const lid_t& i) {
        const lid_t new_elem = new_particle_elements(i);
        const lid_t new_row = element_to_row_local(new_elem);
        const lid_t index = Kokkos::atomic_fetch_add(&(counting_offset_index(new_row)), 1);
        movingPtclIndices(index) = i;
        isFromSCS(index) = 0;
      });

    //Assign hole index for moving particles
    kkLidView holes("holeIndex", num_moving_ptcls);
    auto assignPtclsToHoles = PS_LAMBDA(const lid_t& element_id,const lid_t& particle_id, const bool& mask){
      const lid_t row = element_to_row_local(element_id);
      if (!mask) {
        const lid_t moving_index = Kokkos::atomic_fetch_add(&(offset_new_particles(row)),1);
        const lid_t max_index = counting_offset_index(row);
        if (moving_index < max_index) {
          holes(moving_index) = particle_id;
        }
      }
    };
    parallel_for(assignPtclsToHoles, "assignPtclsToHoles");

    //Update particle mask
    Kokkos::parallel_for(num_moving_ptcls, KOKKOS_LAMBDA(const lid_t& i) {
        const lid_t old_index = movingPtclIndices(i);
        const lid_t new_index = holes(i);
        const lid_t fromSCS = isFromSCS(i);
        if (fromSCS == 1)
          particle_mask_local(old_index) = 0;
        particle_mask_local(new_index) = 1;
      });

    //Shift SCS values
    ShuffleParticles<SellCSigma<DataTypes, MemSpace>, DataTypes>(ptcl_data,
                                                                 new_particles,
                                                                 movingPtclIndices, holes,
                                                                 isFromSCS);

    //Count number of active particles
    Kokkos::parallel_reduce(capacity(), KOKKOS_LAMBDA(const lid_t& i, lid_t& sum) {
        sum += particle_mask_local(i);
      }, num_ptcls);
    return true;
  }

  template<class DataTypes, typename MemSpace>
    void SellCSigma<DataTypes,MemSpace>::rebuild(kkLidView new_element,
                                                 kkLidView new_particle_elements,
                                                 MTVs new_particles) {
    const auto btime = prebarrier();
    Kokkos::Profiling::pushRegion("scs_rebuild");
    Kokkos::Timer timer;
    int comm_rank, comm_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &comm_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &comm_size);

    //Count particles including new and leaving
    kkLidView new_particles_per_elem("new_particles_per_elem", numRows());
    auto countNewParticles = PS_LAMBDA(lid_t element_id,lid_t particle_id, bool mask){
      const lid_t new_elem = new_element(particle_id);
      if (new_elem != -1)
        Kokkos::atomic_fetch_add(&(new_particles_per_elem(new_elem)), mask);
    };
    parallel_for(countNewParticles, "countNewParticles");
    // Add new particles to counts
    Kokkos::parallel_for("rebuild_count", new_particle_elements.size(), KOKKOS_LAMBDA(const lid_t& i) {
        const lid_t new_elem = new_particle_elements(i);
        Kokkos::atomic_fetch_add(&(new_particles_per_elem(new_elem)), 1);
      });

    //Reduce the count of particles
    lid_t activePtcls;
    Kokkos::parallel_reduce(numRows(), KOKKOS_LAMBDA(const lid_t& i, lid_t& sum) {
        sum+= new_particles_per_elem(i);
      }, activePtcls);

    //If there are no particles left, then destroy the structure
    if(activePtcls == 0) {
      num_ptcls = 0;
      auto local_mask = particle_mask;
      auto resetMask = PS_LAMBDA(lid_t e, lid_t p, bool mask) {
        local_mask(p) = false;
      };
      parallel_for(resetMask, "resetMask");

      RecordTime(name +" rebuild", timer.seconds(), btime);
      Kokkos::Profiling::popRegion();

      return;
    }

    //If tryShuffling is on and shuffling works then rebuild is complete
    if (tryShuffling && reshuffle(new_element, new_particle_elements, new_particles)) {
      RecordTime(name + " rebuild", timer.seconds(), btime);
      Kokkos::Profiling::popRegion();
      return;
    }

    lid_t new_num_ptcls = activePtcls;

    int new_C = chooseChunkHeight(C_max, new_particles_per_elem);
    int old_C = C_;
    C_ = new_C;
    //Perform sorting
    Kokkos::Profiling::pushRegion("Sorting");
    PairView ptcls;
    sigmaSort(ptcls,num_elems,new_particles_per_elem, sigma);
    Kokkos::Profiling::popRegion();

    // Number of chunks without vertical slicing
    kkLidView chunk_widths;
    lid_t new_nchunks;
    kkLidView new_row_to_element;
    kkLidView new_element_to_row;
    constructChunks(ptcls, new_nchunks, chunk_widths, new_row_to_element, new_element_to_row);

    lid_t new_num_slices;
    lid_t new_capacity;
    kkLidView new_offsets;
    kkLidView new_slice_to_chunk;
    //Create offsets into each chunk/vertical slice
    constructOffsets(new_nchunks, new_num_slices, chunk_widths, new_offsets, new_slice_to_chunk,
                     new_capacity);

    //Allocate the SCS
    lid_t new_cap = getLastValue<lid_t>(new_offsets);
    kkLidView new_particle_mask("new_particle_mask", new_cap);
    if (swap_size < new_cap) {
      destroyViews<DataTypes, memory_space>(scs_data_swap);
      CreateViews<device_type, DataTypes>(scs_data_swap, new_cap*1.1);
      swap_size = new_cap * 1.1;
    }


    /* //Fill the SCS */
    kkLidView interior_slice_of_chunk("interior_slice_of_chunk", new_num_slices);
    Kokkos::parallel_for("set_interior_slice_of_chunk", Kokkos::RangePolicy<>(1,new_num_slices),
                         KOKKOS_LAMBDA(const lid_t& i) {
                           const lid_t my_chunk = new_slice_to_chunk(i);
                           const lid_t prev_chunk = new_slice_to_chunk(i-1);
                           interior_slice_of_chunk(i) = my_chunk == prev_chunk;
                         });
    lid_t C_local = C_;
    kkLidView element_index("element_index", new_nchunks * C_local);
    Kokkos::parallel_for("set_element_index", new_num_slices, KOKKOS_LAMBDA(const lid_t& i) {
        const lid_t chunk = new_slice_to_chunk(i);
        for (lid_t e = 0; e < C_local; ++e) {
          Kokkos::atomic_fetch_add(&element_index(chunk*C_local + e),
                                   (new_offsets(i) + e) * !interior_slice_of_chunk(i));
        }
      });
    C_ = old_C;
    kkLidView new_indices("new_scs_index", capacity());
    auto copySCS = PS_LAMBDA(lid_t elm_id, lid_t ptcl_id, bool mask) {
      const lid_t new_elem = new_element(ptcl_id);
      //TODO remove conditional
      if (mask && new_elem != -1) {
        const lid_t new_row = new_element_to_row(new_elem);
        new_indices(ptcl_id) = Kokkos::atomic_fetch_add(&element_index(new_row), new_C);
        const lid_t new_index = new_indices(ptcl_id);
        new_particle_mask(new_index) = 1;
      }
    };
    parallel_for(copySCS);

    CopyPSToPS<SellCSigma<DataTypes, MemSpace>, DataTypes>(this, scs_data_swap, ptcl_data,
                                                           new_element, new_indices);
    //Add new particles
    lid_t num_new_ptcls = new_particle_elements.size();
    kkLidView new_particle_indices("new_particle_scs_indices", num_new_ptcls);

    Kokkos::parallel_for("set_new_particle", num_new_ptcls, KOKKOS_LAMBDA(const lid_t& i) {
        lid_t new_elem = new_particle_elements(i);
        lid_t new_row = new_element_to_row(new_elem);
        new_particle_indices(i) = Kokkos::atomic_fetch_add(&element_index(new_row), new_C);
        lid_t new_index = new_particle_indices(i);
        new_particle_mask(new_index) = 1;
      });

    if (new_particle_elements.size() > 0)
      CopyViewsToViews<kkLidView, DataTypes>(scs_data_swap, new_particles, new_particle_indices);

    //set scs to point to new values
    C_ = new_C;
    num_ptcls = new_num_ptcls;
    num_chunks = new_nchunks;
    num_slices = new_num_slices;
    capacity_ = new_capacity;
    num_rows = num_chunks * C_;
    row_to_element = new_row_to_element;
    element_to_row = new_element_to_row;
    offsets = new_offsets;
    slice_to_chunk = new_slice_to_chunk;
    particle_mask = new_particle_mask;
    MTVs tmp = ptcl_data;
    ptcl_data = scs_data_swap;
    scs_data_swap = tmp;
    std::size_t tmp_size = current_size;
    current_size = swap_size;
    swap_size = tmp_size;

    RecordTime(name +" rebuild", timer.seconds(), btime);
    Kokkos::Profiling::popRegion();
  }

}

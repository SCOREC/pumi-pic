#pragma once
namespace pumipic {
  template<class DataTypes, typename MemSpace>
  int SellCSigma<DataTypes, MemSpace>::chooseChunkHeight(int maxC,
                                                         kkLidView ptcls_per_elem) {
    lid_t num_elems_with_ptcls = 0;
    Kokkos::parallel_reduce("count_elems", ptcls_per_elem.size(),
                            KOKKOS_LAMBDA(const lid_t& i, lid_t& sum) {
      sum += ptcls_per_elem(i) > 0;
    }, num_elems_with_ptcls);
    if (num_elems_with_ptcls == 0)
      return 1;
    if (num_elems_with_ptcls < maxC)
      return num_elems_with_ptcls;
    return maxC;
  }

  template<class DataTypes, typename MemSpace>
    void SellCSigma<DataTypes, MemSpace>::constructChunks(kkLidView ptcls,
                                                          kkLidView index,
                                                          lid_t& nchunks,
                                                          kkLidView& chunk_widths,
                                                          kkLidView& row_element,
                                                          kkLidView& element_row) {
    nchunks = num_elems / C_ + (num_elems % C_ != 0);
    chunk_widths = kkLidView(Kokkos::ViewAllocateWithoutInitializing("chunk_widths"), 
			     nchunks);
    row_element = kkLidView(Kokkos::ViewAllocateWithoutInitializing("row_element"), 
			    nchunks * C_);
    element_row = kkLidView(Kokkos::ViewAllocateWithoutInitializing("element_row"), 
			    nchunks * C_);
    kkLidView empty("empty_elems", 1);
    Kokkos::parallel_for(num_elems, KOKKOS_LAMBDA(const lid_t& i) {
        const lid_t element = index(i);
        row_element(i) = element;
        element_row(element) = i;
        Kokkos::atomic_fetch_add(&empty[0], ptcls(i) == 0);
      });
    Kokkos::parallel_for(Kokkos::RangePolicy<>(num_elems, nchunks * C_),
                         KOKKOS_LAMBDA(const lid_t& i) {
                           row_element(i) = i;
                           element_row(i) = i;
                           Kokkos::atomic_fetch_add(&empty[0], 1);
                         });

    num_empty_elements = getLastValue(empty);
    const PolicyType policy(nchunks, C_);
    lid_t C_local = C_;
    lid_t num_elems_local = num_elems;
    Kokkos::parallel_for(policy, KOKKOS_LAMBDA(const typename PolicyType::member_type& thread) {
        const lid_t chunk_id = thread.league_rank();
        const lid_t row_num = chunk_id * C_local + thread.team_rank();
        lid_t width = 0;
        if (row_num < num_elems_local) {
          width = ptcls(row_num);
        }
        thread.team_reduce(Kokkos::Max<lid_t,MemSpace>(width));
        chunk_widths[chunk_id] = width;
      });

    //Apply padding
    if (shuffle_padding > 0) {
      lid_t cw_sum, cw_sum_count;
      double cw_sum_inv;
      Kokkos::parallel_reduce("sum_chunk_widths", nchunks,
                              KOKKOS_LAMBDA(const lid_t& i, lid_t& sum) {
        sum += chunk_widths[i];
      }, cw_sum);
      Kokkos::parallel_reduce("sum_chunk_widths", nchunks,
                              KOKKOS_LAMBDA(const lid_t& i, lid_t& sum) {
        sum += chunk_widths[i] > 0;
      }, cw_sum_count);
      Kokkos::parallel_reduce("sum_chunk_widths", nchunks,
                              KOKKOS_LAMBDA(const lid_t& i, double& sum) {
        if (chunk_widths[i] > 0)
          sum += 1.0/ chunk_widths[i];
      }, cw_sum_inv);
      if (cw_sum > 0) {
        const double cw_sum2 = cw_sum / cw_sum_inv * shuffle_padding;
        const lid_t avg_pad = cw_sum * shuffle_padding / cw_sum_count;
        const double local_padding = shuffle_padding;
        if (pad_strat == PAD_EVENLY)
          Kokkos::parallel_for(nchunks, KOKKOS_LAMBDA(const lid_t& i) {
              if (chunk_widths[i] > 0)
                chunk_widths[i] += avg_pad;
            });
        else if (pad_strat == PAD_PROPORTIONALLY)
          Kokkos::parallel_for(nchunks, KOKKOS_LAMBDA(const lid_t& i) {
              chunk_widths[i] += chunk_widths[i] * local_padding;
            });
        else if (pad_strat == PAD_INVERSELY)
          Kokkos::parallel_for(nchunks, KOKKOS_LAMBDA(const lid_t& i) {
              if (chunk_widths[i] != 0)
                chunk_widths[i] += cw_sum2 / chunk_widths[i];
            });
      }
    }
  }

  template<class DataTypes, typename MemSpace>
    void SellCSigma<DataTypes, MemSpace>::createGlobalMapping(kkGidView elmGid,kkGidView& elm2Gid,
                                                              GID_Mapping& elmGid2Lid) {
    elm2Gid = kkGidView("row to element gid", numRows());
    Kokkos::parallel_for(num_elems, KOKKOS_LAMBDA(const lid_t& i) {
      const gid_t gid = elmGid(i);
      elm2Gid(i) = gid;
      elmGid2Lid.insert(gid, i);
    });
    Kokkos::parallel_for(Kokkos::RangePolicy<>(num_elems, numRows()), KOKKOS_LAMBDA(const lid_t& i) {
      elm2Gid(i) = -1;
    });
  }

  template<class DataTypes, typename MemSpace>
    void SellCSigma<DataTypes, MemSpace>::constructOffsets(lid_t nChunks, lid_t& nSlices,
                                                           kkLidView chunk_widths,
                                                           kkLidView& offs,
                                                           kkLidView& s2c, lid_t& cap) {
    kkLidView slices_per_chunk(Kokkos::ViewAllocateWithoutInitializing("slices_per_chunk"), nChunks + 1);
    const lid_t V_local = V_;
    Kokkos::parallel_for(nChunks, KOKKOS_LAMBDA(const lid_t& i) {
        const lid_t width = chunk_widths(i);
        const lid_t val1 = width / V_local;
        const lid_t val2 = width % V_local;
        const bool val3 = val2 != 0;
        slices_per_chunk(i) = val1 + val3;
      });
    kkLidView offset_nslices("offset_nslices",nChunks+1);
    exclusive_scan(slices_per_chunk, offset_nslices, execution_space());

    nSlices = getLastValue(offset_nslices);
    offs = kkLidView("SCS offset", nSlices + 1);
    s2c = kkLidView(Kokkos::ViewAllocateWithoutInitializing("slice to chunk"), nSlices);
    kkLidView slice_size(Kokkos::ViewAllocateWithoutInitializing("slice_size"), 
			 nSlices + 1);
    const lid_t nat_size = V_*C_;
    const lid_t C_local = C_;
    Kokkos::parallel_for(nChunks, KOKKOS_LAMBDA(const lid_t& i) {
      const lid_t start = offset_nslices(i);
      const lid_t end = offset_nslices(i+1);
      for (lid_t j = start; j < end; ++j) {
        s2c(j) = i;
        const lid_t rem = chunk_widths(i) % V_local;
        const lid_t val = rem + (rem==0)*V_local;
        const bool is_last = (j == end-1);
        slice_size(j) = (!is_last) * nat_size;
        slice_size(j) += (is_last) * (val) * C_local;
      }
    });

    exclusive_scan(slice_size, offs, execution_space());
    cap = getLastValue(offs);
  }
  template<class DataTypes, typename MemSpace>
  void SellCSigma<DataTypes, MemSpace>::setupParticleMask(Kokkos::View<lid_t*> mask,
                                                          kkLidView ptcls,
                                                          kkLidView chunk_widths,
                                                          kkLidView& chunk_starts) {
    //Get start of each chunk
    auto offsets_cpy = offsets;
    auto slice_to_chunk_cpy = slice_to_chunk;
    chunk_starts = kkLidView("chunk_starts", num_chunks);
    lid_t cap_local = capacity_;
    lid_t first_s2c = getFirstValue(slice_to_chunk);
    Kokkos::parallel_for(Kokkos::RangePolicy<>(first_s2c+1,num_chunks), KOKKOS_LAMBDA(const lid_t& i) {
      chunk_starts[i] = cap_local;
    });
    Kokkos::parallel_for(num_slices-1, KOKKOS_LAMBDA(const lid_t& i) {
      const lid_t my_chunk = slice_to_chunk_cpy(i);
      const lid_t next_chunk = slice_to_chunk_cpy(i+1);
      if (my_chunk != next_chunk) {
        for (int j = my_chunk + 1; j <= next_chunk; ++j)
          chunk_starts(j) = offsets_cpy(i+1);
      }
    });
    //Fill the SCS
    const lid_t league_size = num_chunks;
    const lid_t team_size = C_;
    const lid_t ne = num_elems;
    const PolicyType policy(league_size, team_size);
    auto row_to_element_cpy = row_to_element;
    Kokkos::parallel_for(policy, KOKKOS_LAMBDA(const typename PolicyType::member_type& thread) {
      const lid_t chunk = thread.league_rank();
      const lid_t chunk_row = thread.team_rank();
      const lid_t rowLen = chunk_widths(chunk);
      const lid_t start = chunk_starts(chunk) + chunk_row;
      Kokkos::parallel_for(Kokkos::TeamThreadRange(thread, team_size), [=] (lid_t& j) {
          const lid_t row = chunk * team_size + chunk_row;
          const lid_t element_id = row_to_element_cpy(row);
          Kokkos::parallel_for(Kokkos::ThreadVectorRange(thread, rowLen), [=] (lid_t& p) {
              const lid_t particle_id = start+(p*team_size);
              if (element_id < ne)
                mask(particle_id) = p < ptcls(row);
	      else
		mask(particle_id) = 0;
            });
        });
    });
  }

  template<class DataTypes, typename MemSpace>
  void SellCSigma<DataTypes, MemSpace>::initSCSData(kkLidView chunk_starts,
                                                    kkLidView particle_elements,
                                                    MTVs particle_info) {
    lid_t given_particles = particle_elements.size();
    assert(given_particles == num_ptcls);
    kkLidView element_to_row_local = element_to_row;
    //Setup starting point for each row
    lid_t C_local = C_;
    kkLidView row_index(Kokkos::ViewAllocateWithoutInitializing("row_index"), numRows());
    Kokkos::parallel_for(numRows(), KOKKOS_LAMBDA(const int& i) {
      int chunk = i / C_local;
      int row_of_chunk = i % C_local;
      row_index(i) = chunk_starts(chunk) + row_of_chunk;
    });

    kkLidView particle_indices(Kokkos::ViewAllocateWithoutInitializing("new_particle_scs_indices"), given_particles);
    Kokkos::parallel_for(given_particles, KOKKOS_LAMBDA(const lid_t& i) {
      lid_t new_elem = particle_elements(i);
      lid_t new_row = element_to_row_local(new_elem);
      particle_indices(i) = Kokkos::atomic_fetch_add(&row_index(new_row), C_local);
    });

    CopyViewsToViews<kkLidView, DataTypes>(ptcl_data, particle_info, particle_indices);
  }
}

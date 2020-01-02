#pragma once
namespace particle_structs {
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
    void SellCSigma<DataTypes, MemSpace>::constructChunks(PairView ptcls,
                                                          lid_t& nchunks,
                                                          kkLidView& chunk_widths,
                                                          kkLidView& row_element,
                                                          kkLidView& element_row) {
    nchunks = num_elems / C_ + (num_elems % C_ != 0);
    chunk_widths = kkLidView("chunk_widths", nchunks);
    row_element = kkLidView("row_element", nchunks * C_);
    element_row = kkLidView("element_row", nchunks * C_);
    kkLidView empty("empty_elems", 1);
    Kokkos::parallel_for(num_elems, KOKKOS_LAMBDA(const lid_t& i) {
        const lid_t element = ptcls(i).second;
        row_element(i) = element;
        element_row(element) = i;
        Kokkos::atomic_fetch_add(&empty[0], ptcls(i).first == 0);
      });
    Kokkos::parallel_for(Kokkos::RangePolicy<>(num_elems, nchunks * C_),
                         KOKKOS_LAMBDA(const lid_t& i) {
                           row_element(i) = i;
                           element_row(i) = i;
                           Kokkos::atomic_fetch_add(&empty[0], 1);
                         });

    num_empty_elements = getLastValue<lid_t>(empty);
    const PolicyType policy(nchunks, C_);
    lid_t C_local = C_;
    lid_t num_elems_local = num_elems;
    Kokkos::parallel_for(policy, KOKKOS_LAMBDA(const typename PolicyType::member_type& thread) {
        const lid_t chunk_id = thread.league_rank();
        const lid_t row_num = chunk_id * C_local + thread.team_rank();
        lid_t width = 0;
        if (row_num < num_elems_local) {
          width = ptcls(row_num).first;
        }
        thread.team_reduce(Kokkos::Max<lid_t,MemSpace>(width));
        chunk_widths[chunk_id] = width;
      });
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
                                                           kkLidView chunk_widths, kkLidView& offs,
                                                           kkLidView& s2c, lid_t& cap) {
    kkLidView slices_per_chunk("slices_per_chunk", nChunks);
    const lid_t V_local = V_;
    Kokkos::parallel_for(nChunks, KOKKOS_LAMBDA(const lid_t& i) {
        const lid_t width = chunk_widths(i);
        const lid_t val1 = width / V_local;
        const lid_t val2 = width % V_local;
        const bool val3 = val2 != 0;
        slices_per_chunk(i) = val1 + val3;
      });
    kkLidView offset_nslices("offset_nslices",nChunks+1);
    Kokkos::parallel_scan(nChunks, KOKKOS_LAMBDA(const lid_t& i, lid_t& cur, const bool& final) {
        cur += slices_per_chunk(i);
        if (final)
          offset_nslices(i+1) += cur;
      });

    nSlices = getLastValue<lid_t>(offset_nslices);
    offs = kkLidView("SCS offset", nSlices + 1);
    s2c = kkLidView("slice to chunk", nSlices);
    kkLidView slice_size("slice_size", nSlices);
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
    Kokkos::parallel_scan(nSlices, KOKKOS_LAMBDA(const lid_t& i, lid_t& cur, const bool final) {
        cur += slice_size(i);
        if (final) {
          const lid_t index = i+1;
          offs(index) += cur;
        }
      });
    cap = getLastValue<lid_t>(offs);
  }
  template<class DataTypes, typename MemSpace>
    void SellCSigma<DataTypes, MemSpace>::setupParticleMask(kkLidView mask,
                                                            PairView ptcls,
                                                            kkLidView chunk_widths) {
    //Get start of each chunk
    auto offsets_cpy = offsets;
    auto slice_to_chunk_cpy = slice_to_chunk;
    kkLidView chunk_starts("chunk_starts", num_chunks);
    Kokkos::parallel_for(num_slices-1, KOKKOS_LAMBDA(const lid_t& i) {
        const lid_t my_chunk = slice_to_chunk_cpy(i);
        const lid_t next_chunk = slice_to_chunk_cpy(i+1);
        if (my_chunk != next_chunk) {
          chunk_starts(next_chunk) = offsets_cpy(i+1);
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
            Kokkos::parallel_for(Kokkos::ThreadVectorRange(thread, rowLen), [&] (lid_t& p) {
                const lid_t particle_id = start+(p*team_size);
                if (element_id < ne)
                  mask(particle_id) =  p < ptcls(row).first;
              });
          });
      });
    \
  }
  template<class DataTypes, typename MemSpace>
    void SellCSigma<DataTypes, MemSpace>::initSCSData(kkLidView chunk_widths,
                                                      kkLidView particle_elements,
                                                      MTVs particle_info) {
    lid_t given_particles = particle_elements.size();
    kkLidView element_to_row_local = element_to_row;
    //Setup starting point for each row
    lid_t C_local = C_;
    kkLidView row_index("row_index", numRows());
    Kokkos::parallel_scan(num_chunks, KOKKOS_LAMBDA(const lid_t& i, lid_t& sum, const bool& final) {
        if (final) {
          for (lid_t j = 0; j < C_local; ++j)
            row_index(i*C_local+j) = sum + j;
        }
        sum += chunk_widths(i) * C_local;
      });
    //Determine index for each particle
    kkLidView particle_indices("new_particle_scs_indices", given_particles);
    Kokkos::parallel_for(given_particles, KOKKOS_LAMBDA(const lid_t& i) {
        lid_t new_elem = particle_elements(i);
        lid_t new_row = element_to_row_local(new_elem);
        particle_indices(i) = Kokkos::atomic_fetch_add(&row_index(new_row), C_local);
      });

    CopyNewParticlesToPS<SellCSigma<DataTypes, MemSpace>, DataTypes>(this, ptcl_data,
                                                                     particle_info,
                                                                     given_particles,
                                                                     particle_indices);
  }
}

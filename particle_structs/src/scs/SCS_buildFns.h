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
    kkLidView slice_size("slice_size", nSlices + 1);
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

#ifdef PP_USE_CUDA
    thrust::exclusive_scan(thrust::device, slice_size.data(), slice_size.data() + nSlices + 1,
                           offs.data(), 0);
#else
    Kokkos::parallel_scan(nSlices + 1, KOKKOS_LAMBDA(const lid_t& i, lid_t& cur, const bool final) {
      if (final) {
        offs(i) = cur;
      }
      cur += slice_size(i);
    });
#endif
    cap = getLastValue<lid_t>(offs);
  }
  template<class DataTypes, typename MemSpace>
  void SellCSigma<DataTypes, MemSpace>::setupParticleMask(kkLidView mask,
                                                          PairView ptcls,
                                                          kkLidView chunk_widths,
                                                          kkLidView& chunk_starts) {
    //Get start of each chunk
    auto offsets_cpy = offsets;
    //Check that offset is growing
    Kokkos::parallel_for(Kokkos::RangePolicy<>(1, num_slices), KOKKOS_LAMBDA(const lid_t& i) {
      if (offsets(i-1) > offsets(i)) {
        printf("[ERROR] Offsets is not increasing from slice %d to %d (%d > %d)\n",  i-1, i, offsets(i-1), offsets(i));
      }
    });
    auto slice_to_chunk_cpy = slice_to_chunk;
    chunk_starts = kkLidView("chunk_starts", num_chunks);
    lid_t cap_local = capacity_;
    Kokkos::parallel_for(Kokkos::RangePolicy<>(1,num_chunks), KOKKOS_LAMBDA(const lid_t& i) {
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

    //Check that chunk_starts is growing
    Kokkos::parallel_for(Kokkos::RangePolicy<>(1, num_chunks), KOKKOS_LAMBDA(const lid_t& i) {
      if (chunk_starts(i-1) > chunk_starts(i)) {
        printf("[ERROR] Chunk starts is not increasing from chunk %d to %d (%d > %d)\n",  i-1, i, chunk_starts(i-1), chunk_starts(i));
      }
    });

    //Fill the SCS
    const lid_t league_size = num_chunks;
    const lid_t team_size = C_;
    const lid_t ne = num_elems;
    const PolicyType policy(league_size, team_size);
    auto row_to_element_cpy = row_to_element;
    lid_t cap = capacity_;
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
              if (particle_id >= cap) {
                printf("[ERROR] Particle is over capacity %d > %d in element %d on row %d in chunk %d\n", particle_id, cap, element_id, row, chunk);
              }
              if (element_id < ne)
                mask(particle_id) =  p < ptcls(row).first;
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
    kkLidView row_starts("row_starts", numRows());
    kkLidView row_index("row_index", numRows());
    kkLidView row_ends("row_ends", numRows());
    lid_t nr_local = numRows();
    lid_t cap_local = capacity_;
    lid_t num_chunk = num_chunks;
    Kokkos::parallel_for(nr_local, KOKKOS_LAMBDA(const int& i) {
      int chunk = i / C_local;
      int row_of_chunk = i % C_local;
      row_index(i) = chunk_starts(chunk) + row_of_chunk;
      row_starts(i) = row_index(i);
      if (i < nr_local - C_local - 1)
        row_ends(i) = chunk_starts(chunk + 1) + row_of_chunk;
      else {
        row_ends(i) = cap_local + row_of_chunk;
      }
      if (row_ends(i) < row_starts(i)) {
        printf("[ERROR] Row ends before row starts on row %d/%d on chunk %d/%d [%d < %d] (cap: %d)\n", i, nr_local, chunk, num_chunk, row_ends(i), row_starts(i), cap_local);
      }
    });

    kkLidView particle_indices("new_particle_scs_indices", given_particles);
    Kokkos::parallel_for(given_particles, KOKKOS_LAMBDA(const lid_t& i) {
      lid_t new_elem = particle_elements(i);
      lid_t new_row = element_to_row_local(new_elem);
      particle_indices(i) = Kokkos::atomic_fetch_add(&row_index(new_row), C_local);
      if (particle_indices(i) > row_ends(new_row))
        printf("[ERROR] Particle %d exceeds row length on row %d/%d which is element %d\n",
               i, new_row, nr_local, new_elem);

    });

    kkLidView checks("index_checks", capacity_);
    Kokkos::parallel_for(given_particles, KOKKOS_LAMBDA(const lid_t& i) {
      const int index = particle_indices(i);
      Kokkos::atomic_add(&(checks(index)), 1);
    });

    kkLidView check_fails("check fails", 3);
    auto checkChecks = PS_LAMBDA(const int& e, const int& p, const bool& mask) {
      int nc = checks(p);
      if (nc > 1 && mask)
        Kokkos::atomic_add(&(check_fails(0)), 1);
      if (nc > 0 && !mask)
        Kokkos::atomic_add(&(check_fails(1)), 1);
      if (nc == 0 && mask)
        Kokkos::atomic_add(&(check_fails(2)), 1);

    };
    parallel_for(checkChecks);
    kkLidHostMirror fail_host = deviceToHost(check_fails);

    if(fail_host(0) != 0)
      fprintf(stderr, "[ERROR] %d/%d particles are set multiple times\n",
              fail_host(0), num_ptcls);
    if(fail_host(1) != 0)
      fprintf(stderr, "[ERROR] %d/%d padded cells are set\n", fail_host(1), num_ptcls);
    if(fail_host(2) != 0)
      fprintf(stderr, "[ERROR] %d/%d particles are not set\n", fail_host(2), num_ptcls);

    CopyViewsToViews<kkLidView, DataTypes>(ptcl_data, particle_info, particle_indices);

    kkLidView fails("fails",1);
    auto ids = this->template get<2>();
    auto checkIDsSet = PS_LAMBDA(const int& e, const int& p, const bool& mask) {
      if (mask) {
        if (ids(p) == 0)
          Kokkos::atomic_add(&(fails(0)), 1);
      }
    };
    parallel_for(checkIDsSet);

    int f = getLastValue<lid_t>(fails) - 1;
    if (f > 0) {
      fprintf(stderr, "[ERROR] %d/%d points did not set the id\n", f, num_ptcls);
    }
  }
}

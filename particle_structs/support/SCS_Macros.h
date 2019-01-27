#ifndef __SCS_MACROS_H__
#define __SCS_MACROS_H__


#define PS_PARALLEL_FOR_ELEMENTS(SCS, thread, element_id, ALGORITHM) \
  typedef Kokkos::View<lid_t*, exe_space::device_type> kkLidView; \
  kkLidView offsets = SCS->offsets_d; \
  kkLidView chunk_size = SCS->chunksz_d; \
  kkLidView slice_to_chunk = SCS->slice_to_chunk_d; \
  kkLidView row_to_element = SCS->row_to_element_d; \
  const int league_size = SCS->num_slices; \
  const int team_size = SCS->C; \
  typedef Kokkos::TeamPolicy<> team_policy; \
  const team_policy policy(league_size, team_size); \
  Kokkos::parallel_for(policy, KOKKOS_LAMBDA(const team_policy::member_type& thread) { \
    const int slice = thread.league_rank(); \
    const int slice_row = thread.team_rank(); \
    const int rowLen = (offsets(slice+1)-offsets(slice))/chunk_size(0); \
    const int start = offsets(slice) + slice_row; \
    Kokkos::parallel_for(Kokkos::TeamThreadRange(thread, chunk_size(0)), [=] (int& j) { \
        const int row = slice_to_chunk(slice) * chunk_size(0) + slice_row; \
        const int element_id = row_to_element(row); \
        ALGORITHM \
    }); \
  });

#define PS_PARALLEL_FOR_PARTICLES(SCS, thread, particle_id, ALGORITHM) \
  Kokkos::parallel_for(Kokkos::ThreadVectorRange(thread, rowLen), [&] (int& p) { \
    const int particle_id = start+(p*chunk_size(0)); \
    ALGORITHM \
  });
#endif

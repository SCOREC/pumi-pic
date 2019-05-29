#ifndef __SCS_MACROS_H__
#define __SCS_MACROS_H__

#ifdef SCS_USE_CUDA
#define SCS_DEVICE __device__ inline
#define SCS_LAMBDA [=] __device__
#else
#define SCS_DEVICE inline
#define SCS_LAMBDA [=]
#endif


#define PS_PARALLEL_FOR_ELEMENTS(SCS, thread, element_id, ALGORITHM) \
  typedef Kokkos::DefaultExecutionSpace exe_space; \
  typedef Kokkos::View<lid_t*, exe_space::device_type> kkLidView; \
  kkLidView offsets = SCS->offsets; \
  kkLidView slice_to_chunk = SCS->slice_to_chunk; \
  kkLidView row_to_element = SCS->row_to_element; \
  kkLidView particle_mask = SCS->particle_mask; \
  const int league_size = SCS->num_slices; \
  const int team_size = SCS->C; \
  typedef Kokkos::TeamPolicy<> team_policy; \
  const team_policy policy(league_size, team_size); \
  Kokkos::parallel_for(policy, KOKKOS_LAMBDA(const team_policy::member_type& thread) { \
    const int slice = thread.league_rank(); \
    const int slice_row = thread.team_rank(); \
    const int rowLen = (offsets(slice+1)-offsets(slice))/team_size; \
    const int start = offsets(slice) + slice_row; \
    Kokkos::parallel_for(Kokkos::TeamThreadRange(thread, team_size), [=] (int& j) { \
        const int row = slice_to_chunk(slice) * team_size + slice_row; \
        const int element_id = row_to_element(row); \
        ALGORITHM \
    }); \
  });

#define PS_PARALLEL_FOR_PARTICLES(SCS, thread, particle_id, ALGORITHM) \
  Kokkos::parallel_for(Kokkos::ThreadVectorRange(thread, rowLen), [&] (int& p) { \
    const int particle_id = start+(p*team_size); \
    ALGORITHM \
  });
#endif

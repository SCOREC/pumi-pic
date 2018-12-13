#ifndef __SCS_MACROS_H__
#define __SCS_MACROS_H__


#define PARALLEL_FOR_ELEMENTS(SCS, offsets, chunk_size, slice_to_chunk, row_to_element, thread, element_id) \
  const int league_size = SCS->num_slices; \
  const int team_size = SCS->C; \
  typedef Kokkos::TeamPolicy<> team_policy; \
  const team_policy policy(league_size, team_size); \
Kokkos::parallel_for(policy, KOKKOS_LAMBDA(const typename team_policy::member_type& thread) { \
    const int slice = thread.league_rank(); \
    const int slice_row = thread.team_rank(); \
    const int rowLen = (offsets(slice+1)-offsets(slice))/chunk_size(0); \
    const int start = offsets(slice) + slice_row; \
    parallel_for(TeamThreadRange(thread, chunk_size(0)), [=] (int& j) { \
	const int row = slice_to_chunk(slice) * chunk_size(0) + slice_row; \
	const int element_id = row_to_element(row);

#define PARALLEL_FOR_PARTICLES(SCS, thread, chunk_size, particle_id) \
	Kokkos::parallel_for(ThreadVectorRange(thread, rowLen), [&] (int& p) { \
	  const int pid = start+(p*chunksz_d(0));

#define END_PARALLEL_FOR_PARTICLES });

#define END_PARALLEL_FOR_ELEMENTS }); });



#endif

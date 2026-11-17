set(TEST_SOURCES
  ../particle_structs/test/Distribute.cpp
)

function(make_test exename srcname)
  add_executable(${exename} ${srcname} ${TEST_SOURCES})
  target_link_libraries(${exename} pumipic Omega_h::omega_h)
endfunction(make_test)

make_test(ps_combo160 ps_combo160.cpp)
make_test(ps_combo264 ps_combo264.cpp)
add_executable(search_benchmark search.cpp)
target_link_libraries(search_benchmark pumipic Omega_h::omega_h)

# The former BCC counterexample must succeed with output recording both on and off.
mpi_test(search_nonconvex_segment 1 ./search_benchmark 2 1 1 segment nonconvex)
mpi_test(search_nonconvex_no_output 1 ./search_benchmark 2 1 1 no-output nonconvex)
set_tests_properties(search_nonconvex_segment search_nonconvex_no_output
  PROPERTIES LABELS "search" TIMEOUT 120)

bob_end_subdir()

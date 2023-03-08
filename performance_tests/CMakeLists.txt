set(TEST_SOURCES
  ../particle_structs/test/Distribute.cpp
)

function(make_test exename srcname)
  add_executable(${exename} ${srcname} ${TEST_SOURCES})
  target_link_libraries(${exename} pumipic Omega_h::omega_h)
endfunction(make_test)

make_test(ps_combo ps_combo.cpp)

bob_end_subdir()

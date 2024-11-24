set(TEST_SOURCES
  Distribute.cpp
)
function(make_test exename srcname)
  add_executable(${exename} ${srcname} ${TEST_SOURCES})
  target_link_libraries(${exename} particleStructs)
endfunction(make_test)

make_test(smoke_test_particle smoke_test_particle.cpp)

make_test(typeTest typeTest.cpp)

make_test(sortTest sortTest.cpp)

make_test(scanTest scanTest.cpp)

make_test(viewTest viewTest.cpp)

make_test(buildSCSTest buildSCSTest.cpp)

make_test(initParticles initParticles.cpp)

make_test(lambdaTest lambdaTest.cpp)

make_test(test_scs_padding scs_padding.cpp)

make_test(write_particles write_particle_file.cpp)
make_test(test_structure test_structure.cpp)

make_test(destroy_test destroy_test.cpp)

include(testing.cmake)

bob_end_subdir()

set(TEST_SOURCES
  Distribute.cpp
)
function(make_test exename srcname)
  add_executable(${exename} ${srcname} ${TEST_SOURCES})
  target_link_libraries(${exename} particleStructs)
endfunction(make_test)

make_test(typeTest typeTest.cpp)

make_test(sortTest sortTest.cpp)

make_test(viewTest viewTest.cpp)

make_test(buildSCSTest buildSCSTest.cpp)

make_test(initParticles initParticles.cpp)

make_test(rebuild_scs rebuild_scs.cpp)

make_test(lambdaTest lambdaTest.cpp)

make_test(migrate_scs migrate_scs.cpp)

make_test(test_scs_padding scs_padding.cpp)

make_test(write_particles write_particle_file.cpp)
make_test(test_structure test_structure.cpp)

make_test(test_csr test_csr.cpp)
make_test(rebuild_csr rebuild_csr.cpp)
make_test(migrate_csr migrate_csr.cpp)

make_test(test_cabm test_cabm.cpp)
make_test(rebuild_cabm rebuild_cabm.cpp)
make_test(migrate_cabm migrate_cabm.cpp)

make_test(destroy_scs destroy_scs.cpp)
make_test(destroy_csr destroy_csr.cpp)
make_test(destroy_cabm destroy_cabm.cpp)

include(testing.cmake)

bob_end_subdir()

function(make_test exename srcname)
  add_executable(${exename} ${srcname})
  target_link_libraries(${exename} particleStructs)
endfunction(make_test)

make_test(typeTest typeTest.cpp)

make_test(buildSCSTest buildSCSTest.cpp)

make_test(initParticles initParticles.cpp)

make_test(rebuild rebuild.cpp)

make_test(lambdaTest lambdaTest.cpp)

make_test(migrateTest migrateTest.cpp)

include(testing.cmake)

bob_end_subdir()

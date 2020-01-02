add_test(NAME type_test COMMAND ./typeTest)

add_test(NAME buildSCS COMMAND ./buildSCSTest)

add_test(NAME initParticles COMMAND ./initParticles)

add_test(NAME rebuild COMMAND ./rebuild)

add_test(NAME lambdaTest COMMAND ./lambdaTest)

add_test(NAME migrateNothing COMMAND ./migrateTest)

add_test(NAME migrate4 COMMAND mpirun -np 4 ./migrateTest)

add_test(NAME write_ptcl_small COMMAND ./write_particles 5 25 0 0 small_ptcls_e5_p25_r0)
add_test(NAME write_ptcl_small_4 COMMAND mpirun -np 4 ./write_particles 5 25 0 2
  small_ptcls_e5_p25_r4)
add_test(NAME write_ptcl_4 COMMAND mpirun -np 4 ./write_particles 100 10000 0 2
  small_ptcls_e100_p10k_r4)

add_test(NAME test_structures_small COMMAND ./test_structure small_ptcls_e5_p25_r0)
add_test(NAME test_structures_small_4 COMMAND mpirun -np 4
  ./test_structure small_ptcls_e5_p25_r4)
add_test(NAME test_structures_4 COMMAND mpirun -np 4
  ./test_structure small_ptcls_e100_p10k_r4)

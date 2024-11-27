mpi_test(type_test 1 ./typeTest)

mpi_test(sort_test 1 ./sortTest 5000)

mpi_test(scanTest 1 ./scanTest)

mpi_test(view_test 1 ./viewTest)

mpi_test(initParticles 1 ./initParticles)

mpi_test(buildSCS 1 ./buildSCSTest)

mpi_test(scs_padding 1 ./test_scs_padding)

mpi_test(lambdaTest 1 ./lambdaTest)

mpi_test(write_ptcl_small 1 ./write_particles 5 25 0 0 small_ptcls_e5_p25_r0)
mpi_test(write_ptcl_small_4 4 ./write_particles 5 25 0 2 small_ptcls_e5_p25_r4)
mpi_test(write_ptcl_4 4 ./write_particles 100 10000 0 2 small_ptcls_e100_p10k_r4)
mpi_test(write_ptcl_empty 4 ./write_particles 0 0 0 0 empty_ptcls)
mpi_test(write_ptcl_noptcls 4 ./write_particles 100 0 0 0 no_ptcls_e100)
mpi_test(write_ptcl_medium 1 ./write_particles 500 100000 0 2 medium_ptcls_e500_p10e5_r0)
mpi_test(write_ptcl_large 1 ./write_particles 2500 1000000 0 2 large_ptcls_e2500_p10e6_r0)

mpi_test(test_structures_small 1 ./test_structure small_ptcls_e5_p25_r0)
mpi_test(test_structures_medium 1 ./test_structure medium_ptcls_e500_p10e5_r0)
mpi_test(test_structures_large 1 ./test_structure large_ptcls_e2500_p10e6_r0)
mpi_test(test_structures_small_4 4 ./test_structure small_ptcls_e5_p25_r4)
mpi_test(test_structures_4 4 ./test_structure small_ptcls_e100_p10k_r4)
mpi_test(test_structures_empty 4 ./test_structure empty_ptcls)
mpi_test(test_structures_noptcls 4 ./test_structure no_ptcls_e100)

mpi_test(destroy_test 4 ./destroy_test)

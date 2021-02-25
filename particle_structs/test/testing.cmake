add_test(NAME type_test COMMAND ./typeTest)

add_test(NAME view_test COMMAND ./viewTest)

add_test(NAME initParticles COMMAND ./initParticles)

add_test(NAME buildSCS COMMAND ./buildSCSTest)


add_test(NAME scs_padding COMMAND ./test_scs_padding)

add_test(NAME rebuild_scs COMMAND ./rebuild_scs)

add_test(NAME rebuild_csr_small COMMAND ./rebuild_csr 5 20 1)
add_test(NAME rebuild_csr_medium COMMAND ./rebuild_csr 50 1000 1)
add_test(NAME rebuild_csr_large_0 COMMAND ./rebuild_csr 2500 1000000 0)
add_test(NAME rebuild_csr_large_1 COMMAND ./rebuild_csr 2500 1000000 1)
add_test(NAME rebuild_csr_large_2 COMMAND ./rebuild_csr 2500 1000000 2)
add_test(NAME rebuild_csr_large_3 COMMAND ./rebuild_csr 2500 1000000 3)

add_test(NAME rebuild_cabm_small COMMAND ./rebuild_cabm 5 20 1)
add_test(NAME rebuild_cabm_medium COMMAND ./rebuild_cabm 50 1000 1)
add_test(NAME rebuild_cabm_large_0 COMMAND ./rebuild_cabm 2500 1000000 0)
add_test(NAME rebuild_cabm_large_1 COMMAND ./rebuild_cabm 2500 1000000 1)
add_test(NAME rebuild_cabm_large_2 COMMAND ./rebuild_cabm 2500 1000000 2)
add_test(NAME rebuild_cabm_large_3 COMMAND ./rebuild_cabm 2500 1000000 3)

add_test(NAME lambdaTest COMMAND ./lambdaTest)

add_test(NAME migrateNothing_scs COMMAND ./migrate_scs)
add_test(NAME migrate4_scs COMMAND mpirun -np 4 ./migrate_scs)
add_test(NAME migrateNothing_cabm COMMAND ./migrate_cabm)
add_test(NAME migrate4_cabm COMMAND mpirun -np 4 ./migrate_cabm)

add_test(NAME write_ptcl_small COMMAND ./write_particles 5 25 0 0 small_ptcls_e5_p25_r0)
add_test(NAME write_ptcl_small_4 COMMAND mpirun -np 4 ./write_particles 5 25 0 2
    small_ptcls_e5_p25_r4)
add_test(NAME write_ptcl_4 COMMAND mpirun -np 4 ./write_particles 100 10000 0 2
    small_ptcls_e100_p10k_r4)
add_test(NAME write_ptcl_empty COMMAND mpirun -np 4 ./write_particles 0 0 0 0 empty_ptcls)
add_test(NAME write_ptcl_noptcls COMMAND mpirun -np 4 ./write_particles 100 0 0 0 no_ptcls_e100)
add_test(NAME write_ptcl_medium COMMAND ./write_particles 500 100000 0 2 medium_ptcls_e500_p10e5_r0)
add_test(NAME write_ptcl_large COMMAND ./write_particles 2500 1000000 0 2 large_ptcls_e2500_p10e6_r0)

add_test(NAME test_csr_small COMMAND ./test_csr small_ptcls_e5_p25_r0)
add_test(NAME test_csr_small2 COMMAND ./test_csr small_ptcls_e5_p25_r4)

add_test(NAME test_cabm_small COMMAND ./test_cabm small_ptcls_e5_p25_r0)
add_test(NAME test_cabm_small2 COMMAND ./test_cabm small_ptcls_e5_p25_r4)

add_test(NAME test_structures_small COMMAND ./test_structure small_ptcls_e5_p25_r0)
add_test(NAME test_structures_medium COMMAND ./test_structure medium_ptcls_e500_p10e5_r0)
#add_test(NAME test_structures_large COMMAND ./test_structure large_ptcls_e2500_p10e6_r0)
add_test(NAME test_structures_small_4 COMMAND mpirun -np 4
  ./test_structure small_ptcls_e5_p25_r4)
add_test(NAME test_structures_4 COMMAND mpirun -np 4
  ./test_structure small_ptcls_e100_p10k_r4)
add_test(NAME test_structures_empty COMMAND mpirun -np 4
  ./test_structure empty_ptcls)
add_test(NAME test_structures_noptcls COMMAND mpirun -np 4
  ./test_structure no_ptcls_e100)

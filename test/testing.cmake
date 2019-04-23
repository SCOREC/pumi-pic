function(mpi_test TESTNAME PROCS EXE)
  add_test(
    NAME ${TESTNAME}
    COMMAND ${MPIRUN} ${MPIRUN_PROCFLAG} ${PROCS} ${VALGRIND} ${VALGRIND_ARGS} ${EXE} ${ARGN}
  )
endfunction(mpi_test)

mpi_test(adjSearch_1 1
  ./adj ${TEST_DATA_DIR}/cube.msh 2,0.5,0.2 4,0.9,0.3)

mpi_test(collision_search_1 1
  ./collision ${TEST_DATA_DIR}/cube.msh)

mpi_test(barycentric_1 1
  ./barycentric 0.0,1.0,0.0:0.5,0.0,0.0:1.0,1.0,0.0:0.5,1.0,0.5 )


mpi_test(barycentric_2 1
  ./barycentric 0.0,1.0,0.0:0.5,0.0,0.0:1.0,1.0,0.0:0.5,1.0,0.5  0.5,0.6,0  0,0.3,0.3,0.4)

mpi_test(barycentric_3 1  ./barycentric test1)

mpi_test(barycentric_4 1  ./barycentric test2)

mpi_test(linetri_intersection_1 1   ./linetri_intersection)

mpi_test(linetri_intersection_2 1
  ./linetri_intersection  0.0,1.0,0.0:0.5,0.0,0.0:1.0,1.0,0.0  0.5,0.6,-2  0.5,0.6,2 )

mpi_test(push_and_search_1 1
  ./push_and_search ${TEST_DATA_DIR}/cube.msh )

mpi_test(pseudoPushAndSearch 1
  ./pseudoPushAndSearch --kokkos-threads=1 ${TEST_DATA_DIR}/pisces/gitr.msh)

mpi_test(particleToMesh 1
  ./particleToMesh --kokkos-threads=1 ${TEST_DATA_DIR}/pisces/gitr.msh)

#MPI+X testing

mpi_test(print_partition_cube_2 2 ./print_partition ${TEST_DATA_DIR}/cube.msh testing_cube_ptn)

mpi_test(print_partition_pisces_4 4 ./print_partition ${TEST_DATA_DIR}/pisces.msh testing_pisces_ptn)

mpi_test(ptn_loading_cube 2 ./ptn_loading ${TEST_DATA_DIR}/cube.msh testing_cube 1 3)

mpi_test(ptn_loading_pisces 4 ./ptn_loading ${TEST_DATA_DIR}/pisces.msh testing_pisces 3 6)
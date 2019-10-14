function(mpi_test TESTNAME PROCS EXE)
  add_test(
    NAME ${TESTNAME}
    COMMAND ${MPIRUN} ${MPIRUN_PROCFLAG} ${PROCS} ${VALGRIND} ${VALGRIND_ARGS} ${EXE} ${ARGN}
  )
endfunction(mpi_test)

mpi_test(barycentric_3 1  ./barycentric test1)

mpi_test(barycentric_4 1  ./barycentric test2)

mpi_test(linetri_intersection_2 1
  ./linetri_intersection  0.0,1.0,0.0:0.5,0.0,0.0:1.0,1.0,0.0  0.5,0.6,-2  0.5,0.6,2 )

mpi_test(pseudoPushAndSearch_t1 1
  ./pseudoPushAndSearch --kokkos-threads=1 
  ${TEST_DATA_DIR}/pisces/gitr.msh ignored 200 5 -0.5 0.8 0)
mpi_test(pseudoPushAndSearch_t2 1
  ./pseudoPushAndSearch --kokkos-threads=2
  ${TEST_DATA_DIR}/pisces/gitr.msh ignored 200 5 -0.5 0.8 0)
mpi_test(pseudoPushAndSearch_t2_r2 2
  ./pseudoPushAndSearch --kokkos-threads=2 
  ${TEST_DATA_DIR}/pisces/gitr.msh 
  ${TEST_DATA_DIR}/pisces/pisces_2.ptn 200 5 -0.5 0.8 0)

mpi_test(pseudoPushAndSearch_cube_t1 1
  ./pseudoPushAndSearch --kokkos-threads=1 
  ${TEST_DATA_DIR}/cube/7k.osh ignored 200 156 0 0 1)
mpi_test(pseudoPushAndSearch_cube_t2 1
  ./pseudoPushAndSearch --kokkos-threads=2
  ${TEST_DATA_DIR}/cube/7k.osh ignored 200 156 0 0 1)

mpi_test(search2d 1 ./search2d
  ${TEST_DATA_DIR})

mpi_test(pseudoXGCm_scatter 1
  ./pseudoXGCm_scatter --kokkos-threads=1
  ${TEST_DATA_DIR}/plate/tri8_parDiag.osh)
mpi_test(pseudoXGCm_24kElms 1
  ./pseudoXGCm --kokkos-threads=1
  ${TEST_DATA_DIR}/xgc/itg24k.osh ignored 
  1000 5 100)
mpi_test(pseudoXGCm_24kElms_4 4
  ./pseudoXGCm --kokkos-threads=1
  ${TEST_DATA_DIR}/xgc/itg24k.osh ${TEST_DATA_DIR}/xgc/itg24k.cpn
  1000 2 100)

mpi_test(pseudoXGCm_126kElms 1
  ./pseudoXGCm --kokkos-threads=1
  ${TEST_DATA_DIR}/xgc/itg126k.osh ignored
  10000 141 10)

#MPI+X testing
mpi_test(print_partition_cube_2 2 ./print_partition ${TEST_DATA_DIR}/cube.msh testing_cube)
mpi_test(ptn_loading_cube 2 ./ptn_loading ${TEST_DATA_DIR}/cube.msh testing_cube_2.ptn 1 3)

mpi_test(print_partition_cube_4 4 ./print_partition ${TEST_DATA_DIR}/cube.msh testing_cube)
mpi_test(ptn_loading_cube_4 4 ./ptn_loading ${TEST_DATA_DIR}/cube.msh testing_cube_4.ptn 1 3)

mpi_test(print_partition_pisces_4 4
         ./print_partition ${TEST_DATA_DIR}/pisces/gitr.msh testing_pisces)
mpi_test(ptn_loading_pisces 4
         ./ptn_loading ${TEST_DATA_DIR}/pisces/gitr.msh testing_pisces_4.ptn 1 3)

mpi_test(full_mesh_pisces 4 
         ./full_mesh ${TEST_DATA_DIR}/pisces/gitr.msh testing_pisces_4.ptn)

mpi_test(input_construct_cube 4
         ./input_construct ${TEST_DATA_DIR}/cube.msh testing_cube_4.ptn)
         
mpi_test(comm_array_pisces 4 
         ./comm_array ${TEST_DATA_DIR}/pisces/gitr.msh testing_pisces_4.ptn)

  #mpi_test(borisMove 1
  # ./borisMove --kokkos-threads=1 ${TEST_DATA_DIR}/pisces/gitr.msh 
  #${TEST_DATA_DIR}/inputFields)


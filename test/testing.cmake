function(mpi_test TESTNAME PROCS EXE)
  add_test(
    NAME ${TESTNAME}
    COMMAND ${MPIRUN} ${MPIRUN_PROCFLAG} ${PROCS} ${VALGRIND} ${VALGRIND_ARGS} ${EXE} ${ARGN}
  )
endfunction(mpi_test)

#mpi_test(adjSearch_1 1
#  ./adj ${TEST_DATA_DIR}/cube.msh 2,0.5,0.2 4,0.9,0.3)

#mpi_test(collision_search_1 1
#  ./collision ${TEST_DATA_DIR}/cube.msh)

#mpi_test(barycentric_1 1
#  ./barycentric 0.0,1.0,0.0:0.5,0.0,0.0:1.0,1.0,0.0:0.5,1.0,0.5 )


#mpi_test(barycentric_2 1
#  ./barycentric 0.0,1.0,0.0:0.5,0.0,0.0:1.0,1.0,0.0:0.5,1.0,0.5  0.5,0.6,0  0,0.3,0.3,0.4)

#mpi_test(barycentric_3 1  ./barycentric test1)

#mpi_test(barycentric_4 1  ./barycentric test2)

#mpi_test(linetri_intersection_1 1   ./linetri_intersection)

#mpi_test(linetri_intersection_2 1
#  ./linetri_intersection  0.0,1.0,0.0:0.5,0.0,0.0:1.0,1.0,0.0  0.5,0.6,-2  0.5,0.6,2 )

#mpi_test(pseudoPushAndSearch_t1 1
#  ./pseudoPushAndSearch --kokkos-threads=1 
#  ${TEST_DATA_DIR}/pisces/gitr.msh ignored 200 5 -0.5 0.8 0)
#mpi_test(pseudoPushAndSearch_t2 1
#  ./pseudoPushAndSearch --kokkos-threads=2
#  ${TEST_DATA_DIR}/pisces/gitr.msh ignored 200 5 -0.5 0.8 0)
#mpi_test(pseudoPushAndSearch_t2_r2 2
#  ./pseudoPushAndSearch --kokkos-threads=2 
#  ${TEST_DATA_DIR}/pisces/gitr.msh 
#  ${TEST_DATA_DIR}/pisces/pisces_2.ptn 200 5 -0.5 0.8 0)

#mpi_test(pseudoPushAndSearch_cube_t1 1
#  ./pseudoPushAndSearch --kokkos-threads=1 
#  ${TEST_DATA_DIR}/cube/7k.osh ignored 200 156 0 0 1)
#mpi_test(pseudoPushAndSearch_cube_t2 1
#  ./pseudoPushAndSearch --kokkos-threads=2
#  ${TEST_DATA_DIR}/cube/7k.osh ignored 200 156 0 0 1)

#MPI+X testing
#mpi_test(print_partition_cube_2 2 ./print_partition ${TEST_DATA_DIR}/cube.msh testing_cube)
#mpi_test(ptn_loading_cube 2 ./ptn_loading ${TEST_DATA_DIR}/cube.msh testing_cube_2.ptn 1 3)

#mpi_test(print_partition_cube_4 4 ./print_partition ${TEST_DATA_DIR}/cube.msh testing_cube)
#mpi_test(ptn_loading_cube_4 4 ./ptn_loading ${TEST_DATA_DIR}/cube.msh testing_cube_4.ptn 1 3)

#mpi_test(print_partition_pisces_4 4
#         ./print_partition ${TEST_DATA_DIR}/pisces/gitr.msh testing_pisces)
#mpi_test(ptn_loading_pisces 4
#         ./ptn_loading ${TEST_DATA_DIR}/pisces/gitr.msh testing_pisces_4.ptn 1 3)

#mpi_test(full_mesh_pisces 4 
#         ./full_mesh ${TEST_DATA_DIR}/pisces/gitr.msh testing_pisces_4.ptn)

#mpi_test(comm_array_pisces 4 
#         ./comm_array ${TEST_DATA_DIR}/pisces/gitr.msh testing_pisces_4.ptn)

#mpi_test(borisMove 1
#   ./borisMove --kokkos-threads=1 ${TEST_DATA_DIR}/pisces/gitr.msh 
#   ${TEST_DATA_DIR}/inputFields)

#mpi_test(comm_array_pisces 4 
#         ./comm_array ${TEST_DATA_DIR}/pisces/gitr.msh testing_pisces_4.ptn)

# linext16 inputFields d3dTut_profiles_withE d3d2_profiles 
# d3d_reducedDomain_profiles_ni

#mpi_test(test_scs_fp 1
# ./test_scs_fp --kokkos-threads=1  ${TEST_DATA_DIR}/pisces/gitr.msh )

mpi_test(borisMove 1
  ./borisMove --kokkos-threads=1 ${TEST_DATA_DIR}/pisces/gitr.msh 
  ${TEST_DATA_DIR}/inputFields)

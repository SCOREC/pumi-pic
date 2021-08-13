#simple tests
mpi_test(barycentric_3 1 ./barycentric test1)

mpi_test(search2d 1 ./search2d
  ${TEST_DATA_DIR})

#mesh/partition tests

mpi_test(print_partition_cube_2 2
  ./print_partition ${TEST_DATA_DIR}/cube.msh testing_cube)
mpi_test(ptn_loading_cube 2
  ./ptn_loading ${TEST_DATA_DIR}/cube.msh testing_cube_2.ptn 1 3)

mpi_test(print_partition_cube_4 4
  ./print_partition ${TEST_DATA_DIR}/cube.msh testing_cube)
mpi_test(ptn_loading_cube_4 4
  ./ptn_loading ${TEST_DATA_DIR}/cube.msh testing_cube_4.ptn 1 3)

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

mpi_test(file_rw_cube_4 4
  ./file_rw
  ${TEST_DATA_DIR}/cube.msh
  testing_cube_4.ptn
  full full
  test_cube_file)
mpi_test(file_rw_xgc_24k_1 1
  ./file_rw
  ${TEST_DATA_DIR}/xgc/24k.osh
  ignored
  full full
  ${TEST_DATA_DIR}/xgc/24k)
mpi_test(file_rw_xgc_24k_4 4
  ./file_rw
  ${TEST_DATA_DIR}/xgc/24k.osh
  ${TEST_DATA_DIR}/xgc/24k_4.cpn
  full bfs
  ${TEST_DATA_DIR}/xgc/24k)
mpi_test(file_rw_xgc_120k_1 1
  ./file_rw
  ${TEST_DATA_DIR}/xgc/120k.osh
  ignored
  bfs bfs
  ${TEST_DATA_DIR}/xgc/120k)
mpi_test(file_rw_xgc_120k_4 4
  ./file_rw
  ${TEST_DATA_DIR}/xgc/120k.osh
  ${TEST_DATA_DIR}/xgc/120k_4.cpn
  bfs bfs
  ${TEST_DATA_DIR}/xgc/120k)

#load balancing tests
mpi_test(lb_r1 1 ./test_lb
         ${TEST_DATA_DIR}/cube.msh
         ignored)

mpi_test(lb_r4 4 ./test_lb
         ${TEST_DATA_DIR}/cube.msh
         testing_cube_4.ptn)

#reverse classification tests
mpi_test(revClass_r1 1 ./test_revClass
         ${TEST_DATA_DIR}/cube/7k.osh
         ignored)

#pseudo-simulation tests

mpi_test(pseudoPushAndSearch_t1 1
  ./pseudoPushAndSearch
  ${TEST_DATA_DIR}/pisces/gitr.msh ignored 200 5 -0.5 0.8 0)
mpi_test(pseudoPushAndSearch_t2_r2 2
  ./pseudoPushAndSearch
  ${TEST_DATA_DIR}/pisces/gitr.msh
  ${TEST_DATA_DIR}/pisces/pisces_2.ptn 200 5 -0.5 0.8 0)

mpi_test(pseudoPushAndSearch_cube_t1 1
  ./pseudoPushAndSearch
  ${TEST_DATA_DIR}/cube/7k.osh ignored 200 156 0 0 1)

mpi_test(pseudoXGCm_scatter 1
  ./pseudoXGCm_scatter
  ${TEST_DATA_DIR}/plate/tri8_parDiag.osh)

mpi_test(pseudoXGCm_24kElms 1
  ./pseudoXGCm
  ${TEST_DATA_DIR}/xgc/24k
  1000 5 100 0.5 0)
mpi_test(pseudoXGCm_24kElms_4 4
  ./pseudoXGCm
  ${TEST_DATA_DIR}/xgc/24k
  1000 2 100 0.5 0)

mpi_test(pseudoXGCm_120kElms 1
  ./pseudoXGCm
  ${TEST_DATA_DIR}/xgc/120k
  10000 141 10 0.5 0)
mpi_test(pseudoXGCm_120kElms_4 4
  ./pseudoXGCm
  ${TEST_DATA_DIR}/xgc/120k
  10000 141 10 0.5 0)

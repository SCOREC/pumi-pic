function(mpi_test TESTNAME PROCS EXE)
  add_test(
    NAME ${TESTNAME}
    COMMAND ${MPIRUN} ${MPIRUN_PROCFLAG} ${PROCS} ${VALGRIND} ${VALGRIND_ARGS} ${EXE} ${ARGN}
  )
endfunction(mpi_test)

mpi_test(viewComm_1 1 ./ViewCommTests)
mpi_test(viewComm_2 2 ./ViewCommTests)
mpi_test(viewComm_4 4 ./ViewCommTests)

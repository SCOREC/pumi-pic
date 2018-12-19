add_test(NAME type_test COMMAND ./typeTest)

function(test_suite TESTNAME NE NP DIST)
  add_test(NAME ${TESTNAME} COMMAND ./pskk ${NE} ${NP} ${DIST} 1 ${NP} 0)
  add_test(NAME ${TESTNAME}_sort COMMAND ./pskk ${NE} ${NP} ${DIST} ${NE} ${NP} 0)
  add_test(NAME ${TESTNAME}_slice COMMAND ./pskk ${NE} ${NP} ${DIST} 1 32 0)
  add_test(NAME ${TESTNAME}_sort_slice COMMAND ./pskk ${NE} ${NP} ${DIST} ${NE} 32 0)
endfunction(test_suite)

function(push_test TESTNAME NE NP)
  test_suite(${TESTNAME}_even ${NE} ${NP} 0)
  test_suite(${TESTNAME}_uniform ${NE} ${NP} 1)
  test_suite(${TESTNAME}_gaussian ${NE} ${NP} 2)
  test_suite(${TESTNAME}_exponential ${NE} ${NP} 3)
endfunction(push_test)

push_test(push_small 100 10000)
push_test(push_medium 10000 10000000)
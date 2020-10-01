# particle_structures

Particles strucutres for unstructed mesh particle-in-cell (PIC). 

- Sell-C-sigma (SCS) with vertical slicing 
- Compressed Sparse Row (CSR) (in progress)


# Directory Layout

- cdash
- cmake
- src
  - csr - Compressed Sparse Row implementation
  - scs - Sell-C-Sigma implementation
  - support - MemberTypeArray, Segment, and Distributor source
- test - particle structure specific test source, `ctest3` test generation files


# Running tests

Unit tests can be run with `ctest3` from the overall PUMIPic build directory.

Inidividual tests can be selected by running `ctest3 -R <test_name>` and an optional `-V` flag for output during the run. Test names are defined in `test/CMakeLists.txt` as the first argument to `make_test(test_name test.cpp)` (test commands can be found in `test/testing.cmake`). 

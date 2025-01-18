cmake_minimum_required(VERSION 3.0.0)

project(pumipic VERSION 2.1.5 LANGUAGES CXX)

include(cmake/bob.cmake)

macro(pumipic_export_lib target headers)
  bob_export_target(${target})
  install(FILES ${headers} DESTINATION include)
endmacro(pumipic_export_lib)

#look for config files first then look for modules (FindXYZ.cmake)
set(CMAKE_FIND_PACKAGE_PREFER_CONFIG ON)

bob_begin_package()

set(CMAKE_MODULE_PATH
   ${CMAKE_MODULE_PATH}
   "${CMAKE_CURRENT_SOURCE_DIR}/cmake/")

#Settings options for testing
enable_testing()
include(CTest)

option(IS_TESTING "Build for CTest" OFF)
message(STATUS "IS_TESTING: ${IS_TESTING}")

if(IS_TESTING)
  set(TEST_DATA_DIR "" CACHE PATH
    "Path to a local copy of the pumipic-data repo.")
  if(NOT EXISTS ${TEST_DATA_DIR})
    message(FATAL_ERROR "TEST_DATA_DIR \"${TEST_DATA_DIR}\" is not accessible")
  endif()
endif()

# find MPI
set(pumipic_USE_MPI_DEFAULT ON)
bob_public_dep(MPI 3)
#get the mpirun binary/script
get_filename_component(COMPILER_DIR "${CMAKE_CXX_COMPILER}" PATH)
find_program(MPIRUN NAMES mpirun PATHS "${COMPILER_DIR}")
set(MPIRUN_PROCFLAG "-np" CACHE STRING
  "the command line flag to give process count to MPIRUN")


set(pumipic_USE_Omega_h_DEFAULT ON)
set(Omega_h_PREFIX ${Omega_h_PREFIX})
bob_public_dep(Omega_h)
#Clear the omegah compilation flags that it passes to cuda.  Using the
# kokkos target, and nvcc_wrapper, provide sufficient flags.
set_property(TARGET Omega_h::omega_h PROPERTY INTERFACE_COMPILE_OPTIONS "")

include(CheckCXXSymbolExists)
check_cxx_symbol_exists(ask_revClass "Omega_h_mesh.hpp" OMEGA_HAS_REVCLASS)

set(ENGPAR_ENABLED true)
include_directories(${ENGPAR_INCLUDE_DIR})
message(STATUS "Found EnGPar")
set(pumipic_USE_EnGPar_DEFAULT ON)
set(EnGPar_PREFIX ${EnGPar_PREFIX})
bob_public_dep(EnGPar)

option(ENABLE_CABANA "Build with Cabana" OFF)
if(ENABLE_CABANA)
  enable_language(C)
  # bob package creation { no idea if this will work
  set(pumipic_USE_Cabana_DEFAULT ON)
  set(Cabana_PREFIX ${Cabana_PREFIX})
  bob_public_dep(Cabana)
  # }
  add_definitions(-DPP_ENABLE_CAB)
  if(Cabana_VERSION VERSION_LESS 0.6.0 OR Cabana_VERSION STREQUAL "1.0-dev")
    message(FATAL_ERROR "Cabana version >= 0.6.0 required.")
  endif()
endif()

set(pumipic_USE_Kokkos_DEFAULT ON)
bob_public_dep(Kokkos)
set(KOKKOS_ENABLED true)

if(Kokkos_VERSION VERSION_LESS 4.0.01)
    message(FATAL_ERROR "Kokkos version >= 4.0.01 required.")
endif()

if(Omega_h_VERSION VERSION_LESS 10.8.0)
    message(FATAL_ERROR "Omega_h version >= 10.8.0 required.")
endif()

if(EnGPar_VERSION VERSION_LESS 1.1.0)
    message(FATAL_ERROR "EnGPar version >= 1.1.0 required.")
endif()

option(PUMIPIC_PRINT_ENABLED "PUMIPIC print statements enabled" ON)
find_package(spdlog QUIET)

set(debug_flag)
if (PP_ENABLE_DEBUG_SYMBOLS)
  set(debug_flag "-g")
endif()

set(opt_flag)
if (PP_ENABLE_OPT)
  set(opt_flag "-O3")
endif()

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${debug_flag} ${opt_flag}")
message(STATUS "CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS}")

if (PP_ENABLE_MEMCHECK)
  find_program(VALGRIND "valgrind")
  set(VALGRIND_ARGS "--suppressions=${CMAKE_SOURCE_DIR}/valgrind.supp" "--leak-check=yes" "--error-exitcode=1")
endif()

# testing helper function
function(mpi_test TESTNAME PROCS EXE)
  add_test(
    NAME ${TESTNAME}
    COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} ${PROCS} ${VALGRIND} ${VALGRIND_ARGS} ${EXE} ${ARGN}
  )
endfunction(mpi_test)

# support
add_subdirectory(support)
include_directories(support)

set(ALL_LIBS
  support
  )
if (PP_ENABLE_DEBUG_MODE)
   add_definitions(-DPP_DEBUG)
   target_compile_definitions(support INTERFACE -DPP_DEBUG)
endif()

# particle structures
add_subdirectory(particle_structs)
set(ALL_LIBS
  support
  particleStructs
  )

# Reset cxx flags since they are picked up from omega-h
set(CMAKE_CXX_FLAGS "${debug_flag} ${opt_flag}")

# pumipic
add_subdirectory(src)
set(ALL_LIBS
  ${ALL_LIBS}
  pumipic
  )

if(IS_TESTING)
  add_subdirectory(test)
  add_subdirectory(performance_tests)
endif()

#check for mallinfo2
if(${CMAKE_CXX_COMPILER_ID} STREQUAL "GNU" OR ${CMAKE_CXX_COMPILER_ID} STREQUAL "Clang")
  include(CheckCXXSymbolExists)
  check_cxx_symbol_exists(mallinfo2 "malloc.h" PUMIPIC_HAS_MALLINFO2)
  if(PUMIPIC_HAS_MALLINFO2)
    target_compile_definitions(support INTERFACE -DPUMIPIC_HAS_MALLINFO2)
  endif()
endif()

bob_end_package()

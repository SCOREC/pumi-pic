cmake_minimum_required(VERSION 3.0.0)

project(pumipic VERSION 1.0.0 LANGUAGES CXX)

include(cmake/bob.cmake)

macro(pumipic_export_lib target headers)
  bob_export_target(${target})
  install(FILES ${headers} DESTINATION include)
endmacro(pumipic_export_lib)

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

#get the mpirun binary/script
get_filename_component(COMPILER_DIR "${CMAKE_CXX_COMPILER}" PATH)
find_program(MPIRUN NAMES mpirun PATHS "${COMPILER_DIR}")
set(MPIRUN_PROCFLAG "-np" CACHE STRING
  "the command line flag to give process count to MPIRUN")

find_package(Omega_h 9 REQUIRED)
set(pumipic_USE_Omega_h_DEFAULT ON)
message(STATUS "Found Omega_h: ${Omega_h_VERSION}")
bob_public_dep(Omega_h)

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


#particle structures
add_subdirectory(particle_structs)
set(ALL_LIBS
  support
  particleStructs
  )
if (PP_ENABLE_DEBUG_MODE)
   add_definitions(-DPP_DEBUG)
   target_compile_definitions(support INTERFACE -DPP_DEBUG)
endif()

#pumipic
add_subdirectory(src)
set(ALL_LIBS
  ${ALL_LIBS}
  pumipic
  )

#xgcp
add_subdirectory(xgcp)
set(ALL_LIBS
  ${ALL_LIBS}
  xgcp
  )

if(IS_TESTING)
  add_subdirectory(test)
endif()

bob_end_package()

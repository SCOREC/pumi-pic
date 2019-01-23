#Setting Version Number, Project Name
cmake_minimum_required (VERSION 3.0)
project(particle_structures VERSION 0.0.1 LANGUAGES CXX)

include(cmake/bob.cmake)
bob_begin_package()

set(CMAKE_MODULE_PATH 
   ${CMAKE_MODULE_PATH} 
   "${CMAKE_CURRENT_SOURCE_DIR}/cmake/")

#Settings options for testing
enable_testing()
include(CTest)


option(DEBUG "Turn on for debug output" OFF)

if (DEBUG)
   add_definitions(-DDEBUG)
endif()

option(IS_TESTING "Build for CTest" OFF)
message(STATUS "IS_TESTING: ${IS_TESTING}")

if (ENABLE_KOKKOS)
  find_package(KokkosCore REQUIRED)
  add_definitions(-DKOKKOS_ENABLED)
  set(KOKKOS_ENABLED true)
  include_directories(${KokkosCore_INCLUDE_DIRS})

  message(STATUS "Kokkos enabled - Overriding compiler selection for consistency with Kokkos install")
  set(CMAKE_CXX_COMPILER ${KokkosCore_CXX_COMPILER})
  set(CMAKE_C_COMPILER ${KokkosCore_C_COMPILER})
  set(CMAKE_FORTRAN_COMPILER ${KokkosCore_FORTRAN_COMPILER})
  message(STATUS "Kokkos CMAKE_CXX_COMPILER ${KokkosCore_CXX_COMPILER}")
  message(STATUS "Kokkos CMAKE_C_COMPILER ${KokkosCore_C_COMPILER}")
  message(STATUS "Kokkos CMAKE_FORTRAN_COMPILER ${KokkosCore_FORTRAN_COMPILER}")

  message(STATUS "Kokkos enabled - Overriding compiler flags for consistency with Kokkos install")
  set(CMAKE_CXX_FLAGS ${KokkosCore_CXX_FLAGS})
  set(CMAKE_C_FLAGS ${KokkosCore_C_FLAGS})
  set(CMAKE_FORTRAN_FLAGS ${KokkosCore_FORTRAN_FLAGS})
  message(STATUS "Kokkos CMAKE_CXX_FLAGS ${KokkosCore_CXX_FLAGS}")
  message(STATUS "Kokkos CMAKE_C_FLAGS ${KokkosCore_C_FLAGS}")
  message(STATUS "Kokkos CMAKE_FORTRAN_FLAGS ${KokkosCore_FORTRAN_FLAGS}")
endif()

include_directories(support)
include_directories(algorithms)

set(PS_SOURCES 
  algorithms/Distribute.cpp
  algorithms/Push.cpp
  algorithms/psTypes.cpp
  support/psAssert.cpp
)

set(PS_HEADERS
  support/SCS_Macros.h
  support/SellCSigma.h
  support/MemberTypes.h
  algorithms/psParams.h
  algorithms/psTypes.h
)

add_library(particleStructs ${PS_SOURCES})
install(FILES ${PS_HEADERS} DESTINATION include)
bob_export_target(particleStructs)

if (ENABLE_KOKKOS)
  # bob package creation {
  set(particle_structures_USE_KokkosCore_DEFAULT ON)
  get_filename_component(KokkosCore_PREFIX
    ${KokkosCore_INCLUDE_DIRS} DIRECTORY CACHE)
  bob_public_dep(KokkosCore)
  # }
  target_include_directories(particleStructs INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/support>
    $<INSTALL_INTERFACE:include>)
  target_include_directories(particleStructs INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/algorithms>
    $<INSTALL_INTERFACE:include>)
  target_include_directories(particleStructs PRIVATE ${KokkosCore_INCLUDE_DIRS})
  target_include_directories(particleStructs PRIVATE ${KokkosKernels_INCLUDE_DIRS})
  target_link_libraries(particleStructs PRIVATE
    ${KokkosCore_LIBRARIES} ${KokkosKernels_LIBRARIES})
endif()

option(FP64 "Use 64bits for floating point values" ON)
option(FP32 "Use 32bits for floating point values" OFF)
message(STATUS "FP64: ${FP64}")
message(STATUS "FP32: ${FP32}")
if( (FP64 AND FP32) OR (NOT FP64 AND NOT FP32) )
  message(FATAL_ERROR "Enable either FP64 or FP32, but not both")
endif()
if( FP64 )
  add_definitions(-DFP64)
  target_compile_definitions(particleStructs INTERFACE -DFP64)
endif()
if( FP32 )
  add_definitions(-DFP32)
  target_compile_definitions(particleStructs INTERFACE -DFP32)
endif()

add_subdirectory(test)

bob_end_package()

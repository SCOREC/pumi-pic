#Setting Version Number, Project Name
cmake_minimum_required (VERSION 3.0)
cmake_policy(SET CMP0054 NEW)
project(particle_structures VERSION 0.0.1 LANGUAGES CXX)

include(cmake/bob.cmake)

macro(ps_export_lib target headers)
  bob_export_target(${target})
  install(FILES ${headers} DESTINATION include)
endmacro(ps_export_lib)

bob_begin_package()

set(CMAKE_MODULE_PATH
   ${CMAKE_MODULE_PATH}
   "${CMAKE_CURRENT_SOURCE_DIR}/cmake/")

#Settings options for testing
enable_testing()
include(CTest)


option(PS_DEBUG "Turn on for debug output" OFF)
option(PS_OPT "Turn on for debug output" OFF)

if (PS_DEBUG_PREPROC)
   add_definitions(-DDEBUG)
endif()

set(debug_flag)
if (PS_ENABLE_DEBUG_SYMBOLS)
  set(debug_flag "-g")
endif()

set(opt_flag)
if (PS_ENABLE_OPT)
  set(opt_flag "-O3")
endif()

option(IS_TESTING "Build for CTest" OFF)
message(STATUS "IS_TESTING: ${IS_TESTING}")

if (ENABLE_KOKKOS)
  find_package(Kokkos REQUIRED)
  set(KOKKOS_ENABLED true)
  include_directories(${KOKKOS_INCLUDE_DIR})

  include(${KOKKOS_INCLUDE_DIR}/../kokkos_generated_settings.cmake)
  message(STATUS "Kokkos enabled - Overriding compiler flags for consistency with Kokkos install")
  string(REPLACE ";" " " CMAKE_CXX_FLAGS "${KOKKOS_CXX_FLAGS}")
  message(STATUS "Kokkos CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS}")

  # bob package creation {
  set(particle_structures_USE_Kokkos_DEFAULT ON)
  get_filename_component(Kokkos_PREFIX ${KOKKOS_INCLUDE_DIR} DIRECTORY)
  set(Kokkos_PREFIX_DEFAULT ${Kokkos_PREFIX}/lib/CMake)
  bob_public_dep(Kokkos)
  # }
endif()

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${debug_flag} ${opt_flag}")
message(STATUS "CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS}")

add_subdirectory(support)
include_directories(support)
add_subdirectory(src)
include_directories(src)
include_directories(algorithms)

set(PS_SOURCES
  algorithms/psTypes.cpp
)

set(PS_HEADERS
  algorithms/psParams.h
  algorithms/psTypes.h
)


if(false)
if (ENABLE_KOKKOS)
  add_definitions(-DKOKKOS_ENABLED)
  target_compile_definitions(particleStructs INTERFACE -DKOKKOS_ENABLED)
  target_include_directories(particleStructs INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/support>
    $<INSTALL_INTERFACE:include>)
  target_include_directories(particleStructs INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/algorithms>
    $<INSTALL_INTERFACE:include>)
  target_include_directories(particleStructs PRIVATE ${KOKKOS_INCLUDE_DIR})
  target_link_libraries(particleStructs PRIVATE
    ${KOKKOS_LIBRARY})

  include(${CMAKE_CURRENT_LIST_DIR}/cmake/detect_kokkos_opts.cmake)
  detect_kokkos_opts()
  if (Kokkos_HAS_CUDA)
    add_definitions(-DPS_USE_CUDA)
    target_compile_definitions(particleStructs INTERFACE -DPS_USE_CUDA)
  endif()
  if (Kokkos_HAS_OpenMP)
    add_definitions(-DPS_USE_OPENMP)
    target_compile_definitions(particleStructs INTERFACE -DPS_USE_OPENMP)
  endif()
endif()
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

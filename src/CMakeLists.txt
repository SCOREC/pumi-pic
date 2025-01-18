configure_file(
  pumipic_version.hpp.in
  pumipic_version.hpp
)

set(HEADERS
  pumipic_adjacency.hpp
  pumipic_adjacency.tpp
  pumipic_push.hpp
  pumipic_lb.hpp
  pumipic_ptcl_ops.hpp
  pumipic_utils.hpp
  pumipic_constants.hpp
  pumipic_mesh.hpp
  pumipic_library.hpp
  pumipic_input.hpp
  pumipic_kktypes.hpp
  pumipic_profiling.hpp
  ${CMAKE_CURRENT_BINARY_DIR}/pumipic_version.hpp
)

set(SOURCES
  pumipic_lb.cpp
  pumipic_input.cpp
  pumipic_part_construct.cpp
  pumipic_comm.cpp
  pumipic_kktypes.cpp
  pumipic_mesh.cpp
  pumipic_library.cpp
  pumipic_profiling.cpp
  pumipic_file.cpp
)

add_library(pumipic-core ${SOURCES})
target_include_directories(pumipic-core INTERFACE
  $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
  $<INSTALL_INTERFACE:include>
)
target_link_libraries(pumipic-core
  support
  particleStructs
  Omega_h::omega_h
  Kokkos::kokkos
  EnGPar::engpar
  EnGPar::pcu
  )
pumipic_export_lib(pumipic-core "${HEADERS}")

# this INTERFACE target bundles all the enabled libraries together
add_library(pumipic INTERFACE)
target_link_libraries(pumipic INTERFACE ${pumipic_EXPORTED_TARGETS})
bob_export_target(pumipic)

#Floating point definition
option(FP64 "Use 64bits for floating point values" ON)
option(FP32 "Use 32bits for floating point values" OFF)
message(STATUS "FP64: ${FP64}")
message(STATUS "FP32: ${FP32}")
if( (FP64 AND FP32) OR (NOT FP64 AND NOT FP32) )
  message(FATAL_ERROR "Enable either FP64 or FP32, but not both")
endif()
if( FP64 )
  add_definitions(-DFP64)
  target_compile_definitions(pumipic INTERFACE -DFP64)
endif()
if( FP32 )
  add_definitions(-DFP32)
  target_compile_definitions(pumipic INTERFACE -DFP32)
endif()

bob_end_subdir()

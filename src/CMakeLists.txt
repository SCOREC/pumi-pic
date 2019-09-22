set(HEADERS
  pumipic_adjacency.hpp
  pumipic_push.hpp
  pumipic_utils.hpp
  pumipic_constants.hpp
  pumipic_mesh.hpp
  pumipic_library.hpp
  pumipic_input.hpp
)

set(SOURCES
  pumipic_input.cpp
  pumipic_part_construct.cpp
  pumipic_comm.cpp
  pumipic_mesh.cpp
  pumipic_library.cpp
  GitrmMesh.cpp
  GitrmParticles.cpp
  GitrmIonizeRecombine.cpp
)
add_library(pumipic ${SOURCES})
target_include_directories(pumipic INTERFACE
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
  $<INSTALL_INTERFACE:include>
)
target_link_libraries(pumipic particle_structures::particleStructs Omega_h::omega_h)
pumipic_export_lib(pumipic "${HEADERS}")

bob_end_subdir()

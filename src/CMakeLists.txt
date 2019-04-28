set(HEADERS
  pumipic_adjacency.hpp
  pumipic_push.hpp
  pumipic_utils.hpp
  pumipic_constants.hpp
  pumipi_part_construct.hpp
  GitrmMesh.hpp
  )

set(SOURCES
  pumipic_part_construct.cpp
  GitrmMesh.cpp
)
add_library(pumipic ${SOURCES})
target_include_directories(pumipic INTERFACE
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
  $<INSTALL_INTERFACE:include>
)
target_link_libraries(pumipic particle_structures::particleStructs)
pumipic_export_lib(pumipic "${HEADERS}")

bob_end_subdir()

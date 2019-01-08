set(HEADERS
  pumipic_adjacency.hpp
  pumipic_push.hpp
  pumipic_utils.hpp
  pumipic_constants.hpp
  )

add_library(pumipic INTERFACE)
target_include_directories(pumipic INTERFACE
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
  $<INSTALL_INTERFACE:include>
)
pumipic_export_lib(pumipic "${HEADERS}")

bob_end_subdir()

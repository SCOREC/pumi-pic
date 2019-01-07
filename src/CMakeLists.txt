set(HEADERS
  gitrm_adjacency.hpp
  gitrm_push.hpp
  gitrm_utils.hpp
  gitrm_constants.hpp
  )

add_library(pumipic INTERFACE)
target_include_directories(pumipic INTERFACE
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
  $<INSTALL_INTERFACE:include>
)
pumipic_export_lib(pumipic "${HEADERS}")

bob_end_subdir()

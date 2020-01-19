set(HEADERS
  xgcp_types.hpp
  xgcp_input.hpp
  xgcp_mesh.hpp
  xgcp_gyro_scatter.hpp
  xgcp_push.hpp
  xgcp_particle.hpp
)

set(SOURCES
  xgcp_input.cpp
  xgcp_mesh.cpp
  xgcp_gyro_scatter.cpp
  xgcp_elliptical_push.cpp
  xgcp_particle.cpp
)

add_library(xgcp ${SOURCES})
target_include_directories(xgcp INTERFACE
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
  $<INSTALL_INTERFACE:include>
)
target_link_libraries(xgcp ${ALL_LIBS} Omega_h::omega_h)
pumipic_export_lib(xgcp "${HEADERS}")

bob_end_subdir()

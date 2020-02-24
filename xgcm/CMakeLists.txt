set(HEADERS
  xgcm_types.hpp
  xgcm_input.hpp
  xgcm_mesh.hpp
  xgcm_gyro_scatter.hpp
  xgcm_push.hpp
  xgcm_particle.hpp
)

set(SOURCES
  xgcm_input.cpp
  xgcm_mesh.cpp
  xgcm_gyro_scatter.cpp
  xgcm_elliptical_push.cpp
  xgcm_particle.cpp
)

add_library(xgcm ${SOURCES})
target_include_directories(xgcm INTERFACE
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
  $<INSTALL_INTERFACE:include>
)
target_link_libraries(xgcm ${ALL_LIBS} Omega_h::omega_h)
pumipic_export_lib(xgcm "${HEADERS}")

bob_end_subdir()

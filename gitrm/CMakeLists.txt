set(HEADERS
  GitrmIonizeRecombine.hpp
  GitrmPush.hpp
  GitrmMesh.hpp
  GitrmParticles.hpp
  GitrmInputOutput.hpp
  GitrmSurfaceModel.hpp
  # GitrmCoulombCollision.hpp
  # GitrmThermalForce.hpp
)

set(SOURCES
  GitrmMesh.cpp
  GitrmParticles.cpp
  GitrmIonizeRecombine.cpp
  GitrmInputOutput.cpp
  GitrmSurfaceModel.cpp
 )

add_library(gitrm ${SOURCES})
target_include_directories(gitrm INTERFACE
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
  $<INSTALL_INTERFACE:include>
)
target_link_libraries(gitrm ${ALL_LIBS} Omega_h::omega_h NetCDF::NetCDF)
pumipic_export_lib(gitrm "${HEADERS}")

bob_end_subdir()

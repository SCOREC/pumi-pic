include_directories(
  ${CMAKE_CURRENT_SOURCE_DIR}/scs
  ${CMAKE_CURRENT_SOURCE_DIR}/csr
  ${CMAKE_CURRENT_SOURCE_DIR}

  )

set(HEADERS
  particle_structure.hpp
  ps_for.hpp
  scs/SCS_Macros.h
  scs/SCS_Types.h
  scs/SCSPair.h
  scs/SCS_sort.h
  scs/SCS_rebuild.h
  scs/SCS_migrate.h
  scs/SCS_buildFns.h
  scs/SellCSigma.h
  scs/scs_input.hpp
  csr/CSR.hpp
  particle_structs.hpp
)

set(SOURCES
  scs/SellCSigma.cpp
)

add_library(particleStructs ${SOURCES})
target_link_libraries(particleStructs support)
target_include_directories(particleStructs INTERFACE
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
  $<INSTALL_INTERFACE:include>)
target_include_directories(particleStructs INTERFACE
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/scs>
  $<INSTALL_INTERFACE:include>)

pumipic_export_lib(particleStructs "${HEADERS}")

bob_end_subdir()

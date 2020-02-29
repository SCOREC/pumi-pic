set(HEADERS
  ppTypes.h
  ppMacros.h
  ppView.h
  SupportKK.h
  ViewComm.h
  ppAssert.h
)

set(SOURCES
  ppAssert.cpp
  ViewComm.cpp
)

add_library(support ${SOURCES})
target_include_directories(support INTERFACE
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
  $<INSTALL_INTERFACE:include>
  )


add_definitions(-DKOKKOS_ENABLED)
target_compile_definitions(support INTERFACE -DKOKKOS_ENABLED)
target_include_directories(support INTERFACE
$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/support>
$<INSTALL_INTERFACE:include>)
target_include_directories(support INTERFACE
$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/support>
$<INSTALL_INTERFACE:include>)
target_include_directories(support PRIVATE ${KOKKOS_INCLUDE_DIR})
target_link_libraries(support PRIVATE
${KOKKOS_LIBRARY})

include(${CMAKE_SOURCE_DIR}/cmake/detect_kokkos_opts.cmake)
detect_kokkos_opts()
if (Kokkos_HAS_CUDA)
add_definitions(-DPS_USE_CUDA)
target_compile_definitions(support INTERFACE -DPS_USE_CUDA)
endif()
if (Kokkos_HAS_OpenMP)
add_definitions(-DPS_USE_OPENMP)
target_compile_definitions(support INTERFACE -DPS_USE_OPENMP)
endif()

pumipic_export_lib(support "${HEADERS}")

bob_end_subdir()

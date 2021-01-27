set(HEADERS
  ppTypes.h
  ppMacros.h
  ppView.h
  ppArray.h
  SupportKK.h
  ViewComm.h
  ViewComm_host.hpp
  ViewComm_cuda.hpp
  ppAssert.h
  ppTiming.hpp
)

set(SOURCES
  ppTiming.cpp
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
target_link_libraries(support PUBLIC Kokkos::kokkos)
if(ENABLE_CABANA)
  target_link_libraries(support PUBLIC Cabana::cabanacore)
endif()

cmake_policy(SET CMP0057 NEW)
if ("CUDA" IN_LIST Kokkos_DEVICES)
  add_definitions(-DPP_USE_CUDA)
  target_compile_definitions(support INTERFACE -DPP_USE_CUDA)
endif()
if ("OpenMP" IN_LIST Kokkos_DEVICES)
  add_definitions(-DPP_USE_OPENMP)
  target_compile_definitions(support INTERFACE -DPP_USE_OPENMP)
endif()

pumipic_export_lib(support "${HEADERS}")

if(IS_TESTING)
  add_executable(ViewCommTests ViewComm_test.cpp)
  target_link_libraries(ViewCommTests support)
  include(testing.cmake)
endif()

bob_end_subdir()

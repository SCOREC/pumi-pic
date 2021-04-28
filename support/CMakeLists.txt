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
target_include_directories(support PUBLIC ${KOKKOS_PATH}/include)
if(ENABLE_CABANA)
  target_include_directories(support PUBLIC ${Cabana_INCLUDE_DIR})
  target_link_libraries(support PUBLIC ${Cabana_LIBRARY} kokkos)
else()
  target_include_directories(support PUBLIC ${Kokkos_INCLUDE_DIRS})
  target_link_libraries(support PUBLIC ${Kokkos_LIBRARIES})
endif()

string(REPLACE "," ";" KOKKOS_DEVICES_MIXED_CASE ${KOKKOS_GMAKE_DEVICES})
string(TOUPPER "${KOKKOS_DEVICES_MIXED_CASE}" KOKKOS_DEVICES)
cmake_policy(SET CMP0057 NEW)
if ("CUDA" IN_LIST KOKKOS_DEVICES)
  message(STATUS "KOKKOS has CUDA backend")
  add_definitions(-DPP_USE_CUDA)
  target_compile_definitions(support INTERFACE -DPP_USE_CUDA)
endif()
if ("OPENMP" IN_LIST KOKKOS_DEVICES)
  message(STATUS "KOKKOS has OPENMP backend")
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

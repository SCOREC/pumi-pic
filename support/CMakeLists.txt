set(HEADERS
  ppTypes.h
  ppMacros.h
  ppView.h
  ppArray.h
  SupportKK.h
  ViewComm.h
  ViewComm_host.hpp
  ViewComm_gpu.hpp
  ppAssert.h
  ppTiming.hpp
  ppMemUsage.hpp
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
target_link_libraries(support PUBLIC MPI::MPI_CXX)
if(ENABLE_CABANA)
  target_link_libraries(support PUBLIC Cabana::Core)
endif()

cmake_policy(SET CMP0057 NEW)
if ("CUDA" IN_LIST Kokkos_DEVICES)
  add_definitions(-DPP_USE_CUDA)
  target_compile_definitions(support INTERFACE -DPP_USE_CUDA)
endif()
if ("OPENMP" IN_LIST Kokkos_DEVICES)
  add_definitions(-DPP_USE_OPENMP)
  target_compile_definitions(support INTERFACE -DPP_USE_OPENMP)
endif()
if ("HIP" IN_LIST Kokkos_DEVICES)
  add_definitions(-DPP_USE_HIP)
  target_compile_definitions(support INTERFACE -DPP_USE_HIP)
endif()
if (("CUDA" IN_LIST Kokkos_DEVICES) OR ("HIP" IN_LIST Kokkos_DEVICES))
  add_definitions(-DPP_USE_GPU)
  target_compile_definitions(support INTERFACE -DPP_USE_GPU)
endif()

if(PS_USE_GPU_AWARE_MPI)
  add_definitions(-DPS_GPU_AWARE_MPI)
  target_compile_definitions(support INTERFACE -DPS_GPU_AWARE_MPI)
endif()

pumipic_export_lib(support "${HEADERS}")

if(IS_TESTING)
  add_executable(ViewCommTests ViewComm_test.cpp)
  target_link_libraries(ViewCommTests support)
  include(testing.cmake)
endif()

bob_end_subdir()

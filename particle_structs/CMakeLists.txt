#Setting Version Number, Project Name
cmake_minimum_required (VERSION 3.0)
project(particle_structures VERSION 0.0.1 LANGUAGES CXX C)

set(CMAKE_MODULE_PATH 
   ${CMAKE_MODULE_PATH} 
   "${CMAKE_CURRENT_SOURCE_DIR}/cmake/")

#Settings options for testing
enable_testing()
include(CTest)

option(IS_TESTING "Build for CTest" OFF)
message(STATUS "IS_TESTING: ${IS_TESTING}")

if (ENABLE_KOKKOS)
  find_package(KokkosCore REQUIRED)
  find_package(KokkosKernels REQUIRED)
  add_definitions(-DKOKKOS_ENABLED)
  set(KOKKOS_ENABLED true)
  include_directories(${KokkosCore_INCLUDE_DIRS})

  message(STATUS "Kokkos enabled - Overriding compiler selection for consistency with Kokkos install")
  set(CMAKE_CXX_COMPILER ${KokkosCore_CXX_COMPILER})
  set(CMAKE_C_COMPILER ${KokkosCore_C_COMPILER})
  set(CMAKE_FORTRAN_COMPILER ${KokkosCore_FORTRAN_COMPILER})
  message(STATUS "Kokkos CMAKE_CXX_COMPILER ${KokkosCore_CXX_COMPILER}")
  message(STATUS "Kokkos CMAKE_C_COMPILER ${KokkosCore_C_COMPILER}")
  message(STATUS "Kokkos CMAKE_FORTRAN_COMPILER ${KokkosCore_FORTRAN_COMPILER}")

  message(STATUS "Kokkos enabled - Overriding compiler flags for consistency with Kokkos install")
  set(CMAKE_CXX_FLAGS ${KokkosCore_CXX_FLAGS})
  set(CMAKE_C_FLAGS ${KokkosCore_C_FLAGS})
  set(CMAKE_FORTRAN_FLAGS ${KokkosCore_FORTRAN_FLAGS})
  message(STATUS "Kokkos CMAKE_CXX_FLAGS ${KokkosCore_CXX_FLAGS}")
  message(STATUS "Kokkos CMAKE_C_FLAGS ${KokkosCore_C_FLAGS}")
  message(STATUS "Kokkos CMAKE_FORTRAN_FLAGS ${KokkosCore_FORTRAN_FLAGS}")
endif()

set(PS_SOURCES 
  Distribute.cpp
  Push.cpp
  SellCSigma.cpp
)

add_library(particleStructs ${PS_SOURCES})

if (ENABLE_KOKKOS)
  target_include_directories(particleStructs PRIVATE ${KokkosCore_INCLUDE_DIRS})
  target_include_directories(particleStructs PRIVATE ${KokkosKernels_INCLUDE_DIRS})
  target_link_libraries(particleStructs PRIVATE
    ${KokkosCore_LIBRARIES} ${KokkosKernels_LIBRARIES})
endif()

add_executable(pstest main.cpp)
target_link_libraries(pstest particleStructs) 
add_executable(pskk pskk.cpp)
target_link_libraries(pskk particleStructs)

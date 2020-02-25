set(CMAKE_MODULE_PATH
   ${CMAKE_MODULE_PATH}
   "${CMAKE_CURRENT_SOURCE_DIR}/cmake/")

if (PS_DEBUG_PREPROC)
   add_definitions(-DDEBUG)
endif()


option(PS_IS_TESTING "Build for CTest" OFF)
message(STATUS "PS_IS_TESTING: ${PS_IS_TESTING}")


# Find Kokkos
find_package(Kokkos REQUIRED)
set(KOKKOS_ENABLED true)
include_directories(${KOKKOS_INCLUDE_DIR})

include(${KOKKOS_INCLUDE_DIR}/../kokkos_generated_settings.cmake)
message(STATUS "Kokkos enabled - Overriding compiler flags for consistency with Kokkos install")
string(REPLACE ";" " " CMAKE_CXX_FLAGS "${KOKKOS_CXX_FLAGS}")
message(STATUS "Kokkos CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS}")

# bob package creation {
set(particle_structures_USE_Kokkos_DEFAULT ON)
get_filename_component(Kokkos_PREFIX ${KOKKOS_INCLUDE_DIR} DIRECTORY)
set(Kokkos_PREFIX_DEFAULT ${Kokkos_PREFIX}/lib/CMake)
bob_public_dep(Kokkos)
# }
set(debug_flag)
if (PP_ENABLE_DEBUG_SYMBOLS)
  set(debug_flag "-g")
endif()

set(opt_flag)
if (PP_ENABLE_OPT)
  set(opt_flag "-O3")
endif()

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${debug_flag} ${opt_flag}")
message(STATUS "CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS}")

add_subdirectory(support)
include_directories(support)
add_subdirectory(src)
include_directories(src)

if (PS_IS_TESTING)
   add_subdirectory(test)
endif()
bob_end_subdir()

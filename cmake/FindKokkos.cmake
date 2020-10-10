# - Try to find kokkos
# Once done this will define
#  KOKKOS_FOUND - System has KOKKOS
#  KOKKOS_INCLUDE_DIRS - The KOKKOS include directories
#  KOKKOS_LIBRARIES - The libraries needed to use KOKKOS

set(KOKKOS_PREFIX "" CACHE STRING "Zoltan install directory")
if(KOKKOS_PREFIX)
  message(STATUS "KOKKOS_PREFIX ${KOKKOS_PREFIX}")
endif()

find_path(KOKKOS_INCLUDE_DIR Kokkos_Core.hpp PATHS "${KOKKOS_PREFIX}/include")

find_library(KOKKOS_LIBRARY kokkos PATHS "${KOKKOS_PREFIX}/lib")

set(KOKKOS_LIBRARIES ${KOKKOS_LIBRARY} )
set(KOKKOS_INCLUDE_DIRS ${KOKKOS_INCLUDE_DIR} )

include(FindPackageHandleStandardArgs)
# handle the QUIETLY and REQUIRED arguments and set KOKKOS_FOUND to TRUE
# if all listed variables are TRUE
find_package_handle_standard_args(
    Kokkos
    DEFAULT_MSG
    KOKKOS_LIBRARY KOKKOS_INCLUDE_DIR
)

mark_as_advanced(KOKKOS_INCLUDE_DIR KOKKOS_LIBRARY )

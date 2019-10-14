# - Try to find kokkos
# Once done this will define
#  NETCDF_FOUND - System has NETCDF
#  NETCDF_INCLUDE_DIRS - The NETCDF include directories
#  NETCDF_LIBRARIES - The libraries needed to use NETCDF

set(NETCDF_PREFIX "" CACHE STRING "NETCDF install directory")
if(NETCDF_PREFIX)
  message(STATUS "NETCDF_PREFIX ${NETCDF_PREFIX}")
endif()

find_path(NETCDF_INCLUDE_DIR netcdf PATHS "${NETCDF_PREFIX}/include")

find_library(NETCDF_LIBRARY netcdf-cxx4 PATHS "${NETCDF_PREFIX}/lib")

set(NETCDF_LIBRARIES ${NETCDF_LIBRARY} )
set(NETCDF_INCLUDE_DIRS ${NETCDF_INCLUDE_DIR} )

include(FindPackageHandleStandardArgs)
# handle the QUIETLY and REQUIRED arguments and set NETCDF_FOUND to TRUE
# if all listed variables are TRUE
find_package_handle_standard_args(
    NETCDF
    DEFAULT_MSG
    NETCDF_LIBRARY NETCDF_INCLUDE_DIR
)

mark_as_advanced(NETCDF_INCLUDE_DIR NETCDF_LIBRARY )

if (NETCDF_FOUND)
  if (NOT TARGET NetCDF::NetCDF)
    add_library(NetCDF::NetCDF UNKNOWN IMPORTED)
    set_target_properties(NetCDF::NetCDF PROPERTIES
      IMPORTED_LOCATION "${NETCDF_LIBRARY}"
      INTERFACE_INCLUDE_DIRECTORIES "${NETCDF_INCLUDE_DIR}")
  endif ()
endif ()


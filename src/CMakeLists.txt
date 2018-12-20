
if(NOT CMAKE_BUILD_TYPE)
	set(CMAKE_BUILD_TYPE Debug)
endif(NOT CMAKE_BUILD_TYPE)
message("CMAKE_BUILD_TYPE = ${CMAKE_BUILD_TYPE}")

find_package(Omega_h REQUIRED 8.2.0)
message(STATUS "Found Omega_h: ${Omega_h_VERSION}")
set(SOURCES
  gitrm.cpp
  )
set(HEADERS
  gitrm_adjacency.hpp
  gitrm_utils.hpp
  )

add_executable(gitrm "${SOURCES}" "${HEADERS}")
target_link_libraries(gitrm Omega_h::omega_h)

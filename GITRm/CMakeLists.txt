
project(gitrm VERSION 0.0.1 LANGUAGES C CXX)


if(NOT CMAKE_BUILD_TYPE)
	set(CMAKE_BUILD_TYPE Debug)
endif(NOT CMAKE_BUILD_TYPE)
message("CMAKE_BUILD_TYPE = ${CMAKE_BUILD_TYPE}")

find_package(Omega_h REQUIRED 8.2.0 PATHS ${OMEGA_H_PREFIX})
set(SOURCES
  gitrm.cpp
  )
set(HEADERS
  )


add_executable(gitrm "${SOURCES}")
target_link_libraries(gitrm Omega_h::omega_h)

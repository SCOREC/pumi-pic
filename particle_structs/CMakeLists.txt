set(CMAKE_MODULE_PATH
   ${CMAKE_MODULE_PATH}
   "${CMAKE_CURRENT_SOURCE_DIR}/cmake/")

if (PS_DEBUG_PREPROC)
   add_definitions(-DDEBUG)
endif()


option(PS_IS_TESTING "Build for CTest" OFF)
message(STATUS "PS_IS_TESTING: ${PS_IS_TESTING}")

add_subdirectory(src)
include_directories(src)

if (PS_IS_TESTING)
   add_subdirectory(test)
endif()
bob_end_subdir()

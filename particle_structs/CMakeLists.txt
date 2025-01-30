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

# #define 'smoke tests' to test the install
# add_custom_target(test_install DEPENDS check) # maintain test_install target
# if(NOT BUILD_TESTING)
#   add_custom_target(check
#     COMMAND ${CMAKE_CTEST_COMMAND} -R smoke_test_particle
#     COMMENT "Test installed PUMIPic utilities")
# endif()

# add_executable(smoke_test_particle test/smoke_test_particle.cpp)
# target_link_libraries(smoke_test_particle particleStructs)
# mpi_test(smoke_test_particle 4 ./smoke_test_particle)

if (PS_IS_TESTING)
   add_subdirectory(test)
endif()
bob_end_subdir()

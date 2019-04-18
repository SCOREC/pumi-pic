add_executable(typeTest typeTest.cpp)
target_link_libraries(typeTest particleStructs)

add_executable(rebuild rebuild.cpp)
target_link_libraries(rebuild particleStructs)

add_executable(lambdaTest lambdaTest.cpp)
target_link_libraries(lambdaTest particleStructs)

include(testing.cmake)

bob_end_subdir()

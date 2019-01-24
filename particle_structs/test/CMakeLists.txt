add_executable(typeTest typeTest.cpp)
target_link_libraries(typeTest particleStructs)

add_executable(pskk pskk.cpp)
target_link_libraries(pskk particleStructs)

add_executable(rebuild rebuild.cpp)
target_link_libraries(rebuild particleStructs)

include(testing.cmake)

bob_end_subdir()

add_executable(typeTest typeTest.cpp)
target_link_libraries(typeTest particleStructs)

add_executable(pskk pskk.cpp)
target_link_libraries(pskk particleStructs)


include(testing.cmake)

function(make_test exename srcname)
  add_executable(${exename} ${srcname})
  target_link_libraries(${exename} pumipic Omega_h::omega_h
    particle_structures::particleStructs)
endfunction(make_test)

make_test(collision test_collision.cpp)
make_test(adj test_adj.cpp)
make_test(barycentric test_barycentric.cpp)
make_test(linetri_intersection test_linetri_intersection.cpp)
make_test(push_and_search test_push_and_search.cpp)
make_test(pseudoPushAndSearch pseudoPushAndSearch.cpp)
make_test(particleToMesh pseudoPtclToMesh.cpp)

include(testing.cmake)

bob_end_subdir()

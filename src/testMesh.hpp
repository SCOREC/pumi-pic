#ifndef TEST_MESH_HPP
#define TEST_MESH_HPP

#include "Omega_h_mesh.hpp"


class testMesh {
public:

  testMesh(Omega_h::Mesh &m):mesh(m){}

  Omega_h::Mesh &mesh;
};

#endif

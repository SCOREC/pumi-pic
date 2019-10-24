#include "pumipic_mesh.hpp"

namespace pumipic {
  Mesh::~Mesh() {
    delete picpart;
  }

  bool Mesh::isFullMesh() const {
    return is_full_mesh;
  }
}

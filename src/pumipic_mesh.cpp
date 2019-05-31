#include "pumipic_mesh.hpp"

namespace pumipic {
  Mesh::~Mesh() {
    delete picpart;
  }

  bool Mesh::isFullMesh() const {
    return num_cores[picpart->dim()] == picpart->comm()->size();
  }
}

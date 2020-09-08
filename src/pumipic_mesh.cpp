#include "pumipic_mesh.hpp"
#include "pumipic_lb.hpp"
namespace pumipic {
  Mesh::~Mesh() {
    if (!isFullMesh())
      delete picpart;
    if (ptcl_balancer)
      delete ptcl_balancer;
  }

  bool Mesh::isFullMesh() const {
    return is_full_mesh;
  }
}

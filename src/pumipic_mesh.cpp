#include "pumipic_mesh.hpp"
#include "pumipic_lb.hpp"
namespace pumipic {
  Mesh::Mesh() {
    picpart = NULL;
    is_full_mesh = false;
    ptcl_balancer = NULL;

    for (int i = 0; i < 4; ++i) {
      num_cores[i] = 0;
      num_bounds[i] = 0;
      num_boundaries[i] = 0;
    }
  }
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

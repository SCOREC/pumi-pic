#include "pumipic_mesh.hpp"
#include <Omega_h_for.hpp>
#include <Omega_h_tag.hpp>
#include <Omega_h_adj.hpp>
#include <Omega_h_array.hpp>
#include <Omega_h_mesh.hpp>
#include <Omega_h_build.hpp>
#include <Omega_h_reduce.hpp>

namespace pumipic {
  Mesh::~Mesh() {
    delete picpart;
    delete [] buffered_parts[3];
  }
}

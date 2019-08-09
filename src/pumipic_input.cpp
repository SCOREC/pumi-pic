#include "pumipic_input.hpp"

namespace pumipic {

  Input::Input(Omega_h::Mesh& mesh, Ownership rule, Omega_h::LOs partition_vector,
               Method bufferMethod_, Method safeMethod_) : m(mesh) {
    ownership_rule = rule;
    partition = partition_vector;
    bufferMethod = bufferMethod_;
    if (bufferMethod == NONE) {
      if (!mesh.comm()->rank())
        printf("[WARNING] bufferMethod given as NONE, setting to MINIMUM\n");
      bufferMethod=MINIMUM;
    }
    safeMethod = safeMethod_;

    bridge_dim = 0;
    bufferBFSLayers = 3;
    safeBFSLayers = 1;

    if (bufferMethod == MINIMUM)
      bufferBFSLayers = 0;
    if (safeMethod == MINIMUM)
      safeBFSLayers = 0;
  }
}

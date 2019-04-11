#pragma once

#include <Omega_h_mesh.hpp>

namespace pumipic {
  /*
    Creates PIC parts that contain the core region of the part and any part that is within 
    "buffer_layers" of the part boundary
    * \param full_mesh The full mesh loaded on each process
    * \param partition_vector The owner of each element of the mesh
    * \param safe_layers The number of safe layers required
    * \param buffer_layers The minimum number of buffer layers required
    * \requires safe_layers <= buffer_layers
    * \output picpart The mesh distributed in the format mentioned above
    * \param debug: 0 = silent (default)
    *               1 = basic stats
    *               2 = more output
    *               3 = vtk output
    * 
    * The picpart mesh will have two tags assigned to elements:
    *  "global_id": The global identifier that each process shares for each element
    *  "safe": A flag where 1 = safe element, 0 = unsafe element
  */
  void constructPICParts(Omega_h::Mesh& full_mesh, Omega_h::Write<Omega_h::LO>& partition_vector,
                         int safe_layers, int buffer_layers, Omega_h::Mesh* picpart, int debug = 0);

  //TODO create an XGC classification utilizing method to create picparts
}

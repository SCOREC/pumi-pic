#pragma once
#include "xgcm_types.hpp"

namespace xgcm {
  //Nonphysical elliptical push that pushes particles in an ellipse shape
  namespace ellipticalPush {
    /*
      ps - the particle structure
      h - x coordinate of ellipse center
      k - y coordinate of ellipse center
      d - ratio of ellipse minor axis length (a) to major axis length (b)
     */
    void setup(PS_I* ptcls, const double h_, const double k_, const double d_);
    /*
      ps - the particle structure
      m - the mesh
      deg - number of degrees to move around the ellipse
      iter - the iteration number
     */
    void push(PS_I* ptcls, Omega_h::Mesh& m, const double deg, const int iter);
  }
}

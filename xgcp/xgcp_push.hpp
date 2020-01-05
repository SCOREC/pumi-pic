#pragma once
#include "xgcp_types.hpp"

namespace xgcp {
  //Nonphysical elliptical push that pushes particles in an ellipse shape
  namespace ellipticalPush {
    /*
      scs - the particle structure
      h - x coordinate of ellipse center
      k - y coordinate of ellipse center
      d - ratio of ellipse minor axis length (a) to major axis length (b)
     */
    void setup(SCS_I* scs, const double h_, const double k_, const double d_);
    /*
      scs - the particle structure
      m - the mesh
      deg - number of degrees to move around the ellipse
      iter - the iteration number
     */
    void push(SCS_I* scs, Omega_h::Mesh& m, const double deg, const int iter);
  }
}

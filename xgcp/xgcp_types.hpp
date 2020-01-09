#pragma once

#include <SellCSigma.h>
#include <Omega_h_mesh.hpp>
#include <pumipic_mesh.hpp>


namespace xgcp {
  namespace o = Omega_h;
  namespace p = pumipic;
  namespace ps = particle_structs;
  //Floating Point type
  typedef double fp_t;

  //Vector of 3 floating point types
  typedef fp_t Vector3d[3];

  /*Electron Data Type
    -two fp_t[3] arrays, 'Vector3d', for the current and
      computed (pre adjacency search) positions, and
    -an integer to store the particles id
  */
  typedef ps::MemberTypes<Vector3d, Vector3d, int> Electron;

  /* Ion Data Type
     -two fp_t[3] arrays, 'Vector3d', for the current and
       computed (pre adjacency search) positions, and
     -an integer to store the particles id
     -a float to store the value of the constant 'b'
       that defines the ellipse
     -a float to store the angle of the particle in polar coordinates
  */
  typedef ps::MemberTypes<Vector3d, Vector3d, int, float, float> Ion;

  //Particle Data Structures for Electrons and Ions
  typedef ps::SellCSigma<Electron> SCS_E;
  typedef ps::SellCSigma<Ion> SCS_I;
}

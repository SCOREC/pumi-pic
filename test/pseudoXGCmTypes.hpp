#ifndef PSEUDO_XGCM_TYPES_H
#define PSEUDO_XGCM_TYPES_H
#include "pumipic_kktypes.hpp"
#include <psTypes.h>
#include <SellCSigma.h>
#include <Distribute.h>
#include <Kokkos_Core.hpp>

using particle_structs::fp_t;
using particle_structs::lid_t;
using particle_structs::Vector3d;
using particle_structs::SellCSigma;
using particle_structs::MemberTypes;
using particle_structs::distribute_particles;
using particle_structs::distribute_name;
using particle_structs::elemCoords;

namespace o = Omega_h;
namespace p = pumipic;

typedef MemberTypes<Vector3d, Vector3d, int> Point;
typedef SellCSigma<Point> SCSpt;

//To demonstrate push and adjacency search we store:
//-two fp_t[3] arrays, 'Vector3d', for the current and
// computed (pre adjacency search) positions, and
//-an integer to store the particles id
//-a float to store the value of the constant 'b' 
// that defines the ellipse
//-a float to store the angle of the particle in polar coordinates
typedef MemberTypes<Vector3d, Vector3d, int, float, float> Particle;
typedef SellCSigma<Particle> SCS;

#endif

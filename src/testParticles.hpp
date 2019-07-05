#ifndef TEST_PARTICLES_HPP
#define TEST_PARTICLES_HPP

#include "Omega_h_mesh.hpp"
#include "pumipic_kktypes.hpp"
#include <psTypes.h>
#include <SellCSigma.h>
#include <SCS_Macros.h>
#include <Distribute.h>
#include <Kokkos_Core.hpp>
#include "pumipic_library.hpp"

using particle_structs::fp_t;
using particle_structs::lid_t;
using particle_structs::Vector3d;
using particle_structs::SellCSigma;
using particle_structs::MemberTypes;

namespace o = Omega_h;
namespace p = pumipic;

// TODO: initialize these to its default values: ids =-1, reals=0
typedef MemberTypes < Vector3d, Vector3d, int,  Vector3d, int, int, Vector3d, 
       Vector3d, Vector3d> Particle;

// 'Particle' definition retrieval positions. 
enum {PTCL_POS_PREV, PTCL_POS, PTCL_ID, XPOINT, XPOINT_FACE, PTCL_BDRY_FACEID, 
     PTCL_BDRY_CLOSEPT, PTCL_EFIELD_PREV, PTCL_VEL};
typedef SellCSigma<Particle> SCS;

class testParticles {
public:
  testParticles(o::Mesh &m):mesh(m){}
  ~testParticles(){delete scs;}

  void testDefineParticles(int numPtcls, int elId);
  void testInitImpurityPtclsInADir(o::LO numPtcls);
    
  SCS* scs;
  o::Mesh &mesh;
};

#endif

#ifndef GITRM_PARTICLES_HPP
#define GITRM_PARTICLES_HPP

#include "pumipic_adjacency.hpp"
#include "pumipic_kktypes.hpp"

#include <psTypes.h>
#include <SellCSigma.h>
#include <SCS_Macros.h>
#include <Distribute.h>
#include <Kokkos_Core.hpp>
#include <chrono>
#include <thread>


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

// TODO: initialize these to its default values: ids =-1, reals=0
typedef MemberTypes < Vector3d, Vector3d, int,  int, Vector3d, 
       Vector3d, Vector3d> Particle;

// 'Particle' definition retrieval positions. 
enum {PCL_POS_PREV, PCL_POS, PCL_ID, PCL_BDRY_FACEID, PCL_BDRY_CLOSEPT, 
     PCL_EFIELD_PREV, PCL_VEL};

class GitrmParticles {
public:
  GitrmParticles(o::Mesh &m);
  ~GitrmParticles();
  GitrmParticles(GitrmParticles const&) = delete;
  void operator=(GitrmParticles const&) = delete;

  void defineParticles();

  SellCSigma<Particle>* scs;
  o::Mesh &mesh;
};


//HACK to avoid having an unguarded comma in the SCS PARALLEL macro
OMEGA_H_INLINE o::Matrix<3, 4> gatherVectors(o::Reals const& a, o::Few<o::LO, 4> v) {
  return o::gather_vectors<4, 3>(a, v);
}


inline void setPtclIds(SellCSigma<Particle>* scs) {
  fprintf(stderr, "%s\n", __func__);
  scs->transferToDevice();
  p::kkLidView pid_d("pid_d", scs->offsets[scs->num_slices]);
  p::hostToDeviceLid(pid_d, scs->getSCS<2>() );
  PS_PARALLEL_FOR_ELEMENTS(scs, thread, e, {
    (void)e;
    PS_PARALLEL_FOR_PARTICLES(scs, thread, pid, {
      pid_d(pid) = pid;
    });
  });
  p::deviceToHostLid(pid_d, scs->getSCS<2>() );
}

inline void setInitialPtclCoords(o::Mesh& mesh, SellCSigma<Particle>* scs) {
  //get centroid of parent element and set the child particle coordinates
  //most of this is copied from Omega_h_overlay.cpp get_cell_center_location
  //It isn't clear why the template parameter for gather_[verts|vectors] was
  //sized eight... maybe something to do with the 'Overlay'.  Given that there
  //are four vertices bounding a tet, I'm setting that parameter to four below.
  auto cells2nodes = mesh.get_adj(o::REGION, o::VERT).ab2b;
  auto nodes2coords = mesh.coords();
  //set particle positions and parent element ids
  scs->transferToDevice();
  p::kkFp3View x_scs_d("x_scs_d", scs->offsets[scs->num_slices]);
  p::hostToDeviceFp(x_scs_d, scs->getSCS<PCL_POS_PREV>() );

  //TODO
  p::kkFp3View x_scs_pos_d("x_scs_pos_d", scs->offsets[scs->num_slices]);
  p::hostToDeviceFp(x_scs_pos_d, scs->getSCS<PCL_POS>() );

  PS_PARALLEL_FOR_ELEMENTS(scs, thread, e, {
    auto cell_nodes2nodes = o::gather_verts<4>(cells2nodes, o::LO(e));
    auto cell_nodes2coords = gatherVectors(nodes2coords, cell_nodes2nodes);
    auto center = average(cell_nodes2coords);
    PS_PARALLEL_FOR_PARTICLES(scs, thread, pid, {
      if(e%500 == 0) 
        printf("elm %d xyz %f %f %f\n", e, center[0], center[1], center[2]);
      for(int i=0; i<3; i++) {
        x_scs_d(pid,i) = center[i];
        x_scs_pos_d(pid,i) = center[i];
      }
    });
  });
  p::deviceToHostFp(x_scs_d, scs->getSCS<PCL_POS_PREV>() );

  //TODO Temp for replacing a Boris move call ?
  p::deviceToHostFp(x_scs_pos_d, scs->getSCS<PCL_POS>() );
}

// spherical coordinates
//inline void GitrmParticles::initParticles(o::Real theta, o::Real phi){

//}

#endif //define


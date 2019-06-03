#ifndef GITRM_PARTICLES_HPP
#define GITRM_PARTICLES_HPP

#include "pumipic_adjacency.hpp"
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
using particle_structs::distribute_particles;
using particle_structs::distribute_name;
using particle_structs::elemCoords;

namespace o = Omega_h;
namespace p = pumipic;

// TODO: initialize these to its default values: ids =-1, reals=0
typedef MemberTypes < Vector3d, Vector3d, int,  int, Vector3d, 
       Vector3d, Vector3d, Vector3d, int> Particle;

// 'Particle' definition retrieval positions. 
enum {PTCL_POS_PREV, PTCL_POS, PTCL_ID, PTCL_BDRY_FACEID, PTCL_BDRY_CLOSEPT, 
     PTCL_EFIELD_PREV, PTCL_VEL, XPOINT, XPOINT_FACE};
typedef SellCSigma<Particle> SCS;

enum {BDRY_DATA_SIZE=7 }; //TODO

class GitrmParticles {
public:
  GitrmParticles(o::Mesh &m, const int np);
  ~GitrmParticles();
  GitrmParticles(GitrmParticles const&) = delete;
  void operator=(GitrmParticles const&) = delete;

  void defineParticles(const int elId, const int numPtcls=100);
  void findInitBdryElemId(o::Real theta, o::Real phi, const o::Real r,
     o::LO &initEl, o::Write<o::LO> &elemAndFace, 
     const o::LO maxLoops = 100, const o::Real outer=2);
  void setImpurityPtclInitCoords(o::Write<o::LO> &);
  void initImpurityPtcls(const o::LO numPtcls,o::Real theta, o::Real phi, 
    const o::Real r, const o::LO maxLoops = 100, const o::Real outer=2);
  
  SCS* scs;
  o::Mesh &mesh;
};

inline void setPtclIds(SCS* scs) {
  fprintf(stderr, "%s\n", __func__);
  auto pid_d = scs->get<2>();
  PS_PARALLEL_FOR_ELEMENTS(scs, thread, e, {
    (void)e;
    PS_PARALLEL_FOR_PARTICLES(scs, thread, pid, {
      pid_d(pid) = pid;
    });
  });
}


inline void applySurfaceModel(o::Mesh& mesh, SCS* scs, o::Write<o::LO>& elem_ids) {
  const auto coords = mesh.coords();
  const auto face_verts = mesh.ask_verts_of(2);
  const auto f2r_ptr = mesh.ask_up(o::FACE, o::REGION).a2ab;
  const auto f2r_elem = mesh.ask_up(o::FACE, o::REGION).ab2b;
  const auto side_is_exposed = mark_exposed_sides(&mesh);

  o::LO dof = BDRY_DATA_SIZE; //7
  //get mesh tag for boundary data id,xpt,vel
  auto xtag_d  = o::deep_copy(mesh.get_array<o::Real>(o::FACE, "bdryData"));
  
  auto ppos_d = scs->template get<PTCL_POS>();
  auto pxpt_d = scs->template get<XPOINT>();
  const auto xface_d = scs->template get<XPOINT_FACE>();
  auto pvel_d = scs->template get<PTCL_VEL>();
  auto lamb = SCS_LAMBDA(const int& e, const int& pid, const int& mask) {
    auto fid = xface_d(pid);

    if(fid >= 0) {
      OMEGA_H_CHECK(side_is_exposed[fid]);
      std::srand(std::time(nullptr));
      double rn = (double)std::rand()/RAND_MAX;
      auto vel = p::makeVector3(pid, pvel_d );
      auto xpt = p::makeVector3(pid, pxpt_d);
      auto fnv = p::get_face_normal(fid, coords, face_verts);

      auto angle = p::angle_between(fnv, vel);
      // reflection
      if(angle > o::PI/4.0 && rn >0.5) {
        // R = D- 2(D.N)N;
        auto rvel = vel - 2* p::osh_dot(vel, fnv) * fnv;
        pvel_d(pid, 0) = rvel[0];
        pvel_d(pid, 1) = rvel[1];
        pvel_d(pid, 2) = rvel[2];
        // TODO move it a boit inwards
        // Current position is that of xpoint
        ppos_d(pid, 0) = xpt[0];
        ppos_d(pid, 1) = xpt[1];
        ppos_d(pid, 2) = xpt[2];
        elem_ids[pid] = p::elem_of_bdry_face(fid, f2r_ptr, f2r_elem);
      } else {

     printf("\n****DOF=BDRY_DATA_SIZE: %d \n", dof);
        xtag_d[pid*dof] = static_cast<o::Real>(pid);
        for(int i=1; i<4; ++i) {
          xtag_d[pid*dof+i] = xpt[i];  
        }
        for(int i=5; i<7; ++i) {
          xtag_d[pid*dof+i] = vel[i];
        }
      }
    }
  };
  scs->parallel_for(lamb);
  o::HostWrite<o::Real> tag(xtag_d);
  mesh.set_tag(o::FACE, "bdryData", o::Reals(tag)); //TODO
}

inline void storeData(o::Mesh& mesh, SCS* scs, o::Write<o::Real> &data_d) {
  const auto coords = mesh.coords();
  const auto face_verts = mesh.ask_verts_of(2);
  const auto f2r_ptr = mesh.ask_up(o::FACE, o::REGION).a2ab;
  const auto f2r_elem = mesh.ask_up(o::FACE, o::REGION).ab2b;
  const auto side_is_exposed = mark_exposed_sides(&mesh);

  auto ppos_d = scs->template get<PTCL_POS>();
  auto pvel_d = scs->template get<PTCL_VEL>();

  auto lamb = SCS_LAMBDA(const int& e, const int& pid, const int& mask) {
    auto vel = p::makeVector3(pid, pvel_d);
    auto pos = p::makeVector3(pid, ppos_d);
/*
  // find location
    data_d[pid*dof] = static_cast<double>(pid);
    int s = 1;
    for(int i=s; i<s+3; ++i)
      data_d[pid*dof+i] = pos[i-s];
    s = 4;
    for(int i=s; i<s+3; ++i)
      data_d[pid*dof+i] = vel[i-s];
 */
  };
  scs->parallel_for(lamb);
}

#endif//define


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
       Vector3d, Vector3d> Particle;

// 'Particle' definition retrieval positions. 
enum {PTCL_POS_PREV, PTCL_POS, PTCL_ID, PTCL_BDRY_FACEID, PTCL_BDRY_CLOSEPT, 
     PTCL_EFIELD_PREV, PTCL_VEL};
typedef SellCSigma<Particle> SCS;

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

//TODO this update has to be in Boris Move ?, or kept track of inside it
inline void updatePtclPositions(SCS* scs) {
  auto x_scs_d = scs->get<0>();
  auto xtgt_scs_d = scs->get<1>();
  PS_PARALLEL_FOR_ELEMENTS(scs, thread, e, {
    (void)e;
    PS_PARALLEL_FOR_PARTICLES(scs, thread, pid, {
      x_scs_d(pid,0) = xtgt_scs_d(pid,0);
      x_scs_d(pid,1) = xtgt_scs_d(pid,1);
      x_scs_d(pid,2) = xtgt_scs_d(pid,2);
      xtgt_scs_d(pid,0) = 0;
      xtgt_scs_d(pid,1) = 0;
      xtgt_scs_d(pid,2) = 0;
    });
  });
}

inline void rebuild(SCS* scs, o::LOs elem_ids) {
  fprintf(stderr, "rebuild\n");
  updatePtclPositions(scs);
  const int scs_capacity = scs->capacity();
  auto printElmIds = SCS_LAMBDA(const int& e, const int& pid, const int& mask) {
    if(mask > 0)
      printf("elem_ids[%d] %d\n", pid, elem_ids[pid]);
  };
  scs->parallel_for(printElmIds);

  SCS::kkLidView scs_elem_ids("scs_elem_ids", scs_capacity);

  auto lamb = SCS_LAMBDA(const int& e, const int& pid, const int& mask) {
    (void)e;
    scs_elem_ids(pid) = elem_ids[pid];
  };
  scs->parallel_for(lamb);
  
  scs->rebuild(scs_elem_ids);
}

inline void search(o::Mesh& mesh, SCS* scs) {
  fprintf(stderr, "search\n");
  assert(scs->num_elems == mesh.nelems());
  Omega_h::LO maxLoops = 100;
  const auto scsCapacity = scs->capacity();
  o::Write<o::LO> elem_ids(scsCapacity,-1);
  bool isFound = p::search_mesh<Particle>(mesh, scs, elem_ids, maxLoops);
  assert(isFound);
  //rebuild the SCS to set the new element-to-particle lists
  rebuild(scs, elem_ids);
}


#endif //define


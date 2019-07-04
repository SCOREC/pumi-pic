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

namespace o = Omega_h;
namespace p = pumipic;

// TODO: initialize these to its default values: ids =-1, reals=0
typedef MemberTypes < Vector3d, Vector3d, int,  Vector3d, int, int, Vector3d, 
       Vector3d, Vector3d> Particle;

// 'Particle' definition retrieval positions. 
enum {PTCL_POS_PREV, PTCL_POS, PTCL_ID, XPOINT, XPOINT_FACE, PTCL_BDRY_FACEID, 
     PTCL_BDRY_CLOSEPT, PTCL_EFIELD_PREV, PTCL_VEL};
typedef SellCSigma<Particle> SCS;
struct PtclInitStruct;

class GitrmParticles {
public:
  GitrmParticles(o::Mesh &m);
  ~GitrmParticles();
  GitrmParticles(GitrmParticles const&) = delete;
  void operator=(GitrmParticles const&) = delete;

  void defineParticles(int numPtcls, o::LOs& ptclsInElem, int elId=-1);
  void findInitialBdryElemId(o::Real theta, o::Real phi, o::Real r,
    o::LO &initEl, o::Write<o::LO> &elemAndFace, 
    o::LO maxLoops=100, o::Real outer=2);
  void setImpurityPtclInitCoords(o::Write<o::LO> &);
  void initImpurityPtclsInADir(o::Real, o::LO numPtcls,o::Real theta, 
    o::Real phi, o::Real r, o::LO maxLoops = 100, o::Real outer=2);
  void setInitialTargetCoords(o::Real dTime);
  void initImpurityPtclsFromFile(const std::string& fName, 
    o::LO numPtcls, o::LO maxLoops);
  void processPtclInitFile(const std::string &fName,
    o::HostWrite<o::Real> &data, PtclInitStruct &ps);
  void findElemIdsOfPtclFileCoords(o::LO numPtcls, const o::Reals& data_r,
    o::Write<o::LO>& elemIds, o::Write<o::LO>& ptclsInElem, int maxLoops=100);
  void setImpurityPtclInitData(o::LO numPtcls, const o::Reals& data, 
    const o::LOs& ptclIdPtrsOfElem, const o::LOs& ptclIdsOfElem, 
    const o::LOs& elemIds, int maxLoops=100);
  void findElemIdsOfPtclFileCoordsByAdjSearch(o::LO numPtcls, 
    const o::Reals& data, o::Write<o::LO>& elemIds, o::LOs& numPtclsInElems);
  void convertInitPtclElemIdsToCSR(const o::LOs& numPtclsInElems,
    o::LOs& ptclIdPtrsInElem, o::LOs& ptclIdsOfElem, o::LOs& elemIds,
    o::LO numPtcls);
  SCS* scs;
  o::Mesh &mesh;
};


struct PtclInitStruct {
  PtclInitStruct(std::string n, std::string np, std::string x, std::string y,
    std::string z,std::string vx, std::string vy, std::string vz):
    name(n), nPname(np), xName(x), yName(y), zName(z), 
    vxName(vx), vyName(vy), vzName(vz) {}
  std::string name;
  std::string nPname;// "nP"
  std::string xName;
  std::string yName;
  std::string zName;
  std::string vxName;
  std::string vyName;
  std::string vzName;
  int nComp = 6;  //pos, vel
  int nP = 0;
};

/*
inline void setPtclIds(SCS* scs, bool verb=false) {
  if(verb)
    fprintf(stderr, " %s\n", __func__);
  auto pid_scs = scs->template get<2>();
  PS_PARALLEL_FOR_ELEMENTS(scs, thread, e, {
    (void)e;
    PS_PARALLEL_FOR_PARTICLES(scs, thread, pid, {
      pid_scs(pid) = pid;
    });
  });
}
*/

//call this before re-building, since mask of exiting ptcl removed from origin elem
inline void storeAndPrintData(o::Mesh& mesh, SCS* scs, o::LO iter, 
    o::Write<o::LO> &data_d, bool print=true) {
  const auto coords = mesh.coords();
  const auto face_verts = mesh.ask_verts_of(2);
  const auto f2r_ptr = mesh.ask_up(o::FACE, o::REGION).a2ab;
  const auto f2r_elem = mesh.ask_up(o::FACE, o::REGION).ab2b;
  const auto side_is_exposed = mark_exposed_sides(&mesh);
  
  auto pos_scs = scs->get<PTCL_POS>();
  auto ppos_scs = scs->get<PTCL_POS_PREV>();
  auto vel_scs = scs->get<PTCL_VEL>();
  auto pid_scs = scs->get<PTCL_ID>();
  auto xface_scs = scs->get<XPOINT_FACE>();
  auto xpt_scs = scs->get<XPOINT>();
  auto lamb = SCS_LAMBDA(const int& e, const int& pid, const int& mask) {
    if(mask >0) {
      auto vel = p::makeVector3(pid, vel_scs);
      auto pos = p::makeVector3(pid, pos_scs);

    // find grid index
      o::Real dr = 2.0; //TODO set same or pass from main
      o::Real r, theta, phi;
      p::cartesianToSpeherical(pos[0], pos[1], pos[2], r, theta, phi);
      o::LO ir = (o::LO) r/dr;
      data_d[ir] = data_d[ir] + 1;

      o::Vector<3> old = pos;
      o::LO intx = 0; 
      if(xface_scs(pid)>=0) {
        intx = 1;
        pos = p::makeVector3(pid, xpt_scs);
      }

      if(print)
        printf("ptclID %d iter %d pos %.5f %.5f %.5f prev %.5f %.5f %.5f actual %.5f %.5f %.5f exit %d\n", 
          pid_scs(pid), iter, pos[0], pos[1], pos[2],
          ppos_scs(pid,0), ppos_scs(pid,1), ppos_scs(pid,2),
          old[0], old[1], old[2], intx );
    }
  };
  scs->parallel_for(lamb);
}



/** @brief Calculate distance of particles to domain boundary 
 * TODO add description of data size of bdryFaces, bdryFaceInds and indexes
 */
inline void gitrm_findDistanceToBdry(  particle_structs::SellCSigma<Particle>* scs,
  o::Mesh &mesh,  const o::Reals &bdryFaces, const o::LOs &bdryFaceInds, 
  const o::LO fsize, const o::LO fskip) {
  const auto nel = mesh.nelems();
  const auto coords = mesh.coords();
  const auto mesh2verts = mesh.ask_elem_verts(); 
  //fskip is 2, since 1st 2 are not part of face vertices
  OMEGA_H_CHECK(fsize > 0 && nel >0);
  auto pos_d = scs->template get<PTCL_POS>();
  auto closestPoint_d = scs->template get<PTCL_BDRY_CLOSEPT>();
  auto bdry_face_d = scs->template get<PTCL_BDRY_FACEID>();
  auto distRun = SCS_LAMBDA(const int &elem, const int &pid,
                                const int &mask){ 
    if (mask > 0) {
    // elem 42200
      o::LO verbose = 1; //(elem%500==0)?2:1;

      o::LO beg = bdryFaceInds[elem];
      o::LO nFaces = bdryFaceInds[elem+1] - beg;
      if(nFaces >0) {
        o::Real dist = 0;
        o::Real min = 1e10;
        o::Matrix<3, 3> face;
        o::Vector<3> point = o::zero_vector<3>();
        o::Vector<3> pt;
        o::LO fe = -1;
        o::LO fel = -1;
        o::LO fi = -1;
        o::LO fid = -1;
        o::LO minRegion = -1;

        if(verbose >2)
          printf("\n e_%d nFaces_%d %d %d \n", elem, nFaces, bdryFaceInds[elem], 
            bdryFaceInds[elem+1]);

        for(o::LO ii = 0; ii < nFaces; ++ii) {
          // TODO put in a function
          o::LO ind = (beg + ii)*fsize;
          // fskip=2 for faceId, elId
          OMEGA_H_CHECK(fskip ==2);
          fi = static_cast<o::LO>(bdryFaces[ind]);
          fe = static_cast<o::LO>(bdryFaces[ind + 1]);
          for(o::LO i=0; i<3; ++i) { //Tet vertexes
            for(o::LO j=0; j<3; ++j) { //coords
              face[i][j] = bdryFaces[ind + i*3 + j + fskip];
            }
          }
          auto ref = p::makeVector3(pid, pos_d);
          if(verbose > 2 && ii == 0)
            printf("ref: %d %f %f %f \n", pid, ref[0], ref[1], ref[2]);
          //o::LO region = p::find_closest_point_on_triangle_with_normal(face, ref, point);
          o::LO region = p::find_closest_point_on_triangle(face, ref, pt); 
          dist = p::osh_dot(pt - ref, pt - ref);
          if(verbose >3)
            printf(": dist_%0.6f e_%d reg_%d fe_%d \n", dist, elem, region, fe);

          if(dist < min) {
            min = dist;
            fel = fe;
            fid = fi;
            minRegion = region;
            for(int i=0; i<3; ++i)
              point[0] = pt[i];

            if(verbose >2){
              printf("update:: e_%d dist_%0.6f region_%d fi_%d fe_%d\n", 
                elem, min, region, fi, fe);
              p::print_osh_vector(point, "Nearest_pt");
            }
          }
        }

        min = std::sqrt(min);
        if(verbose >1) {
          printf("  el=%d MINdist=%0.8f fid=%d face_el=%d reg=%d\n", 
            elem, min, fid, fel, minRegion);
        }
        OMEGA_H_CHECK(fid >= 0);
        bdry_face_d(pid) = fid;
        closestPoint_d(pid, 0) = point[0];
        closestPoint_d(pid, 1) = point[1];
        closestPoint_d(pid, 2) = point[2];   
      } 
    }
  };

  scs->parallel_for(distRun);

}

/*
//TODO this doesn't work !
inline void printGridData(o::Write<o::LO> &data_d) {
  //mask not 1 after rebuild at the ened of iterations in main
  for(o::LO i=0; i<data_d.size(); ++i)
      printf("ptclRad ind %d num %d\n", i, data_d(i));
}
*/
#endif//define


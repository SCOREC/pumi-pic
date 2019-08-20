#ifndef GITRM_PARTICLES_HPP
#define GITRM_PARTICLES_HPP

#include "GitrmMesh.hpp"
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

constexpr int PTCL_READIN_DATA_SIZE_PER_PTCL = 6;

// TODO: initialize these to its default values: ids =-1, reals=0
typedef MemberTypes < Vector3d, Vector3d, int,  Vector3d, Vector3d, 
   int, fp_t, int, fp_t, fp_t> Particle;

// 'Particle' definition retrieval positions. 
enum {PTCL_POS, PTCL_NEXT_POS, PTCL_ID, PTCL_VEL, PTCL_EFIELD, PTCL_CHARGE,
  PTCL_FIRST_IONIZEZ, PTCL_PREV_IONIZE, PTCL_FIRST_IONIZET, PTCL_PREV_RECOMBINE};

typedef SellCSigma<Particle> SCS;
struct PtclInitStruct;

class GitrmParticles {
public:
  GitrmParticles(o::Mesh &m, double dT);
  ~GitrmParticles();
  GitrmParticles(GitrmParticles const&) = delete;
  void operator=(GitrmParticles const&) = delete;

  void defineParticles(int numPtcls, o::LOs& ptclsInElem, int elId=-1);
  void findInitialBdryElemIdInADir(o::Real theta, o::Real phi, o::Real r,
    o::LO &initEl, o::Write<o::LO> &elemAndFace, 
    o::LO maxLoops=100, o::Real outer=2);
  void setImpurityPtclInitRndDistribution(o::Write<o::LO> &);
  void initImpurityPtclsInADir(o::LO numPtcls,o::Real theta, 
    o::Real phi, o::Real r, o::LO maxLoops = 100, o::Real outer=2);
  void setInitialTargetCoords(o::Real dTime);
  void initImpurityPtclsFromFile(const std::string& fName, 
    o::LO numPtcls=0, o::LO maxLoops=100, bool print=false);
  void processPtclInitFile(const std::string &fName,
    o::HostWrite<o::Real> &data, PtclInitStruct &ps, o::LO& numPtcls);
  void findElemIdsOfPtclFileCoords(o::LO numPtcls, const o::Reals& data_r,
    o::Write<o::LO>& elemIds, o::Write<o::LO>& ptclsInElem, int maxLoops=100);
  void setImpurityPtclInitData(o::LO numPtcls, const o::Reals& data, 
    const o::LOs& ptclIdPtrsOfElem, const o::LOs& ptclIdsOfElem, 
    const o::LOs& elemIds, int maxLoops=100);
  void findElemIdsOfPtclFileCoordsByAdjSearch(o::LO numPtcls, 
    const o::Reals& data_r, o::LOs& elemIdOfPtcls, o::LOs& numPtclsInElems);
  void convertInitPtclElemIdsToCSR(const o::LOs& numPtclsInElems,
    o::LOs& ptclIdPtrsOfElem, o::LOs& ptclIdsOfElem, o::LOs& elemIds,
    o::LO numPtcls);
  void printPtclSource(o::Reals& data, int nPtcls=0, int dof=6);

  o::Real timeStep;
  SCS* scs;
  o::Mesh &mesh;

  // particle dist to bdry
  o::Reals closestPoints;
  o::LOs closestBdryFaceIds;
  // wall collision
  o::Reals collisionPoints;
  o::LOs collisionPointFaceIds;
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

//TODO dimensions set for pisces to be removed
//call this before re-building, since mask of exiting ptcl removed from origin elem
inline void storePiscesData(o::Mesh& mesh, GitrmParticles& gp, o::LO iter, 
    o::Write<o::LO> &data_d, bool debug=true) {
  // test TODO move test part to separate unit test
  double radMax = 0.05; //m 0.0446+0.005
  double zMin = 0; //m height min
  double zMax = 0.15; //m height max 0.14275
  double htBead1 =  0.01275; //m ht of 1st bead
  double dz = 0.01; //m ht of beads 2..14
  SCS* scs = gp.scs;
  auto pisces_ids = mesh.get_array<o::LO>(o::FACE, "piscesTiRod_ind");
  auto pid_scs = scs->get<PTCL_ID>();
  const auto& xpoints = gp.collisionPoints;
  const auto& xpointFids = gp.collisionPointFaceIds;

  auto lamb = SCS_LAMBDA(const int& e, const int& pid, const int& mask) {
    auto fid = xpointFids[pid];
    if(mask >0 && fid>=0) {
      // test
      o::Vector<3> xpt;
      for(o::LO i=0; i<3; ++i)
        xpt[i] = xpoints[pid*3+i];
      auto x = xpt[0], y = xpt[1], z = xpt[2];
      o::Real rad = std::sqrt(x*x + y*y);
      o::LO zInd = -1;
      if(rad < radMax && z <= zMax && z >= zMin)
        zInd = (z > htBead1) ? (1+(o::LO)((z-htBead1)/dz)) : 0;
      
      auto detId = pisces_ids[fid];
      if(detId > -1) { //TODO
        if(debug)
          printf("ptclID %d zInd %d detId %d pos %.5f %.5f %.5f iter %d\n", 
            pid_scs(pid), zInd, detId, x, y, z, iter);
        Kokkos::atomic_fetch_add(&data_d[detId], 1);
      }
    }
  };
  scs->parallel_for(lamb);
}

//storePtclDataInGridsRZ(scs, iter, data_d, 1, 10, true);
// gridsR gets preference. If >1 then gridsZ not taken
inline void storePtclDataInGridsRZ(SCS* scs, o::LO iter, o::Write<o::GO> &data_d, 
  int gridsR=1, int gridsZ=10, bool debug=false, double radMax=0.2, 
  double zMax=0.8, double radMin=0, double zMin=0) {
  auto dz = (zMax - zMin)/gridsZ;
  auto dr = (radMax - radMin)/gridsR;
  auto pid_scs = scs->get<PTCL_ID>();
  auto tgt_scs = scs->get<PTCL_NEXT_POS>();
  auto lamb = SCS_LAMBDA(const int& e, const int& pid, const int& mask) {
    if(mask >0) {
      auto x = tgt_scs(pid, 0);
      auto y = tgt_scs(pid, 1);
      auto z = tgt_scs(pid, 2);
      auto rad = std::sqrt(x*x + y*y);
      int ind = -1;
      if(rad < radMax && radMin >= radMin && z <= zMax && z >= zMin)
        if(gridsR >1) //prefer radial
          ind = (int)((rad - radMin)/dr);
        else if(gridsZ >1)
          ind = (int)((z - zMin)/dz);
      int dir = (gridsR >1)?1:0;
      if(ind >=0) {
        if(debug)
          printf("grid_ptclID %d ind %d pos %.5f %.5f %.5f iter %d rdir %d\n", 
            pid_scs(pid), ind, x, y, z, iter, dir);
        Kokkos::atomic_fetch_add(&data_d[ind], 1);
      }
    } //mask
  };
  scs->parallel_for(lamb);
}

// print from host, since missing entries from device print  
inline void printGridDataNComp(o::Write<o::GO> &data_d, int nComp=1) {
  o::HostRead<o::GO> data(data_d);
  auto maxInd = data.size()/nComp;
  int sum = 0;
  for(int i=0; i<data.size(); ++i)
    sum += data[i];
  printf("index  fraction1 ...  \n");  
  for(int i=0; i<maxInd; ++i) {
    printf("%d ", i);
    for(int j=0; j<nComp; ++j)
      printf("%.3f ", data[i*nComp+j]/sum);
    printf("\n");
  }
}

template <typename T>
inline void printGridData(o::Write<T> &data_d) {
  o::Write<T>total(1,0);
  o::parallel_for(data_d.size(), OMEGA_H_LAMBDA(const int& i) {
    auto num = data_d[i];
    if(num>0)
      Kokkos::atomic_fetch_add(&total[0], num);
  });
  o::HostRead<T> tot(total);
  printf("total in device %d \n", tot[0]);
  o::HostRead<T> data(data_d);
  printf("index total  \n");  
  for(int i=0; i<data.size(); ++i)
      printf("%d %d\n", i, data[i]);
}


/** @brief Calculate distance of particles to domain boundary 
 * TODO add description of data size of bdryFac       es, bdryFaceInds and indexes
 */
inline void gitrm_findDistanceToBdry(GitrmParticles& gp,
  const GitrmMesh &gm, bool debug = false) {
  //o::Mesh &mesh,  const o::Reals &bdryFaces, const o::LOs &bdryFaceInds, 
  //const o::LO fsize, const o::LO fskip, bool debug = false) {
  auto* scs = gp.scs;
  o::Mesh &mesh = gm.mesh;  
  const auto nel = mesh.nelems();
  const auto coords = mesh.coords();
  const auto mesh2verts = mesh.ask_elem_verts(); 

  const o::Reals& bdryFaces = gm.bdryFaces;
  const o::LOs& bdryFaceInds = gm.bdryFaceInds;
  const auto scsCapacity = scs->capacity();
  o::Write<o::Real> closestPoints(scsCapacity*3, 0, "closest_points");
  o::Write<o::LO> closestBdryFaceIds(scsCapacity, -1, "closest_fids");
  o::LO fsize = SIZE_PER_FACE;
  o::LO fskip = FSKIP;
  //fskip is 2, since 1st 2 are not part of face vertices
  OMEGA_H_CHECK(fsize > 0 && nel >0);
  auto pos_d = scs->template get<PTCL_NEXT_POS>();

  auto distRun = SCS_LAMBDA(const int &elem, const int &pid,
                                const int &mask){ 
    if (mask > 0) {
      o::LO beg = bdryFaceInds[elem];
      o::LO nFaces = bdryFaceInds[elem+1] - beg;
      if(nFaces >0) {
        o::Real dist = 0;
        o::Real min = 1.0e10;
        o::Matrix<3, 3> face;
        auto point = o::zero_vector<3>();
        auto pt = o::zero_vector<3>();
        o::LO fe, fel, fi, fid, minRegion;
        fe = fi = fel = fid = minRegion = -1;
        if(debug && pid<1)
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
          if(debug && ii == 0)
            printf("pos: %d %g %g %g \n", pid, ref[0], ref[1], ref[2]);
          //o::LO region = p::find_closest_point_on_triangle_with_normal(face, ref, point);
          o::LO region = p::find_closest_point_on_triangle(face, ref, pt); 
          dist = p::osh_dot(pt - ref, pt - ref);
          dist = sqrt(dist); // use square ?
          if(debug) {
            printf(": dist_%g e_%d reg_%d fe_%d \n", dist, elem, region, fe);
            p::print_osh_vector(pt, "closest_pt");
          }
          if(ii==0 || dist < min) {
            min = dist;
            fel = fe;
            fid = fi;
            minRegion = region;
            for(int i=0; i<3; ++i)
              point[i] = pt[i];

            if(debug){
              printf("update:: e_%d dist_%g region_%d fi_%d fe_%d\n", 
                elem, min, region, fi, fe);
              p::print_osh_vector(point, "this_nearest_pt");
            }
          }
        } //for

        if(debug) {
          printf("dist: el=%d MINdist=%g fid=%d face_el=%d reg=%d\n", 
            elem, min, fid, fel, minRegion);
          p::print_osh_vector(point, "Nearest_pt");

        }
        OMEGA_H_CHECK(fid >= 0);
        closestBdryFaceIds[pid] = fid;
        for(o::LO j=0; j<3; ++j)
          closestPoints[pid*3+j] = point[j];
      } 
    }
  };

  scs->parallel_for(distRun);
  gp.closestPoints = o::Reals(closestPoints);
  gp.closestBdryFaceIds = o::LOs(closestBdryFaceIds);
}
#endif//define


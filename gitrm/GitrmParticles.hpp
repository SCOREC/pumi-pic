#ifndef GITRM_PARTICLES_HPP
#define GITRM_PARTICLES_HPP

#include <fstream>
#include "GitrmMesh.hpp"
#include <netcdf>
#include <Kokkos_Core.hpp>
#include <particle_structs.hpp>
//#include <PS_Types.h>
//#include "pumipic_kktypes.hpp"
//#include "pumipic_library.hpp"
#include "pumipic_mesh.hpp"
//#include <psTypes.h>
//#include <SellCSigma.h>
//#include <SCS_Macros.h>
//#include <Distribute.h>

using pumipic::fp_t;
using pumipic::Vector3d;
using particle_structs::lid_t;
using particle_structs::SellCSigma;
using particle_structs::MemberTypes;

namespace o = Omega_h;
namespace p = pumipic;
namespace ps = particle_structs;

namespace gitrm {
const o::Real ELECTRON_CHARGE = 1.60217662e-19;
const o::Real PROTON_MASS = 1.6737236e-27;
const o::Real BACKGROUND_AMU = 4.0; //for pisces
const o::Real PTCL_AMU=184.0; //W,tungsten
const o::LO PARTICLE_Z = 74;
}

typedef MemberTypes < Vector3d, Vector3d, int,  Vector3d, Vector3d, 
   int, fp_t, fp_t, int, fp_t, fp_t, int, fp_t> Particle;

// 'Particle' definition retrieval indices. 
enum {PTCL_POS, PTCL_NEXT_POS, PTCL_ID, PTCL_VEL, PTCL_EFIELD, PTCL_CHARGE,
 PTCL_WEIGHT, PTCL_FIRST_IONIZEZ, PTCL_PREV_IONIZE, PTCL_FIRST_IONIZET, 
 PTCL_PREV_RECOMBINE, PTCL_HIT_NUM, PTCL_VMAG_NEW};

typedef ps::ParticleStructure<Particle> PS;

class GitrmParticles {
public:
  GitrmParticles(o::Mesh& m, double dT);
  ~GitrmParticles();
  GitrmParticles(GitrmParticles const&) = delete;
  void operator=(GitrmParticles const&) = delete;

  void defineParticles(p::Mesh& picparts, int numPtcls, o::LOs& ptclsInElem, 
    int elId=-1);
  
  void initPtclCollisionData(int numPtcls); 
  
  void findInitialBdryElemIdInADir(o::Real theta, o::Real phi, o::Real r,
    o::LO &initEl, o::Write<o::LO> &elemAndFace, 
    o::LO maxLoops=100, o::Real outer=2);
  
  void setPtclInitRndDistribution(o::Write<o::LO> &);
  
  void initPtclsInADirection(p::Mesh& picparts, o::LO numPtcls,o::Real theta, 
    o::Real phi, o::Real r, o::LO maxLoops = 100, o::Real outer=2);
  
  void setInitialTargetCoords(o::Real dTime);
  
  void initPtclsFromFile(p::Mesh& picparts, const std::string& fName, 
    o::LO& numPtcls, o::LO maxLoops=100, bool print=false);
  
  void initPtclChargeIoniRecombData();
  void initPtclSurfaceModelData();

  void setPidsOfPtclsLoadedFromFile(const o::LOs& ptclIdPtrsOfElem,
    const o::LOs& ptclIdsInElem,  const o::LOs& elemIdOfPtcls, 
    const o::LO numPtcls, const o::LO nel);
  
  void setPtclInitData(const o::Reals& data, int nPtclsRead);
  
  void findElemIdsOfPtclFileCoordsByAdjSearch(const o::Reals& data, 
    o::LOs& elemIdOfPtcls, o::LOs& numPtclsInElems, o::LO numPtcls, 
    o::LO numPtclsRead);
  
  void convertInitPtclElemIdsToCSR(const o::LOs& numPtclsInElems,
    o::LOs& ptclIdPtrsOfElem, o::LOs& ptclIdsOfElem, o::LOs& elemIds,
    o::LO numPtcls);
  
  int readGITRPtclStepDataNcFile(const std::string& ncFileName, 
  int& maxNPtcls, int& numPtclsRead, bool debug=false);

  o::Real timeStep;
  PS* ptcls;
  o::Mesh& mesh;

  // particle dist to bdry
  o::Reals closestPoints;
  o::LOs closestBdryFaceIds;
  
  // wall collision; changed to per ptcl from per pid
  o::Write<o::Real> collisionPoints;
  o::Write<o::LO> collisionPointFaceIds;

  // test GITR step data
  o::Reals testGitrPtclStepData;
  int testGitrDataIoniRandInd = -1;
  int testGitrDataRecRandInd = -1;
  int testGitrStepDataDof = -1;
  int testGitrStepDataNumTsteps = -1;
  int testGitrStepDataNumPtcls = -1;
  int testGitrCollisionRndn1Ind = -1;
  int testGitrCollisionRndn2Ind = -1;
  int testGitrCollisionRndxsiInd = -1;
  int testGitrCrossFieldDiffRndInd = -1;
  int testGitrReflectionRndInd = -1;
  int testGitrOptIoniRec = 0;
  int testGitrOptDiffusion = 0;
  int testGitrOptCollision = 0;
  int testGitrOptSurfaceModel = 0;
};

//timestep +1
extern int iTimePlusOne;
// TODO move to class, data is kept as members  
void updatePtclStepData(PS* ptcls, o::Write<o::Real>& ptclStepData,
  o::Write<o::LO>& lastFilledTimeSteps, int nP, int dof, int histStep=1); 

void printPtclSource(o::Reals& data, int nPtcls, int nPtclsRead);

void printStepData(std::ofstream& ofsHistory, PS* ptcls, int iter,
  int numPtcls, o::Write<o::Real>& ptclsDataAll, o::Write<o::LO>& lastFilledTimeSteps, 
  o::Write<o::Real>& data, int dof=8, bool accum = false);

void writePtclStepHistoryFile(o::Write<o::Real>& ptclsHistoryData, 
  o::Write<o::LO>& lastFilledTimeSteps, int numPtcls, int dof, 
  int nTHistory, std::string outNcFileName);


/** @brief Calculate distance of particles to domain boundary. 
 * Not yet clear if a pre-determined depth can be used
*/
inline void gitrm_findDistanceToBdry(GitrmParticles& gp,
  const GitrmMesh& gm, int debug=0) {
  int tstep = iTimePlusOne;
  auto* ptcls = gp.ptcls;
  o::Mesh& mesh = gm.mesh;  
  o::LOs modelIdsToSkip = o::LOs(gm.modelIdsToSkipFromD2bdry);
  auto numModelIds = modelIdsToSkip.size();
  auto faceClassIds = mesh.get_array<o::ClassId>(2, "class_id");
  const auto coords = mesh.coords();
  const auto dual_elems = mesh.ask_dual().ab2b;
  const auto dual_faces = mesh.ask_dual().a2ab;
  const auto down_r2f = mesh.ask_down(3, 2).ab2b;
  const auto down_r2fs = mesh.ask_down(3, 2).ab2b;
  
  const int useReadInCsr = USE_READIN_CSR_BDRYFACES;
  const auto& bdryCsrReadInDataPtrs = gm.bdryCsrReadInDataPtrs;
  const auto& bdryCsrReadInData = gm.bdryCsrReadInData;

  const auto nel = mesh.nelems();
  const auto& f2rPtr = mesh.ask_up(o::FACE, o::REGION).a2ab;
  const auto& f2rElem = mesh.ask_up(o::FACE, o::REGION).ab2b;
  const auto& face_verts = mesh.ask_verts_of(2);
  const auto& bdryFaces = gm.bdryFacesSelectedCsr;
  const auto& bdryFacePtrs = gm.bdryFacePtrsSelected;
  const auto psCapacity = ptcls->capacity();
  o::Write<o::Real> closestPoints(psCapacity*3, 0, "closest_points");
  o::Write<o::LO> closestBdryFaceIds(psCapacity, -1, "closest_fids");
  auto pos_d = ptcls->get<PTCL_POS>();
  auto pid_ps = ptcls->get<PTCL_ID>();
  auto lambda = PS_LAMBDA(const int &elem, const int &pid, const int &mask) {
    if (mask > 0) {
      o::LO beg = 0;
      o::LO nFaces = 0;
      if(useReadInCsr) { //fix crash
        beg = bdryCsrReadInDataPtrs[elem];
        nFaces = bdryCsrReadInDataPtrs[elem+1] - beg;
      } else {
        beg = bdryFacePtrs[elem];
        nFaces = bdryFacePtrs[elem+1] - beg;
      }

      if(nFaces >0) {
        auto ptcl = pid_ps(pid);
        double dist = 0;
        double min = 1.0e+30;
        auto point = o::zero_vector<3>();
        auto pt = o::zero_vector<3>();
        o::Int bfid = -1, fid = -1, minRegion = -1;
        auto ref = p::makeVector3(pid, pos_d);
        o::Matrix<3,3> face;
        for(o::LO ii = 0; ii < nFaces; ++ii) {
          auto ind = beg + ii;
          if(useReadInCsr)
            bfid = bdryCsrReadInData[ind];
          else
            bfid = bdryFaces[ind];
          face = p::get_face_coords_of_tet(face_verts, coords, bfid);
          if(debug > 2) {
            auto bfeId = p::elem_id_of_bdry_face_of_tet(bfid, f2rPtr, f2rElem);
            printf(" ptcl %d elem %d d2bdry %g bfid %d bdry-el %d pos: %g %g %g bdry-face: "
              "%g %g %g : %g %g %g : %g %g %g \n", ptcl, elem, dist, bfid, bfeId, 
              ref[0], ref[1], ref[2], face[0][0], face[0][1], face[0][2], face[1][0], 
              face[1][1], face[1][2], face[2][0], face[2][1], face[2][2]);
          }
          int region;
          auto pt = p::closest_point_on_triangle(face, ref, &region); 
          dist = o::norm(pt - ref);
          if(dist < min) {
            min = dist;
            fid = bfid;
            minRegion = region;
            for(int i=0; i<3; ++i)
              point[i] = pt[i];
          }
        } //for nFaces

        if(debug) {
          auto fel = p::elem_id_of_bdry_face_of_tet(fid, f2rPtr, f2rElem);
          auto f = p::get_face_coords_of_tet(face_verts, coords, fid);
          printf("dist: ptcl %d tstep %d el %d MINdist %g nFaces %d fid %d " 
            "face_el %d reg %d pos %g %g %g nearest_pt %g %g %g "
            " face %g %g %g %g %g %g %g %g %g \n", 
            ptcl, tstep, elem, min, nFaces, fid, fel, minRegion, ref[0], 
            ref[1], ref[2], point[0], point[1], point[2], f[0][0], f[0][1],
            f[0][2], f[1][0],f[1][1], f[1][2],f[2][0], f[2][1],f[2][2]);
        }
        OMEGA_H_CHECK(fid >= 0);
        closestBdryFaceIds[pid] = fid;
        for(o::LO j=0; j<3; ++j)
          closestPoints[pid*3+j] = point[j];
      } //if nFaces 
    }
  };
  ps::parallel_for(ptcls, lambda);
  gp.closestPoints = o::Reals(closestPoints);
  gp.closestBdryFaceIds = o::LOs(closestBdryFaceIds);
}


//TODO dimensions set for pisces to be removed
//call this before re-building, since mask of exiting ptcl removed from origin elem
inline void storePiscesData(o::Mesh* mesh, GitrmParticles& gp, 
    o::Write<o::LO>& data_d, o::LO iter, bool resetFids=true, bool debug=true) {
  // test TODO move test part to separate unit test
  double radMax = 0.05; //m 0.0446+0.005
  double zMin = 0; //m height min
  double zMax = 0.15; //m height max 0.14275
  double htBead1 =  0.01275; //m ht of 1st bead
  double dz = 0.01; //m ht of beads 2..14
  PS* ptcls = gp.ptcls;
  auto pisces_ids = mesh->get_array<o::LO>(o::FACE, "piscesBeadCylinder_inds");
  auto pid_ps = ptcls->get<PTCL_ID>();
  //based on ptcl id or ptcls pid ?
  const auto& xpoints = gp.collisionPoints;
  auto& xpointFids = gp.collisionPointFaceIds;

  auto lamb = PS_LAMBDA(const int& e, const int& pid, const int& mask) {
    if(mask >0) {
      auto ptcl = pid_ps(pid);
      auto fid = xpointFids[ptcl];
      if(fid>=0) {
        xpointFids[ptcl] = -1;
        // test
        auto xpt = o::zero_vector<3>();
        for(o::LO i=0; i<3; ++i)
          xpt[i] = xpoints[ptcl*3+i]; //ptcl = 0..numPtcls
        auto x = xpt[0], y = xpt[1], z = xpt[2];
        o::Real rad = sqrt(x*x + y*y);
        o::LO zInd = -1;
        if(rad < radMax && z <= zMax && z >= zMin)
          zInd = (z > htBead1) ? (1+(o::LO)((z-htBead1)/dz)) : 0;
        
        auto detId = pisces_ids[fid];
        if(detId > -1) { //TODO
          if(debug)
            printf("ptclID %d zInd %d detId %d pos %.5f %.5f %.5f iter %d\n", 
              pid_ps(pid), zInd, detId, x, y, z, iter);
          Kokkos::atomic_increment(&data_d[detId]);
        }
      }
    }
  };
  ps::parallel_for(ptcls, lamb, "StorePiscesData");
}

//storePtclDataInGridsRZ(ptcls, iter, data_d, 1, 10, true);
// gridsR gets preference. If >1 then gridsZ not taken
inline void storePtclDataInGridsRZ(PS* ptcls, o::LO iter, o::Write<o::GO> &data_d, 
  int gridsR=1, int gridsZ=10, bool debug=false, double radMax=0.2, 
  double zMax=0.8, double radMin=0, double zMin=0) {
  auto dz = (zMax - zMin)/gridsZ;
  auto dr = (radMax - radMin)/gridsR;
  auto pid_ps = ptcls->get<PTCL_ID>();
  auto tgt_ps = ptcls->get<PTCL_NEXT_POS>();
  auto lamb = PS_LAMBDA(const int& e, const int& pid, const int& mask) {
    if(mask >0) {
      auto x = tgt_ps(pid, 0);
      auto y = tgt_ps(pid, 1);
      auto z = tgt_ps(pid, 2);
      auto rad = sqrt(x*x + y*y);
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
            pid_ps(pid), ind, x, y, z, iter, dir);
        Kokkos::atomic_increment(&data_d[ind]);
      }
    } //mask
  };
  ps::parallel_for(ptcls, lamb);
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
inline void printGridData(o::Write<T> &data_d, std::string fname="", 
    std::string header="") {
  if(fname=="")
    fname = "result.txt";
  std::ofstream outf(fname);
  auto total = 0;
  Kokkos::parallel_reduce(data_d.size(), OMEGA_H_LAMBDA(const int i, o::LO& lsum) {
    lsum += data_d[i];
  }, total);
  outf << header << "\n";
  outf << "total collected " <<  total << "\n";
  o::HostRead<T> data(data_d);
  outf << "index   total\n";  
  for(int i=0; i<data.size(); ++i)
    outf <<  i << " " << data[i] << "\n";
}

inline void printPtclHostData(o::HostWrite<o::Real>& dh, std::ofstream& ofsHistory, 
  int num, int dof, const char* name, int iter=-1) { 
  double pos[3], vel[3];
  for(int i=0; i<num; ++i) {
    for(int j=0; j<3; ++j) {
      pos[j] = dh[i*dof+j];
      vel[j] = dh[i*dof+3+j];
    }
    int id = static_cast<int>(dh[i*dof + 6]);
    int it = static_cast<int>(dh[i*dof + 7]);
    ofsHistory << name << " " << id+1 << " iter " << iter 
      << " pos " << pos[0] << " " << pos[1] << " " << pos[2]
      << " vel " << vel[0] << " " << vel[1] << " " << vel[2] 
      << " updateiter " << it << "\n";
  }
} 
#endif//define


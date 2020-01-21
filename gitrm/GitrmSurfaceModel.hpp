#ifndef GITRM_SURFACE_MODEL_HPP
#define GITRM_SURFACE_MODEL_HPP

#include "pumipic_adjacency.hpp"
#include "GitrmParticles.hpp" 
#include "GitrmMesh.hpp"

namespace o = Omega_h;
namespace p = pumipic;

namespace gitrm {
//TODO get from config
const int SURFACE_FLUX_EA = 1;
const int BOUNDARY_ATOM_Z = 1;
}

class GitrmSurfaceModel {
public:
  GitrmSurfaceModel(GitrmMesh& gm, std::string ncFile);
  void getConfigData(std::string ncFile);
  void initSurfaceModelData(std::string ncFile, bool debug=false);
  void prepareSurfaceModelData();

  void getSurfaceModelDataFromFile(const std::string,
   const std::vector<std::string>&, const std::vector<std::string>&,
   const std::vector<std::vector<int>>&, std::vector<int>&, std::vector<o::Reals>&);
  GitrmMesh& gm;
  o::Mesh& mesh;
  std::string ncFile;
  int numDetectorSurfaceFaces = 0;

  //from NC file
  std::string fileString{};
  std::string nEnSputtRefCoeffStr{};
  std::string nAngSputtRefCoeffStr{};
  std::string nEnSputtRefDistInStr{};
  std::string nAngSputtRefDistInStr{};
  std::string nEnSputtRefDistOutStr{};
  std::string nEnSputtRefDistOutRefStr{};
  std::string nAngSputtRefDistOutStr{};
  std::string enSputtRefCoeffStr{};
  std::string angSputtRefCoeffStr{};
  std::string enSputtRefDistInStr{};
  std::string angSputtRefDistInStr{};
  std::string enSputtRefDistOutStr{};
  std::string enSputtRefDistOutRefStr{};
  std::string angPhiSputtRefDistOutStr{};
  std::string angThetaSputtRefDistOutStr{};
  std::string sputtYldStr{};
  std::string reflYldStr{};
  std::string enDistYStr{};
  std::string angPhiDistYStr{};
  std::string angThetaDistYStr{};
  std::string enDistRStr{};
  std::string angPhiDistRStr{};
  std::string angThetaDistRStr{};

  int nEnSputtRefCoeff = 0; //"nE"
  int nAngSputtRefCoeff = 0; // "nA"
  int nEnSputtRefDistIn = 0; //"nE"
  int nAngSputtRefDistIn = 0; // "nA"
  int nEnSputtRefDistOut = 0; //"nEdistBins"
  int nEnSputtRefDistOutRef = 0; //"nEdistBinsRef"
  int nAngSputtRefDistOut = 0; // "nAdistBins"
  // shapes are given as comments
  o::Reals enSputtRefCoeff; // nEnSputtRefCoeff
  o::Reals angSputtRefCoeff; // nAngSputtRefCoeff
  o::Reals enSputtRefDistIn; //nEnSputtRefDistIn
  o::Reals angSputtRefDistIn; //nAngSputtRefDistIn
  o::Reals enSputtRefDistOut; // nEnSputtRefDistOut
  o::Reals enSputtRefDistOutRef; //nEnSputtRefDistOutRef
  o::Reals angPhiSputtRefDistOut; //nAngSputtRefDistOut
  o::Reals angThetaSputtRefDistOut; //nAngSputtRefDistOut
  o::Reals sputtYld; // nEnSputtRefCoeff X nAngSputtRefCoeff
  o::Reals reflYld; //  nEnSputtRefCoeff X nAngSputtRefCoeff
  o::Reals enDist_Y ; // nEnSputtRefCoeff X nAngSputtRefCoeff X nEnSputtRefDistOut
  o::Reals enDist_R; // nEnSputtRefCoeff X nAngSputtRefCoeff X nEnSputtRefDistOutRef
  o::Reals angPhiDist_Y; //nEnSputtRefCoeff X nAngSputtRefCoeff X nAngSputtRefDistOut
  o::Reals angThetaDist_Y; // ""
  o::Reals angPhiDist_R; // ""
  o::Reals angThetaDist_R; // ""
  
  o::Reals enLogSputtRefCoef; // nEnSputtRefCoeff
  o::Reals enLogSputtRefDistIn; //nEnSputtRefDistIn
  o::Reals energyDistGrid01; //nEnSputtRefDistOut
  o::Reals energyDistGrid01Ref; //nEnSputtRefDistOutRef
  o::Reals angleDistGrid01; //nAngSputtRefDistOut

  o::Reals enDist_CDF_Y_regrid; //EDist_CDF_Y_regrid(nDistE_surfaceModel)
  o::Reals angPhiDist_CDF_Y_regrid;  //AphiDist_CDF_R_regrid(nDistA_surfaceModel)
  o::Reals angPhiDist_CDF_R_regrid;  //APhiDist_CDF_R_regrid(nDistA_surfaceModel)
  o::Reals enDist_CDF_R_regrid;  //EDist_CDF_R_regrid(nDistE_surfaceModelRef)
  o::Reals angThetaDist_CDF_R_regrid; //
  o::Reals angThetaDist_CDF_Y_regrid;//

  int nDistEsurfaceModel = 0;
  int nDistEsurfaceModelRef = 0;
  int nDistAsurfaceModel = 0;

  int nEnDist = 0;
  double en0Dist = 0;
  double enDist = 0;
  int nAngDist = 0; 
  double ang0Dist = 0;
  double angDist = 0; 
  double dEdist = 0;
  double dAdist = 0;
  
  //size/bdry_face in comments. Distribute data upon partitioning.
  o::Write<o::Real> energyDistribution; //9k/detFace
  o::Write<o::Real> sputtDistribution;
  o::Write<o::Real> reflDistribution;

  int fluxEA = 0;

  template<typename T>
  void regrid2dCDF(const int nX, const int nY, const int nZ, 
    const o::HostWrite<T>& xGrid, const int nNew, const T maxNew, 
    const o::HostWrite<T>& cdf, o::HostWrite<T>& cdf_regrid);

  template<typename T>
  void make2dCDF(const int nX, const int nY, const int nZ, 
    const o::HostWrite<T>& distribution, o::HostWrite<T>& cdf);

  template<typename T>
  T interp1dUnstructured(const T samplePoint, const int nx, const T max_x, 
    const T* data, int& lowInd);

  void test();
};


OMEGA_H_DEVICE o::Real screeningLength(const o::Real projectileZ, 
    const o::Real targetZ) {
  o::Real bohrRadius = 5.29177e-11; //TODO
  return 0.885341*bohrRadius*pow(pow(projectileZ,(2.0/3.0)) + 
      pow(targetZ,(2.0/3.0)),(-1.0/2.0));
}

OMEGA_H_DEVICE o::Real stoppingPower (const o::Vector<3>& vel, const o::Real targetM, 
  const o::Real targetZ, const o::Real screenLength) {
  o::Real elCharge = gitrm::ELECTRON_CHARGE;
  o::Real ke2 = 14.4e-10; //TODO
  o::Real amu = gitrm::PTCL_AMU;
  o::Real atomZ = gitrm::PARTICLE_Z;
  auto protonMass = gitrm::PROTON_MASS;
  o::Real E0 = 0.5*amu*protonMass *1.0/elCharge * p::osh_dot(vel, vel);
  o::Real reducedEnergy = E0*(targetM/(amu+targetM))* (screenLength/(atomZ*targetZ*ke2));
  o::Real stopPower = 0.5*log(1.0 + 1.2288*reducedEnergy)/(reducedEnergy +
          0.1728*sqrt(reducedEnergy) + 0.008*pow(reducedEnergy, 0.1504));
  return stopPower;
}

//not used ? 
inline void surfaceErosion(PS* ptcls, o::Write<o::Real>& erosionData) {
  const auto psCapacity = ptcls->capacity();
  int atomZ = gitrm::PARTICLE_Z;
  auto amu = gitrm::PTCL_AMU;
  auto elCharge = gitrm::ELECTRON_CHARGE; //1.60217662e-19
  auto protonMass = gitrm::PROTON_MASS; //1.6737236e-27
  o::Real q = 18.6006;
  o::Real lambda = 2.2697;
  o::Real mu = 3.1273;
  o::Real Eth = 24.9885;
  o::Real targetZ = 74.0;
  o::Real targetM = 183.84;

  auto pid_ps = ptcls->get<PTCL_ID>();
  auto vel_ps = ptcls->get<PTCL_VEL>();
  auto lamb = PS_LAMBDA(const int& e, const int& pid, const int& mask) {
    auto ptcl = pid_ps(pid);
    auto screenLength = screeningLength(atomZ, targetZ);
    auto vel = p::makeVector3(pid, vel_ps);
    o::Real stopPower = stoppingPower(vel, targetM, targetZ, screenLength);
    o::Real E0 = 0.5*amu*protonMass* 1/elCharge * p::osh_dot(vel, vel);
    o::Real term = pow((E0/Eth - 1), mu);
    o::Real Y0 = q*stopPower*term/(lambda + term);
    erosionData[pid] = Y0;
  };
  ps::parallel_for(ptcls, lamb, "surfaceErosion");
}


//Note, elem_ids indexed by pids of ps, not ptcl. Don't rebuild after search_mesh 

inline void gitrm_surfaceReflection_test(PS* ptcls, GitrmSurfaceModel& sm,
    GitrmParticles& gp, const GitrmMesh& gm, o::Write<o::LO>& elem_ids) {
}
#endif

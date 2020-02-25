#ifndef GITRM_SURFACE_MODEL_HPP
#define GITRM_SURFACE_MODEL_HPP

#include <cstdarg>
#include "pumipic_adjacency.hpp"
#include "GitrmParticles.hpp" 
#include "GitrmMesh.hpp"

namespace o = Omega_h;
namespace p = pumipic;

namespace gitrm {
//TODO get from config
const int SURFACE_FLUX_EA = 0;
const int BOUNDARY_ATOM_Z = surfaceAndMaterialModelZ;
const o::Real DELTA_SHIFT_BDRY_REFL = 1.0e-4; 
const int SURFACE_ID_MAX = 260;

}

class GitrmSurfaceModel {
public:
  GitrmSurfaceModel(GitrmMesh& gm, std::string ncFile);
  void getConfigData(std::string ncFile);
  void initSurfaceModelData(std::string ncFile, bool debug=false);
  void prepareSurfaceModelData();
  void setFaceId2SurfaceIdMap();
  void writeOutSurfaceData(std::string fileName="surface.nc");

  void getSurfaceModelData(const std::string fileName,
   const std::string dataName, const std::vector<std::string>& shapeNames,
   const std::vector<int> shapeInds, o::Reals& data, int* size=nullptr);

  GitrmMesh& gm;
  o::Mesh& mesh;
  std::string ncFile;
  int numDetectorSurfaceFaces = 0;
  o::HostWrite<o::LO> surfaceAndMaterialModelIds;
  int numSurfMaterialFaces = 0;
  o::LOs surfaceAndMaterialOrderedIds;
  int nDetectSurfaces = 0;
  o::LOs detectorSurfaceOrderedIds;
  o::Reals bdryFaceMaterialZs;

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

  //input surface model data
  // shapes are given as comments
  o::Reals enSputtRefCoeff; //nEnSputtRefCoeff
  o::Reals angSputtRefCoeff; // nAngSputtRefCoeff
  o::Reals enSputtRefDistIn; //nEnSputtRefDistIn
  o::Reals angSputtRefDistIn; //nAngSputtRefDistIn
  o::Reals enSputtRefDistOut; // nEnSputtRefDistOut
  o::Reals enSputtRefDistOutRef; //nEnSputtRefDistOutRef
  o::Reals angPhiSputtRefDistOut; //nAngSputtRefDistOut
  o::Reals angThetaSputtRefDistOut; //nAngSputtRefDistOut
  o::Reals sputtYld; // nEnSputtRefCoeff X nAngSputtRefCoeff
  o::Reals reflYld; //  nEnSputtRefCoeff X nAngSputtRefCoeff
  // nEnSputtRefCoeff X nAngSputtRefCoeff X nEnSputtRefDistOut
  o::Reals enDist_Y;
  // nEnSputtRefCoeff X nAngSputtRefCoeff X nEnSputtRefDistOutRef 
  o::Reals enDist_R;
  //nEnSputtRefCoeff X nAngSputtRefCoeff X nAngSputtRefDistOut
  o::Reals angPhiDist_Y;
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

  void regrid2dCDF(const int nX, const int nY, const int nZ, 
    const o::HostWrite<o::Real>& xGrid, const int nNew, const o::Real maxNew, 
    const o::HostWrite<o::Real>& cdf, o::HostWrite<o::Real>& cdf_regrid);

  void make2dCDF(const int nX, const int nY, const int nZ, 
    const o::HostWrite<o::Real>& distribution, o::HostWrite<o::Real>& cdf);

  o::Real interp1dUnstructured(const o::Real samplePoint, const int nx, 
    const o::Real max_x, const o::Real* data, int& lowInd);
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
  o::Real E0 = 0.5*amu*protonMass *1.0/elCharge * o::inner_product(vel, vel);
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
  //TODO get from config
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
    o::Real E0 = 0.5*amu*protonMass* 1/elCharge * o::inner_product(vel, vel);
    o::Real term = pow((E0/Eth - 1), mu);
    o::Real Y0 = q*stopPower*term/(lambda + term);
    erosionData[pid] = Y0;
  };
  ps::parallel_for(ptcls, lamb, "surfaceErosion");
}

//Note, elem_ids indexed by pids of ps, not ptcl. Don't rebuild after search_mesh 
inline void gitrm_surfaceReflection(PS* ptcls, GitrmSurfaceModel& sm,
    GitrmParticles& gp, const GitrmMesh& gm, o::Write<o::LO>& elem_ids,
    bool debug=false ) {
  bool useGitrRnd = USE_GITR_RND_NUMS;
  if(!gp.ranSurfaceReflection)
    gp.ranSurfaceReflection = true;
  const auto& testGitrPtclStepData = gp.testGitrPtclStepData;
  const auto testGDof = gp.testGitrStepDataDof;
  const auto testGNT = gp.testGitrStepDataNumTsteps;
  const auto testGitrReflInd = gp.testGitrReflectionRndInd;
  const auto iTimeStep = iTimePlusOne - 1;
  if(useGitrRnd)
    OMEGA_H_CHECK(gp.testGitrOptSurfaceModel);
  else
    OMEGA_H_CHECK(!gp.testGitrOptSurfaceModel);

  auto bdrys = gm.bdryFaceOrderedIds;

  auto surfIdMax = gitrm::SURFACE_ID_MAX;
  o::Real pi = o::PI;
  o::Real shiftRefl = gitrm::DELTA_SHIFT_BDRY_REFL;
  o::LO fluxEA = gitrm::SURFACE_FLUX_EA;
  auto amu = gitrm::PTCL_AMU;
  auto elCharge = gitrm::ELECTRON_CHARGE;
  auto protonMass = gitrm::PROTON_MASS;
  auto mesh = sm.mesh;
  const auto coords = mesh.coords();
  const auto face_verts = mesh.ask_verts_of(2);
  const auto f2r_ptr = mesh.ask_up(o::FACE, o::REGION).a2ab;
  const auto f2r_elem = mesh.ask_up(o::FACE, o::REGION).ab2b;
  const auto side_is_exposed = mark_exposed_sides(&mesh);
  const auto mesh2verts = mesh.ask_elem_verts();
  const auto down_r2fs = mesh.ask_down(3, 2).ab2b;

  //TODO replace by Kokkos random
  const auto scsCapacity = ptcls->capacity();
  o::HostWrite<o::Real> rnd_h(4*scsCapacity);
  for(auto i=0; i<scsCapacity; ++i) {
    rnd_h[i] = (double)(std::rand())/RAND_MAX;
  }
  auto rands = o::Reals(o::Write<o::Real>(rnd_h));
  
  if(false) {
    auto sputtDist_test = sm.sputtDistribution;
    printf("sputDistribution-size: %d \n", sputtDist_test.size());
    auto pid_ps_test = ptcls->get<PTCL_ID>();
    auto lam = PS_LAMBDA(const int& elem, const int& pid, const int& mask) {
      if(mask >0 && pid_ps_test(pid)<5 ) {
        for(int i=0; i<2; ++i)
          printf("test sputt %.15e \n", sputtDist_test[i]);
      }
    };
    ps::parallel_for(ptcls,lam, "testLambda");
  }

  //input data
  const auto nEnSputtRefCoeff = sm.nEnSputtRefCoeff; // nE_sputtRefCoeff
  const auto nAngSputtRefCoeff = sm.nAngSputtRefCoeff; // nA_sputtRefCoeff 
  auto angSputtRefCoeff = sm.angSputtRefCoeff;  // A_sputtRefCoeff
  auto enLogSputtRefCoef = sm.enLogSputtRefCoef; //Elog_sputtRefCoeff
  auto sputtYld = sm.sputtYld; //spyl_surfaceModel 
  auto reflYld = sm.reflYld; // rfyl_surfaceModel
  const auto nEnSputtRefDistOut = sm.nEnSputtRefDistOut; // nE_sputtRefDistOut
  const auto nEnSputtRefDistOutRef = sm.nEnSputtRefDistOutRef; //nE_sputtRefDistOutRef
  const auto nAngSputtRefDistOut = sm.nAngSputtRefDistOut; //nA_sputtRefDistOut
  const auto nEnSputtRefDistIn = sm.nEnSputtRefDistIn; // nE_sputtRefDistIn
  const auto nAngSputtRefDistIn = sm.nAngSputtRefDistIn; // nA_sputtRefDistIn
  auto enSputtRefDistIn = sm.enSputtRefDistIn; // E_sputtRefDistIn
  auto angSputtRefDistIn = sm.angSputtRefDistIn; // A_sputtRefDistIn 
  auto enSputtRefDistOut = sm.enSputtRefDistOut;// E_sputtRefDistOut
  auto enSputtRefDistOutRef = sm.enSputtRefDistOutRef; //E_sputtRefDistOutRef
  auto angPhiSputtRefDistOut = sm.angPhiSputtRefDistOut; // A_sputtRefDistOut
  auto energyDistGrid01 = sm.energyDistGrid01; //energyDistGrid01
  auto energyDistGrid01Ref = sm.energyDistGrid01Ref; // energyDistGrid01Ref 
  auto angleDistGrid01 = sm.angleDistGrid01; // angleDistGrid01
  auto enLogSputtRefDistIn = sm.enLogSputtRefDistIn;
  auto enDist_CDF_Y_regrid = sm.enDist_CDF_Y_regrid; //EDist_CDF_Y_regrid
  auto angPhiDist_CDF_Y_regrid = sm.angPhiDist_CDF_Y_regrid; //ADist_CDF_Y_regrid
  auto enDist_CDF_R_regrid = sm.enDist_CDF_R_regrid;  //EDist_CDF_R_regrid
  auto angPhiDist_CDF_R_regrid = sm.angPhiDist_CDF_R_regrid; //ADist_CDF_R_regrid

  const auto nEnDist = sm.nEnDist; //nEdist
  const auto en0Dist = sm.en0Dist; // E0dist
  const auto enDist = sm.enDist; // Edist
  const auto nAngDist = sm.nAngDist; // nAdist
  const auto ang0Dist = sm.ang0Dist; // A0dist
  //const auto angDist = sm.angDist; // Adist
  const auto dEdist = sm.dEdist;
  const auto dAdist = sm.dAdist; 

  //data collection
  //auto surfaces = mesh.get_array<o::LO>(o::FACE, "SurfaceIndex");
  auto sumPtclStrike = o::deep_copy(mesh.get_array<o::Int>(o::FACE, "SumParticlesStrike"));
  auto sputtYldCount = o::deep_copy(mesh.get_array<o::Int>(o::FACE, "SputtYldCount")); 
  auto sumWtStrike = o::deep_copy(mesh.get_array<o::Real>(o::FACE, "SumWeightStrike"));
  auto grossDeposition = o::deep_copy(mesh.get_array<o::Real>(o::FACE, "GrossDeposition"));
  auto grossErosion = o::deep_copy(mesh.get_array<o::Real>(o::FACE, "GrossErosion"));
  auto aveSputtYld = o::deep_copy(mesh.get_array<o::Real>(o::FACE, "AveSputtYld"));  
  const auto& xpoints = gp.wallCollisionPts; //idexed by ptcl, not pid of scs
  auto& xfaces = gp.wallCollisionFaceIds;
  auto energyDist = sm.energyDistribution;
  auto sputtDist = sm.sputtDistribution;
  auto reflDist = sm.reflDistribution;
  auto surfaceIds = sm.surfaceAndMaterialOrderedIds;
  auto materials = sm.bdryFaceMaterialZs;

  auto pid_ps = ptcls->get<PTCL_ID>();
  auto next_pos_ps = ptcls->get<PTCL_NEXT_POS>();
  auto pos_ps = ptcls->get<PTCL_POS>();
  auto vel_ps = ptcls->get<PTCL_VEL>();
  auto ps_weight = ptcls->get<PTCL_WEIGHT>();
  auto ps_charge = ptcls->get<PTCL_CHARGE>();
  auto scs_hitNum = ptcls->get<PTCL_HIT_NUM>();
  auto ps_newVelMag = ptcls->get<PTCL_VMAG_NEW>();
  auto lamb = PS_LAMBDA(const int& elem, const int& pid, const int& mask) {
    if(mask >0  && elem_ids[pid]==-1) {
      auto elemId = elem;
      auto ptcl = pid_ps(pid);
      auto fid = xfaces[ptcl];
      if(debug) 
        printf("surface model fid %d\n",fid);
      if(fid >= 0) {
        if(debug && side_is_exposed[fid])
          printf("surf0 timestep %d ptcl %d fid %d\n", iTimeStep, ptcl, fid);
        OMEGA_H_CHECK(side_is_exposed[fid]);
        auto weight = ps_weight(pid);
        auto newWeight = weight; 
        auto surfId = surfaceIds[fid]; //surfaces[fid]; //ids 0..
        auto gridId = fid;
        auto pelem = p::elem_id_of_bdry_face_of_tet(fid, f2r_ptr, f2r_elem);
        if(elemId != pelem)
          elemId = pelem;
        auto vel = p::makeVector3(pid, vel_ps );
        auto xpoint = o::zero_vector<3>();
        for(o::LO i=0; i<3; ++i)
          xpoint[i] = xpoints[ptcl*3+i];
        //firstColl(pid) = 1; //scs
        auto magPath = o::norm(vel);
        auto E0 = 0.5*amu*protonMass*(magPath*magPath)/elCharge;

        if(debug) {
          auto face1 = p::get_face_coords_of_tet(face_verts, coords, fid);
          printf("surf1 faceid %d face %g %g %g : %g %g %g : %g %g %g\n", 
            fid, face1[0][0], face1[0][1], face1[0][2],
            face1[1][0], face1[1][1], face1[1][2], face1[2][0], face1[2][1], face1[2][2]);
        }
         
        if(debug)
          printf("surf2 timestep %d ptcl %d xpoint= pos  %.15e %.15e  %.15e elemId %d "
            "vel %.15e %.15e %.15e amu %.15e  weight %.15e mag %.15e E0 %.15e nEnDist %d\n", 
            iTimeStep, ptcl, xpoint[0],xpoint[1], xpoint[2], 
            elemId, vel[0], vel[1], vel[2], amu, weight, magPath, E0, nEnDist);

        OMEGA_H_CHECK(nEnDist > 0);
        if(E0 > enDist) //1000 
          E0 = enDist - enDist/nEnDist; // 990;   
        //boundary normal points outwards
        auto surfNormOut = p::face_normal_of_tet(fid, elemId, coords, mesh2verts, 
          face_verts, down_r2fs); 
        auto surfNormIn = -surfNormOut;
        //debug only
        auto bFaceNorm = p::bdry_face_normal_of_tet(fid, coords, face_verts);
        auto abc = p::get_face_coords_of_tet(face_verts, coords, fid);
         if(debug)
          printf("surf3 timestep %d ptcl %d surfNormOut %.15e %.15e %.15e bfaceNorm "
            "%.15e %.15e %.15e bdry %g %g %g : %g %g %g : %g %g %g \n",iTimeStep, ptcl,
            surfNormOut[0], surfNormOut[1], surfNormOut[2], bFaceNorm[0], bFaceNorm[1],
            bFaceNorm[2], abc[0][0], abc[0][1], abc[0][2], abc[1][0], abc[1][1], 
            abc[1][2], abc[2][0], abc[2][1], abc[2][2]);
        //end debug

        auto magSurfNorm = o::norm(surfNormIn);
        auto normVel = o::normalize(vel);
        auto ptclProj = o::inner_product(normVel, surfNormIn);
        auto thetaImpact = acos(ptclProj);
        if(thetaImpact > pi*0.5)
           thetaImpact = abs(thetaImpact - pi);
        thetaImpact = thetaImpact*180.0/pi;
        if(thetaImpact < 0) 
          thetaImpact = 0;
        if(o::are_close(E0, 0))
          thetaImpact = 0;
        o::Real Y0 = 0;
        o::Real R0 = 0;

        auto materialZ = materials[fid];
        if(materialZ > 0) {
          Y0 = p::interpolate2d_wgrid(sputtYld, angSputtRefCoeff, enLogSputtRefCoef,
             nAngSputtRefCoeff, nEnSputtRefCoeff, thetaImpact, log10(E0), true,1,0,1);
          R0 = p::interpolate2d_wgrid(reflYld, angSputtRefCoeff, enLogSputtRefCoef,
            nAngSputtRefCoeff, nEnSputtRefCoeff, thetaImpact, log10(E0), true,1,0,1);
        }
        if(debug)
          printf("surf4 timestep %d ptcl %d interpolated Y0 %.15e R0 %.15e normVel "
            "%.15e %.15e %.15e surfNormOut %.15e %.15e %.15e thetaImpact %.15e "
            " materialZ %g \n", iTimeStep, ptcl, Y0, R0, normVel[0], normVel[1],
            normVel[2], surfNormOut[0], surfNormOut[1], surfNormOut[2], thetaImpact,
            materialZ);
        auto totalYR = Y0 + R0;
        //particle either reflects or deposits
        auto sputtProb = (totalYR >0) ? Y0/totalYR : 0;
        int didReflect = 0;

        double rand7 = 0, rand8 = 0, rand9 = 0, rand10 = 0;
        if(useGitrRnd) {
          auto beg = ptcl*testGNT*testGDof + iTimeStep*testGDof + testGitrReflInd;
          rand7 = testGitrPtclStepData[beg];
          rand8 = testGitrPtclStepData[beg+1];
          rand9 = testGitrPtclStepData[beg+2];
          rand10 = testGitrPtclStepData[beg+3];
        } else {
          rand7 = rands[4*pid];
          rand8 = rands[4*pid+1];
          rand9 = rands[4*pid+2];
          rand10 = rands[4*pid+3];
        }
        o::Real eInterpVal = 0;
        o::Real aInterpVal = 0;
        o::Real addGrossDep = 0; 
        o::Real addGrossEros = 0;
        o::Real addAveSput = 0;
        o::Int addSpYCount = 0;
        o::Real addSumWtStk = 0;
        o::Int addSumPtclStk = 0;
        if(debug)
          printf("surf5 timestep %d ptcl %d totalYR %.15e surfId %d gridId %d "
            "sputtProb %.15e rand7 %.15e bdry %d \n", 
            iTimeStep, ptcl, totalYR, surfId, gridId, sputtProb, rand7, bdrys[fid]);
        if(totalYR > 0) {
          if(rand7 > sputtProb) { //reflect
            //resetting hitface
            xfaces[ptcl] = -1;
            // for next push
            if(debug)
              printf(" surf51 timestep %d ptcl %d nA %d nASOut %d nESIn %d "
                " nE %d nASIn %d nESIn %d rand8 %.15e thetaImpact %.15e log10(E0) %.15e \n", 
                iTimeStep, ptcl, nAngSputtRefDistOut, nAngSputtRefDistIn, 
                nEnSputtRefDistIn, nEnSputtRefDistOutRef, nAngSputtRefDistIn, 
                nEnSputtRefDistIn, rand8, thetaImpact, log10(E0)); 

            didReflect = 1;
            aInterpVal = p::interpolate3d_field(rand8, thetaImpact, log10(E0),
               nAngSputtRefDistOut, nAngSputtRefDistIn, nEnSputtRefDistIn, 
               angleDistGrid01, angSputtRefDistIn, enLogSputtRefDistIn, 
               angPhiDist_CDF_R_regrid);
            eInterpVal = p::interpolate3d_field(rand9, thetaImpact, log10(E0),
               nEnSputtRefDistOutRef, nAngSputtRefDistIn, nEnSputtRefDistIn, 
               energyDistGrid01Ref, angSputtRefDistIn, enLogSputtRefDistIn, 
               enDist_CDF_R_regrid);
            //newWeight=(R0/(1.0f-sputtProb))*weight;
            newWeight = weight*totalYR;
            if(debug)
              printf("surf6 reflects timestep %d ptcl %d  weight %.15e newWeight %.15e "
                "totalYR %.15e fluxEA %d aInterpVal %.15e  eInterpVal %.15e \n", 
                iTimeStep, ptcl, weight, newWeight,  totalYR, fluxEA, aInterpVal,
                eInterpVal);
            if(fluxEA > 0) {
              auto eDistInd = floor((eInterpVal-en0Dist)/dEdist);
              auto aDistInd = floor((aInterpVal-ang0Dist)/dAdist);
              if(surfId >=0 && eDistInd >= 0 && eDistInd < nEnDist && 
                 aDistInd >= 0 && aDistInd < nAngDist) {
                auto idx = surfId*nEnDist*nAngDist + eDistInd*nAngDist + aDistInd;
                if(debug)
                  printf("surf61 timestep %d ptcl %d reflDist @ %d newWeight %.15e"
                    " prev %.15e \n", iTimeStep, ptcl, idx, newWeight, reflDist[idx]);
                Kokkos::atomic_fetch_add(&(reflDist[idx]), newWeight);
              }
            } 
            if(surfId >= 0) { //id 0..
              if(debug)
                printf("surf7 surfId %d GrossDep+ %.15e addGrossDep %.15e  \n", 
                    surfId, weight*(1.0-R0), addGrossDep);
              addGrossDep += weight*(1.0-R0);
            }
          } else {//sputters
            if(debug) {
              printf("surf8 timestep %d ptcl %d interpolate3d E0 %.15e\n", iTimeStep, ptcl, E0);
              printf("rand81 %.15e thetaImpact %.15e log10(E0) %.15e nAngSputtRefDistOut %d "
                "nAngSputtRefDistIn %d nEnSputtRefDistIn %d\n", rand8, thetaImpact, log10(E0), 
                nAngSputtRefDistOut, nAngSputtRefDistIn, nEnSputtRefDistIn);              
            }
            //TODO merge with the above same 
            aInterpVal = p::interpolate3d_field(rand8, thetaImpact, log10(E0),
              nAngSputtRefDistOut, nAngSputtRefDistIn, nEnSputtRefDistIn,
              angleDistGrid01, angSputtRefDistIn, enLogSputtRefDistIn, angPhiDist_CDF_Y_regrid);
            if(debug)
              printf(" surf82 timestep %d ptcl %d interpolate3d \n", iTimeStep, ptcl);            
            eInterpVal = p::interpolate3d_field(rand9,thetaImpact, log10(E0),
              nEnSputtRefDistOut, nAngSputtRefDistIn, nEnSputtRefDistIn,
              energyDistGrid01, angSputtRefDistIn, enLogSputtRefDistIn, enDist_CDF_Y_regrid);
            newWeight = weight*totalYR;

            if(debug)
              printf(" surf83 timestep %d ptcl %d nA %d nASOut %d nESIn %d "
                " nE %d nASIn %d nESIn %d\n", 
                iTimeStep, ptcl, nAngSputtRefDistOut, 
                nAngSputtRefDistIn, nEnSputtRefDistIn,
                nEnSputtRefDistOutRef, nAngSputtRefDistIn, nEnSputtRefDistIn); 
            if(debug)
               printf("surf8 sputters timestep %d ptcl %d weight %.15e newWeight %.15e "
                " sputtProb %.15e aInterpVal %.15e eInterpVal %.15e\n", iTimeStep, ptcl, weight,
                newWeight, sputtProb, aInterpVal, eInterpVal);
            if(fluxEA > 0) {
              auto eDistInd = floor((eInterpVal-en0Dist)/dEdist);
              auto aDistInd = floor((aInterpVal-ang0Dist)/dAdist);
              if(surfId >= 0 && eDistInd >= 0 && eDistInd < nEnDist && 
                 aDistInd >= 0 && aDistInd < nAngDist) {
                auto idx = surfId*nEnDist*nAngDist + eDistInd*nAngDist + aDistInd;
                if(debug)
                  printf("surf9 timestep %d ptcl %d sputtDist idx  %d newWeight %.15e prev %.15e %.15e \n", 
                    iTimeStep, ptcl, idx, newWeight, sputtDist[idx], sputtDist[10]);
                Kokkos::atomic_fetch_add(&(sputtDist[idx]), newWeight);
              }
            }
            if(o::are_close(sputtProb, 0))
              newWeight = 0;
            if(surfId >= 0) {
              addGrossDep += weight*(1.0-R0);
              addGrossEros += newWeight;
              addAveSput += Y0;
              if(weight > 0) {
                addSpYCount += 1;
              }
            }
            if(debug)
              printf("surf10 surfId %d timestep %d ptcl %d newWeight %.15e GrossDep %.15e "
                " GrossEros %.15e AveSput %.15e weight %.15e SpYCount %d \n", surfId, iTimeStep,
                ptcl, newWeight, addGrossDep, addGrossEros, addSpYCount);
          }
        } else { // totalYR
          newWeight = 0;
          scs_hitNum(pid) = 2;
          double grossDep_ = 0;
          if(surfId >= 0) {
            addGrossDep += weight;  //TODO Dep ?
            grossDep_ = weight;
          }
          if(debug)
            printf("surf11 totalYR timestep %d ptcl %d weight %.15e surfId %d "
              " newWeight %.15e grossDep+ %.15e totalGrossDep %.15e \n", iTimeStep, ptcl,
              weight,surfId, newWeight, grossDep_, addGrossDep);
        }

        if(eInterpVal <= 0) {
          if(debug)
            printf("surf11+ eInterpVal <= 0 timestep %d ptcl %d didReflect %d "
              " weight %.15e \n", iTimeStep, ptcl, didReflect, weight );
          newWeight = 0;
          scs_hitNum(pid) = 2;
          if(surfId >= 0 && didReflect) {
            addGrossDep += weight; //TODO Dep ?
          }
        }
        if(surfId >=0) {
          addSumWtStk += weight;
          addSumPtclStk += 1;

          if(fluxEA > 0) {
            auto eDistInd = floor((E0-en0Dist)/dEdist);
            auto aDistInd = floor((thetaImpact-ang0Dist)/dAdist);
            if(surfId >= 0 && eDistInd >= 0 && eDistInd < nEnDist && 
                aDistInd >= 0 && aDistInd < nAngDist) {
              auto idx = surfId*nEnDist*nAngDist + eDistInd*nAngDist + aDistInd;
              if(debug)
                printf("surf12 timestep %d ptcl %d energyDist @ %d prev %.15e\n", 
                  iTimeStep, ptcl, idx, energyDist[idx]);
              Kokkos::atomic_fetch_add(&(energyDist[idx]), weight);
            }
          }
        } //surface

        if(debug)
          printf("surf13 timestep %d ptcl %d Atomics @id %d dep %.15e erosion %.15e "
           "avesput %.15e spYld %d wtStrike %.15e ptclStrike %d\n", iTimeStep, ptcl, 
           gridId, addGrossDep, addGrossEros, addAveSput, addSpYCount, 
           addSumWtStk, addSumPtclStk); 
        Kokkos::atomic_fetch_add(&(grossDeposition[gridId]), addGrossDep); 
        Kokkos::atomic_fetch_add(&(grossErosion[gridId]), addGrossEros);
        Kokkos::atomic_fetch_add(&(aveSputtYld[gridId]), addAveSput);
        Kokkos::atomic_fetch_add(&(sputtYldCount[gridId]), addSpYCount); 
        Kokkos::atomic_fetch_add(&(sumWtStrike[gridId]), addSumWtStk);
        Kokkos::atomic_fetch_add(&(sumPtclStrike[gridId]), addSumPtclStk);
        if(debug)
          printf("surf14 timestep %d ptcl %d materialZ %d newWeight %.15e elCharge %.15e "
            "protonMass %.15e\n",iTimeStep, ptcl, materialZ, newWeight, elCharge, protonMass);
        if(materialZ > 0 && newWeight > 0) {
          ps_weight(pid) = newWeight;
          scs_hitNum(pid) = 0;
          ps_charge(pid) = 0;
          o::Real elCharge = 1.602e-19; //FIXME  
          o::Real protonMass = 1.66e-27;//FIXME
          auto v0 = sqrt(2*eInterpVal*elCharge/(amu*protonMass));
          ps_newVelMag(pid) = v0;
          auto vSampled = o::zero_vector<3>(); 
          o::Real pi = 3.1415;//FIXME
          vSampled[0] = v0*sin(aInterpVal*pi/180)*cos(2.0*pi*rand10);
          vSampled[1] = v0*sin(aInterpVal*pi/180)*sin(2.0*pi*rand10);
          vSampled[2] = v0*cos(aInterpVal*pi/180);
          //done on top auto surfNormOut = p::bdry_face_normal_of_tet(fid,coords,face_verts);
          auto face = p::get_face_coords_of_tet(face_verts, coords, fid);
          auto surfPar = o::normalize(face[1] - face[0]); // bdry face vtx
          auto vecY = o::cross(surfNormIn, surfPar);
          if(debug)
            printf("surf15 timestep %d ptcl %d V0 %.15e rand10 %.15e vSampled %.15e %.15e %.15e "
              "norm %.15e %.15e %.15e surfPar %.15e %.15e %.15e  vecY %.15e %.15e %.15e \n", 
              iTimeStep, ptcl,v0, rand10, vSampled[0], vSampled[1], vSampled[2], 
              surfNormOut[0], surfNormOut[1], surfNormOut[2], surfPar[0], surfPar[1], 
              surfPar[2], vecY[0], vecY[1],vecY[2]);

          auto v = vSampled[0]*surfPar + vSampled[1]*vecY + vSampled[2]*surfNormIn;
          vSampled = v;
          for(int i=0; i<3; ++i)
            vel_ps(pid, i) = vSampled[i]*surfNormIn[i];
          //move reflected particle inwards
          //TODO for all particles, not only reflected ?
          auto newPos =  xpoint + shiftRefl*surfNormIn; //-1.0e-4*surfNorm GITR
          //updating next position of only intersecting ptcls. For others, add up.
          for(o::LO i=0; i<3; ++i)
            next_pos_ps(pid,i) = newPos[i]; //TODO other updates ?

          if(debug)
            printf("surf16 timestep %d ptcl %d xpt= pos %.15e %.15e %.15e => "
              " %.15e %.15e %.15e vel  %.15e %.15e %.15e =>  %.15e %.15e %.15e "
              " vsampled final  %.15e %.15e %.15e\n", 
              iTimeStep, ptcl, xpoint[0], xpoint[1], xpoint[2], newPos[0], newPos[1], 
              newPos[2], vel[0], vel[1], vel[2], vel_ps(pid, 0), vel_ps(pid, 1),
              vel_ps(pid, 2), vSampled[0], vSampled[1], vSampled[2]); 
        } else { //materialZ, newWeight
          scs_hitNum(pid) = 2;
        }

      } //fid
    } //mask
  }; //lambda
  ps::parallel_for(ptcls, lamb, "surfaceModel");
  mesh.add_tag(o::FACE, "SumParticlesStrike", 1, o::Read<o::Int>(sumPtclStrike));
  mesh.add_tag(o::FACE, "SputtYldCount", 1, o::Read<o::Int>(sputtYldCount));
  mesh.add_tag(o::FACE, "SumWeightStrike", 1, o::Reals(sumWtStrike));
  mesh.add_tag(o::FACE, "GrossDeposition", 1, o::Reals(grossDeposition));
  mesh.add_tag(o::FACE, "GrossErosion", 1, o::Reals(grossErosion));
  mesh.add_tag(o::FACE, "AveSputtYld", 1, o::Reals(aveSputtYld));
}

#endif

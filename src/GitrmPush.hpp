#ifndef GITRM_PUSH_HPP
#define GITRM_PUSH_HPP

#include "GitrmMesh.hpp"
#include "GitrmParticles.hpp"

#include <iostream>
#include <cmath>
#include <utility>

#include "Omega_h_adj.hpp"
#include "Omega_h_array_ops.hpp"
#include "Omega_h_scalar.hpp" //divide
#include "Omega_h_fail.hpp"


// Angle, DebyeLength etc were calculated at center of LONG tet, using BField.
inline void gitrm_calculateE(GitrmParticles& gp, o::Mesh &mesh, bool debug, const GitrmMesh &gm) {
  const int compareWithGitr = COMPARE_WITH_GITR;
  const int useGitrCLD = USE_GITR_CLD;
  const int replaceEQ = USE_GITR_EFILED_AND_Q;
  const int useGitrMidPt = USE_GITR_BFACE_MIDPT_N_CALC_CLD;
  const int useGitrD2bdry = USE_GITR_DIST2BDRY;
  const double biasPot = BIAS_POTENTIAL;
  if(compareWithGitr)
    iTimePlusOne++;
  
  const auto& temEl_d = gm.temEl_d;
  const auto& densEl_d = gm.densEl_d;
  const auto densElX0 = gm.densElX0;
  const auto densElZ0 = gm.densElZ0;
  const auto densElNx = gm.densElNx;
  const auto densElNz = gm.densElNz;
  const auto densElDx = gm.densElDx;
  const auto densElDz = gm.densElDz;
  const auto tempElX0 = gm.tempElX0;
  const auto tempElZ0 = gm.tempElZ0;
  const auto tempElNx = gm.tempElNx;
  const auto tempElNz = gm.tempElNz;
  const auto tempElDx = gm.tempElDx;
  const auto tempElDz = gm.tempElDz;
  o::Write<o::Real> larmorRadius_d = gm.larmorRadius_d;
  o::Write<o::Real> childLangmuirDist_d = gm.childLangmuirDist_d;
  o::LO checkAllBdryFaces = CHECK_ALL_BDRYFACES_FOR_D2BDRY;

  const auto& testGitrPtclStepData = gp.testGitrPtclStepData;
  const auto testGDof = gp.testGitrStepDataDof;
  const auto testGNT = gp.testGitrStepDataNumTsteps;
  const auto testGEind = gp.testGitrStepDataEfieldInd;
  const auto testGqInd = gp.testGitrStepDataChargeInd;
  const auto testGCLDInd = gp.testGitrStepDataCLDInd;
  const auto testGMinDistInd = gp.testGitrStepDataMinDistInd;
  const auto testGMidPtInd = gp.testGitrStepDataMidPtInd;
  const o::LO background_Z = BACKGROUND_Z;
  const auto iTimeStep = iTimePlusOne - 1;
  
  const auto biasedSurface = BIASED_SURFACE;
  const auto angles = mesh.get_array<o::Real>(o::FACE, "angleBdryBfield");
  const auto potentials = mesh.get_array<o::Real>(o::FACE, "potential");
  const auto debyeLengths = mesh.get_array<o::Real>(o::FACE, "DebyeLength");
  const auto larmorRadii = mesh.get_array<o::Real>(o::FACE, "LarmorRadius");
  const auto childLangmuirDists = mesh.get_array<o::Real>(o::FACE, "ChildLangmuirDist");
  auto* scs = gp.scs;
  auto pos_scs = scs->get<PTCL_POS>();
  auto efield_scs  = scs->get<PTCL_EFIELD>();
  auto pid_scs = scs->get<PTCL_ID>();

  auto charge_scs = scs->get<PTCL_CHARGE>();

  //NOTE arrays is based on pid, which is reset upon each rebuild
  const auto& closestPoints =  gp.closestPoints;
  const auto& faceIds = gp.closestBdryFaceIds;

  auto run = SCS_LAMBDA(const int& elem, const int& pid, const int& mask) { 
    if(mask >0) {
      auto ptcl = pid_scs(pid);
      auto faceId = faceIds[pid];
      //if(faceId < 0) faceId = 0;
      if(faceId < 0) {
        efield_scs(pid, 0) = 0;
        efield_scs(pid, 1) = 0;
        efield_scs(pid, 2) = 0;
      } else {
        // TODO angle is between surface normal and magnetic field at center of face
        // If  face is long, BField is not accurate. Calculate at closest point ?
        o::Real angle = angles[faceId];
        o::Real pot = potentials[faceId];
        //TODO remove this after testing
        pot = biasPot;
        o::Real debyeLength = debyeLengths[faceId];
        o::Real larmorRadius = larmorRadii[faceId];
        o::Real childLangmuirDist = childLangmuirDists[faceId];
        if(checkAllBdryFaces) {
          larmorRadius = larmorRadius_d[faceId];
          childLangmuirDist = childLangmuirDist_d[faceId];
        }

        auto pos = p::makeVector3(pid, pos_scs);
        o::Vector<3> closest;
        for(o::LO i=0; i<3; ++i)
          closest[i] = closestPoints[pid*3+i];
        auto distVector = closest - pos;
        auto dirUnitVector = o::normalize(distVector);
        auto d2bdry = p::osh_mag(distVector);
        o::Real emag = 0;

        o::Real gitrCLD=0;
        o::Real thisCLD=childLangmuirDist;
        o::Real thisD2bdry=d2bdry;
        o::Real gitrD2bdry=0;
        o::Real calcCLD=0;
        o::Real emagOrig=0;
        o::LO beg = ptcl*testGNT*testGDof + iTimeStep*testGDof;
        if(biasedSurface) {
          emag = pot/(2.0*childLangmuirDist)* exp(-d2bdry/(2.0*childLangmuirDist));
          if(compareWithGitr) {
            if(useGitrD2bdry) {
              gitrD2bdry = testGitrPtclStepData[beg + testGMinDistInd];
              thisD2bdry= gitrD2bdry;
            }

            if(useGitrCLD) {
              gitrCLD = testGitrPtclStepData[beg + testGCLDInd]; 
              thisCLD = gitrCLD;
            } else if(useGitrMidPt) {
              auto gitrMidPtx = testGitrPtclStepData[beg + testGMidPtInd];
              auto gitrMidPty = testGitrPtclStepData[beg + testGMidPtInd+1];
              auto gitrMidPtz = testGitrPtclStepData[beg + testGMidPtInd+2];
              o::Real dlen = 0;
              
              auto pos = p::makeVector3FromArray({gitrMidPtx, gitrMidPty, gitrMidPtz});
              o::Real tel = p::interpolate2dField(temEl_d, tempElX0, tempElZ0, tempElDx, 
                tempElDz, tempElNx, tempElNz, pos, true);
              o::Real nel = p::interpolate2dField(densEl_d, densElX0, densElZ0, densElDx, 
                densElDz, densElNx, densElNz, pos, true);

              if(o::are_close(nel, 0.0))
                dlen = 1.0e12;
              else 
                dlen = sqrt(8.854187e-12*tel/(nel*pow(background_Z,2)*1.60217662e-19));
              // Only for BIASED surface
              o::Real calcCLD = 0;
              if(tel > 0.0)
                calcCLD = dlen * pow(abs(pot)/tel, 0.75);
              else 
                calcCLD = 1e12;
              thisCLD = calcCLD;
            }
            
            emag = pot/(2.0*thisCLD)* exp(-thisD2bdry/(2.0*thisCLD));
            emagOrig = pot/(2.0*childLangmuirDist)* exp(-d2bdry/(2.0*childLangmuirDist));
          }
        } else { 
          o::Real fd = 0.98992 + 5.1220E-03 * angle - 7.0040E-04 * pow(angle,2.0) +
                       3.3591E-05 * pow(angle,3.0) - 8.2917E-07 * pow(angle,4.0) +
                       9.5856E-09 * pow(angle,5.0) - 4.2682E-11 * pow(angle,6.0);
          emag = pot*(fd/(2.0 * debyeLength)* exp(-d2bdry/(2.0 * debyeLength))+ 
                  (1.0 - fd)/(larmorRadius)* exp(-d2bdry/larmorRadius));
        }
        if(isnan(emag))
          emag = 0;
        if(o::are_close(d2bdry, 0.0) || o::are_close(larmorRadius, 0.0)) {
          emag = 0.0;
          dirUnitVector[0] = dirUnitVector[1] = dirUnitVector[2] = 0;
        }
        auto exd = emag*dirUnitVector;
        for(int i=0; i<3; ++i)
          efield_scs(pid, i) = exd[i];

        if(debug){
          printf("CalcE: ptcl %d dist2bdry %g  bdryface:%d angle:%g pot:%g"
           " DebL:%g Larm:%g CLD:%g \n", ptcl, d2bdry, faceId, angle, 
           pot, debyeLength, larmorRadius, childLangmuirDist);
          printf("CalcE: ptcl %d pos %g %g %g closest %g %g %g distVec %g %g %g \n", 
            ptcl, pos[0], pos[1], pos[2], closest[0], closest[1], closest[2],
            distVector[0], distVector[1], distVector[2]);
        } 
        if(debug)
          printf("CalcE:: TStep %d ptcl %d d2bdry %g gitrD2bdry %g  emag %g, emagOrig %g "
              "\t CLDorig %g thisCLD %g gitrCLD %g calcCLD %g elem %d dir %g %g %g \n", 
            iTimeStep, ptcl, d2bdry, gitrD2bdry,  emag, emagOrig, childLangmuirDist, thisCLD, 
            gitrCLD, calcCLD, elem, dirUnitVector[0], dirUnitVector[1], dirUnitVector[2]);

        if(compareWithGitr) {// TODO
          //beg: pindex*nT*dof_intermediate + (nthStep-1)*dof_intermediate
          auto beg = ptcl*testGNT*testGDof + iTimeStep*testGDof;
          auto gitrE0 = testGitrPtclStepData[beg + testGEind];
          auto gitrE1 = testGitrPtclStepData[beg + testGEind+1];
          auto gitrE2 = testGitrPtclStepData[beg + testGEind+2];
          auto gitrQ = testGitrPtclStepData[beg + testGqInd];
          auto gitrCLD = testGitrPtclStepData[beg + testGCLDInd];
          auto gitrMidPtx = testGitrPtclStepData[beg + testGMidPtInd];
          auto gitrMidPty = testGitrPtclStepData[beg + testGMidPtInd+1];
          auto gitrMidPtz = testGitrPtclStepData[beg + testGMidPtInd+2];
          auto gitrD2bdry = testGitrPtclStepData[beg + testGMinDistInd];
          if(replaceEQ) {
            efield_scs(pid, 0) = gitrE0;
            efield_scs(pid, 1) = gitrE1;
            efield_scs(pid, 2) = gitrE2;
            charge_scs(pid) = gitrQ;
          }
         // if(debug)
            printf("calcE ptcl %d timestep %d q %d  efield %g %g %g  GITRmindist %g gitrCLD %g "
                "GITR_q %d GITR_Efield %g  %g  %g useGITR_EQ %d useGITR_CLD %d \n", 
                ptcl, iTimeStep, charge_scs(pid), efield_scs(pid, 0), efield_scs(pid, 1), efield_scs(pid, 2), 
               gitrD2bdry, gitrCLD, gitrQ, gitrE0, gitrE1, gitrE2,  replaceEQ, useGitrCLD);
        }
      } //faceId
    } //mask
  };
  scs->parallel_for(run);
}
/*NOTE: on calculateE
  1st 2 particles used have same position in GITR and GITRm, since it is from file
  Only difference is in input CLDist.

  with 580K mesh of specified mesh size
  0 dist2bdry 1e-06 angle:1.30716 pot:250 Deb:8.39096e-06 Larm:0.0170752 CLD:0.00031778 
  1 dist2bdry 1e-06 angle:1.14724 pot:250 Deb:8.31213e-06 Larm:0.0201203 CLD:0.00024613 
  0 pos 0.0137135 -0.0183835 1e-06 closest 0.0137135 -0.0183835 0 distVec 0 3.46945e-18 -1e-06 
  1 pos -0.000247493 0.0197626 1e-06 closest -0.000247493 0.0197626 0 distVec -2.1684e-19 0 -1e-06 
  0 E 0 1.36258e-06 -392736 :d2bdry 1e-06 emag 392736 pot 250 CLD 0.00031778 dir 0 3.46945e-12 -1 
  1 E -1.09902e-07 0 -506832 :d2bdry 1e-06 emag 506832 pot 250 CLD 0.00024613 dir -2.1684e-13 0 -1 

  with large sized mesh with no size specified
  GITR: E 0 0 -400917 Emax 400917 CLD 0.000311285 xyx 0.0137135 -0.0183835 1e-06 minD 1e-06 dir 0 0 -1 
  GITRm E 6.78808e-07 0 -391306 :d2bdry 1e-06 emag 391306 pot 250 CLD 0.000318942 dir 1.73472e-12 0 -1
  GITRm:pos 0.0137135 -0.0183835 1e-06 closest 0.0137135 -0.0183835 0 distVec 1.73472e-18 0 -1e-06

  GITR E 0 0 -573742 Emax 573742 CLD 0.000217368 xyx -0.000247493 0.0197626 1e-06 minD 1e-06 dir 0 0 -1
  GITRm:pos -0.000247493 0.0197626 1e-06 closest -0.000247493 0.0197626 0 distVec 9.21572e-19 3.46945e-18 -1e-06
  GITRm: E 3.3292e-07 1.25335e-06 -361253 :d2bdry 1e-06 emag 361253 pot 250 CLD 0.000345518 dir 9.21572e-13 3.46945e-12 -1
*/

// NOTE: for extruded mesh, TETs are too long that particle's projected position
// onto nearest poloidal plane is to be used to interpolate fields from 
// vertices of corresponding face of the TET. For a non-extruded mesh this projection
// is not possible and interpolation from all points of the TET is to be used.
// no input EFIELD from file in GITR, other than calculated near wall
inline void gitrm_borisMove(particle_structs::SellCSigma<Particle>* scs, 
  const GitrmMesh &gm, const o::Real dTime, bool debug=false,
  bool testExample=false) {
  o::Mesh &mesh = gm.mesh;  
  const auto& coords = mesh.coords();
  const auto& mesh2verts = mesh.ask_elem_verts();
  auto amu = PTCL_AMU;
  // Only 3D field from mesh tags
  auto use3dField = USE3D_BFIELD;
  auto use2dInputFields = USE2D_INPUTFIELDS;
  auto useConstantBField = USE_CONSTANT_BFIELD;

  int iTimeStep = iTimePlusOne - 1;
  
  if(PISCESRUN)
    OMEGA_H_CHECK(useConstantBField);
  o::Vector<3> bFieldConst; // At previous_pos
  if(useConstantBField)
    bFieldConst = p::makeVectorHost(CONSTANT_BFIELD);
  auto eFieldConst = p::makeVectorHost(CONSTANT_EFIELD);

  //const auto BField = o::Reals( mesh.get_array<o::Real>(o::VERT, "BField")); //TODO
  // Only if used 2D field read from file
  auto shiftB = gm.mesh2Bfield2Dshift;
  o::Real bxz[] = {gm.bGridX0, gm.bGridZ0, gm.bGridDx, gm.bGridDz};
  auto bGridNx = gm.bGridNx;
  auto bGridNz = gm.bGridNz;
  const auto &BField_2d = gm.Bfield_2d;
  auto pid_scs = scs->get<PTCL_ID>();
  auto tgt_scs = scs->get<PTCL_NEXT_POS>();
  auto efield_scs  = scs->get<PTCL_EFIELD>();
  auto pos_scs = scs->get<PTCL_POS>();
  auto vel_scs = scs->get<PTCL_VEL>();
  auto charge_scs = scs->get<PTCL_CHARGE>();

  auto boris = SCS_LAMBDA(const int& elem, const int& pid, const int& mask) {
    if(mask >0) {
      auto ptcl = pid_scs(pid);
      auto vel = p::makeVector3(pid, vel_scs);
      auto pos = p::makeVector3(pid, pos_scs);
      auto charge = charge_scs(pid);
      auto bField = o::zero_vector<3>();
      // for neutral tracking skip gitrm_calculateE()
      auto eField = p::makeVector3(pid, efield_scs);
      // In GITR only constant EField is used
      eField += eFieldConst;

      //TODO move to unit test
      if(testExample)
        pos = p::makeVector3FromArray({0.0137135, -0.0183835, 1e-06});
      
      if(useConstantBField) {
        bField = bFieldConst;
      } else if(use3dField) {
        auto bcc = o::zero_vector<4>();
        p::findBCCoordsInTet(coords, mesh2verts, pos, elem, bcc);
        // BField is 3 component array
        // p::interpolate3dFieldTet(mesh2verts, BField, elem, bcc, bField);  
      } else if(use2dInputFields) { //TODO for testing
        auto pos = o::zero_vector<3>();
        //cylindrical symmetry, height (z) is same.
        auto rad = sqrt(pos[0]*pos[0] + pos[1]*pos[1]);
        // projecting point to y=0 plane, since 2D data is on const-y plane.
        pos[0] = rad + shiftB; // D3D 1.6955m.
        pos[1] = 0;
        p::interp2dVector(BField_2d,  bxz[0], bxz[1], bxz[2], bxz[3], bGridNx,
          bGridNz, pos, bField, false);
      }
      //TODO move to unit tests
      if(testExample) {
        //TODO  velocity is to be set
        printf("ptcl %d pid %d pos_new %g %g %g\n", ptcl, pid, pos[0], pos[1], pos[2]);
        printf("ptcl %d pid %d eField_ %g %g %g\n", ptcl, pid, eField[0], 
          eField[1], eField[2]);
        printf("ptcl %d pid %d bField_ %g %g %g\n", ptcl, pid, bField[0],
         bField[1], bField[2]);
        eField = p::makeVector3FromArray({0, 0, -400917}); // E
        //bField = p::makeVector3FromArray({0, 0, -0.08}); //matches
         //xyz 0.013715 -0.0183798 7.45029e-06 //result
         //vel 291.384 726.638 1290.06 // result, but its input is also used
      }

      auto vel0 = vel;

      OMEGA_H_CHECK((amu >0) && (dTime>0));
      o::Real bFieldMag = p::osh_mag(bField);
      o::Real qPrime = charge*1.60217662e-19/(amu*1.6737236e-27) *dTime*0.5;
      o::Real coeff = 2.0*qPrime/(1.0+(qPrime*bFieldMag)*(qPrime*bFieldMag));

      //v_minus = v + q_prime*E;
      o::Vector<3> qpE = qPrime*eField;
      o::Vector<3> vMinus = vel + qpE;
      //v_prime = v_minus + q_prime*(v_minus x B)
      o::Vector<3> vmxB = o::cross(vMinus,bField);
      o::Vector<3> qpVmxB = qPrime*vmxB;
      o::Vector<3> vPrime = vMinus + qpVmxB;
      //v = v_minus + coeff*(v_prime x B)
      o::Vector<3> vpxB = o::cross(vPrime, bField);
      o::Vector<3> cVpxB = coeff*vpxB;
      vel = vMinus + cVpxB;
      //v = v + q_prime*E
      vel = vel + qpE;

      // Next position and velocity
      auto tgt = pos + vel * dTime;
      tgt_scs(pid, 0) = tgt[0];
      tgt_scs(pid, 1) = tgt[1];
      tgt_scs(pid, 2) = tgt[2];
      vel_scs(pid, 0) = vel[0];
      vel_scs(pid, 1) = vel[1];
      vel_scs(pid, 2) = vel[2];
      if(debug) {
        printf("ptcl %d timestep %d e %d charge %d pos %g %g %g =>  %g %g %g  "
          "vel %.1f %.1f %.1f =>  %.1f %.1f %.1f eField %g %g %g\n", ptcl, iTimeStep, 
          elem, charge, pos[0], pos[1], pos[2], tgt[0], tgt[1], tgt[2], 
          vel0[0], vel0[1], vel0[2], vel[0], vel[1], vel[2], eField[0], eField[1], eField[2]);
      }
    }// mask
  };
  scs->parallel_for(boris);
}

inline void neutralBorisMove(SCS* scs,  const o::Real dTime) {
  auto vel_scs = scs->get<PTCL_VEL>();
  auto tgt_scs = scs->get<PTCL_NEXT_POS>();
  auto pos_scs = scs->get<PTCL_POS>();
  auto boris = SCS_LAMBDA(const int& elem, const int& pid, const int& mask) {
    if(mask >0) {
      auto vel = p::makeVector3(pid, vel_scs);
      auto pos = p::makeVector3(pid, pos_scs);
      // Next position and velocity
      tgt_scs(pid, 0) = pos[0] + vel[0] * dTime;
      tgt_scs(pid, 1) = pos[1] + vel[1] * dTime;
      tgt_scs(pid, 2) = pos[2] + vel[2] * dTime;
      vel_scs(pid, 0) = vel[0];
      vel_scs(pid, 1) = vel[1];
      vel_scs(pid, 2) = vel[2];      
    }// mask
  };
  scs->parallel_for(boris, "neutralBorisMove");
} 

inline void neutralBorisMove_float(SCS* scs,  const o::Real dTime, bool debug = false) {
  auto vel_scs = scs->get<PTCL_VEL>();
  auto tgt_scs = scs->get<PTCL_NEXT_POS>();
  auto pos_scs = scs->get<PTCL_POS>();
  auto pid_scs = scs->get<PTCL_ID>();
  auto boris = SCS_LAMBDA(const int& elem, const int& pid, const int& mask) {
    if(mask >0) {
      auto vel = p::makeVector3(pid, vel_scs);
      auto pos = p::makeVector3(pid, pos_scs);
      // Next position and velocity
      float val[3], v2[3];
      for(int i=0; i<3; ++i) {
        val[i] = float(pos[i]) + float(vel[i]) * float(dTime);
        v2[i] = float(vel[i]);
        if(debug)
          printf(" %f", val[i]);
      }
      if(debug)
        printf("\n");

      for(int i=0; i<3; ++i) {      
        tgt_scs(pid, i) = val[i];
        vel_scs(pid, i) = v2[i];
      }
    
    }// mask
  };
  scs->parallel_for(boris, "neutralBorisMove");
} 

#endif //define

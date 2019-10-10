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
inline void gitrm_calculateE(GitrmParticles& gp, o::Mesh &mesh, bool debug=false) {
  auto biasedSurface = BIASED_SURFACE;
  auto angles = mesh.get_array<o::Real>(o::FACE, "angleBdryBfield");
  auto potentials = mesh.get_array<o::Real>(o::FACE, "potential");
  auto debyeLengths = mesh.get_array<o::Real>(o::FACE, "DebyeLength");
  auto larmorRadii = mesh.get_array<o::Real>(o::FACE, "LarmorRadius");
  auto childLangmuirDists = mesh.get_array<o::Real>(o::FACE, "ChildLangmuirDist");
  auto* scs = gp.scs;
  auto pos_scs = scs->get<PTCL_POS>();
  auto efield_scs  = scs->get<PTCL_EFIELD>();
  auto pid_scs = scs->get<PTCL_ID>();
  //NOTE arrays is based on pid, which is reset upon each rebuild
  const auto& closestPoints =  gp.closestPoints;
  const auto& faceIds = gp.closestBdryFaceIds;

  auto run = SCS_LAMBDA(const int &elem, const int &pid, const int &mask) { 
    if(mask >0) {
      auto faceId = faceIds[pid];
      if(faceId < 0) {
        //TODO check
        efield_scs(pid, 0) = 0;
        efield_scs(pid, 1) = 0;
        efield_scs(pid, 2) = 0;
      } else {
        // TODO angle is between surface normal and magnetic field at center of face
        // If  face is long, BField is not accurate. Calculate at closest point ?
        o::Real angle = angles[faceId];
        o::Real pot = potentials[faceId];
        o::Real debyeLength = debyeLengths[faceId];
        o::Real larmorRadius = larmorRadii[faceId];
        o::Real childLangmuirDist = childLangmuirDists[faceId];
        auto ptcl = pid_scs(pid);
        auto pos = p::makeVector3(pid, pos_scs);
        o::Vector<3> closest;
        for(o::LO i=0; i<3; ++i)
          closest[i] = closestPoints[pid*3+i];
        auto distVector = closest - pos;
        auto dirUnitVector = o::normalize(distVector);
        auto md = p::osh_mag(distVector);
        o::Real Emag = 0;
        /*
        // Calculate angle at closest point, instead of using it for centroid of a long tet
        //TODO what about other quantities ???
        */
        if(debug){
          printf("CalcE: ptcl %d dist2bdry %g  bdryface:%d angle:%g pot:%g"
           " Deb:%g Larm:%g CLD:%g \n", ptcl, md, faceId, angle, 
           pot, debyeLength, larmorRadius, childLangmuirDist);
          printf("ptcl %d pos %g %g %g closest %g %g %g distVec %g %g %g \n", 
            ptcl, pos[0], pos[1], pos[2], closest[0], closest[1], closest[2],
            distVector[0], distVector[1], distVector[2]);
        }
        if(biasedSurface) {
          Emag = pot/(2.0*childLangmuirDist)* exp(-md/(2.0*childLangmuirDist));
        } else { 
          o::Real fd = 0.98992 + 5.1220E-03 * angle - 7.0040E-04 * pow(angle,2.0) +
                       3.3591E-05 * pow(angle,3.0) - 8.2917E-07 * pow(angle,4.0) +
                       9.5856E-09 * pow(angle,5.0) - 4.2682E-11 * pow(angle,6.0);
          Emag = pot*(fd/(2.0 * debyeLength)* exp(-md/(2.0 * debyeLength))+ 
                  (1.0 - fd)/(larmorRadius)* exp(-md/larmorRadius));
        }
        if(isnan(Emag))
          Emag = 0;
        if(p::almost_equal(md, 0.0) || p::almost_equal(larmorRadius, 0.0)) {
          Emag = 0.0;
          dirUnitVector = o::zero_vector<3>();
        }
        auto exd = Emag*dirUnitVector;
        efield_scs(pid, 0) = exd[0];
        efield_scs(pid, 1) = exd[1];
        efield_scs(pid, 2) = exd[2];

        if(debug)
          printf("ptcl %d efield %g %g %g :d2bdry %g Emag %g "
            "pot %g CLD %g dir %g %g %g \n", 
            ptcl, exd[0], exd[1], exd[2], md, Emag, pot, childLangmuirDist, 
            dirUnitVector[0], dirUnitVector[1], dirUnitVector[2]);
        // 1st 2 particles used have same position in GITR and GITRm, since it is from file
        // Only difference is in input CLDist.

        //with 580K mesh of specified mesh size
        //0 dist2bdry 1e-06 angle:1.30716 pot:250 Deb:8.39096e-06 Larm:0.0170752 CLD:0.00031778 
        //1 dist2bdry 1e-06 angle:1.14724 pot:250 Deb:8.31213e-06 Larm:0.0201203 CLD:0.00024613 
        // 0 pos 0.0137135 -0.0183835 1e-06 closest 0.0137135 -0.0183835 0 distVec 0 3.46945e-18 -1e-06 
        // 1 pos -0.000247493 0.0197626 1e-06 closest -0.000247493 0.0197626 0 distVec -2.1684e-19 0 -1e-06 
        //0 E 0 1.36258e-06 -392736 :d2bdry 1e-06 Emag 392736 pot 250 CLD 0.00031778 dir 0 3.46945e-12 -1 
        // 1 E -1.09902e-07 0 -506832 :d2bdry 1e-06 Emag 506832 pot 250 CLD 0.00024613 dir -2.1684e-13 0 -1 

        // with large sized mesh with no size specified
        //GITR: E 0 0 -400917 Emax 400917 CLD 0.000311285 xyx 0.0137135 -0.0183835 1e-06 minD 1e-06 dir 0 0 -1 
        //GITRm E 6.78808e-07 0 -391306 :d2bdry 1e-06 Emag 391306 pot 250 CLD 0.000318942 dir 1.73472e-12 0 -1
        //GITRm:pos 0.0137135 -0.0183835 1e-06 closest 0.0137135 -0.0183835 0 distVec 1.73472e-18 0 -1e-06

        // GITR E 0 0 -573742 Emax 573742 CLD 0.000217368 xyx -0.000247493 0.0197626 1e-06 minD 1e-06 dir 0 0 -1
        //GITRm:pos -0.000247493 0.0197626 1e-06 closest -0.000247493 0.0197626 0 distVec 9.21572e-19 3.46945e-18 -1e-06
        //GITRm: E 3.3292e-07 1.25335e-06 -361253 :d2bdry 1e-06 Emag 361253 pot 250 CLD 0.000345518 dir 9.21572e-13 3.46945e-12 -1

      }
    }
  };
  scs->parallel_for(run);
}

// NOTE: for extruded mesh, TETs are too long that particle's projected position
// onto nearest poloidal plane is to be used to interpolate fields from 
// vertices of corresponding face of the TET. For a non-extruded mesh this projection
// is not possible and interpolation from all points of the TET is to be used.
// no input EFIELD from file in GITR, other than calculated near wall
inline void gitrm_borisMove(particle_structs::SellCSigma<Particle>* scs, 
  const GitrmMesh &gm, const o::Real dTime, bool debug=false,
  bool testExample=false) {
  o::Mesh &mesh = gm.mesh;  
  const auto coords = mesh.coords();
  const auto mesh2verts = mesh.ask_elem_verts();
  auto amu = PTCL_AMU;
  // Only 3D field from mesh tags
  auto use3dField = USE3D_BFIELD;
  auto use2dInputFields = USE2D_INPUTFIELDS;
  auto useConstantBField = USE_CONSTANT_BFIELD;

  if(PISCESRUN)
    OMEGA_H_CHECK(useConstantBField);
  o::Vector<3> bFieldConst; // At previous_pos
  if(useConstantBField)
    bFieldConst = p::makeVectorHost(CONSTANT_BFIELD);
  auto eFieldConst = p::makeVectorHost(CONSTANT_EFIELD);

  //const auto BField = o::Reals( mesh.get_array<o::Real>(o::VERT, "BField"));  // TODO FIXME replace this for non-pisces run
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
      OMEGA_H_CHECK((amu >0) && (dTime>0));
      o::Real bFieldMag = p::osh_mag(bField);
      o::Real qPrime = charge*1.60217662e-19/(amu*1.6737236e-27) *dTime*0.5;
      o::Real coeff = 2.0*qPrime/(1.0+(qPrime*bFieldMag)*(qPrime*bFieldMag));
      if(debug)
        printf("ptcl %d Bmag %g charge %d qPrime %g coeff %g eField %g %g %g \n",
         ptcl, bFieldMag, charge, qPrime, coeff, eField[0], eField[1], eField[2]);
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
      if(debug){
        printf("e %d ptcl %d vel %.1f %.1f %.1f \n pos %g %g %g => %g %g %g\n", 
          elem, ptcl, vel[0], vel[1], vel[2], pos[0], pos[1], pos[2],
          tgt[0], tgt[1], tgt[2]);
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
//auto ptcl = pid_scs(pid);
//if(ptcl==6222 || ptcl==6647) printf("BORIS: p %d e %d  pos %.15f %.15f %.15f\n", ptcl,elem, pos[0],pos[1],pos[2]);

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

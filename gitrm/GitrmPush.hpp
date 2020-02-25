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
inline void gitrm_calculateE(GitrmParticles& gp, o::Mesh &mesh, bool debug, 
    const GitrmMesh &gm) {

  const auto& f2rPtr = mesh.ask_up(o::FACE, o::REGION).a2ab;
  const auto& f2rElem = mesh.ask_up(o::FACE, o::REGION).ab2b;
  const auto coords = mesh.coords();
  const auto face_verts = mesh.ask_verts_of(2);

  const int compareWithGitr = USE_GITR_RND_NUMS;
  const double biasPot = BIAS_POTENTIAL;
  if(compareWithGitr)
    iTimePlusOne++;
  
  //const auto larmorRadius_d = gm.larmorRadius_d;
  //const auto childLangmuirDist_d = gm.childLangmuirDist_d;
  const auto iTimeStep = iTimePlusOne - 1;
  
  const auto biasedSurface = BIASED_SURFACE;
  const auto angles = mesh.get_array<o::Real>(o::FACE, "angleBdryBfield");
  const auto potentials = mesh.get_array<o::Real>(o::FACE, "potential");
  const auto debyeLengths = mesh.get_array<o::Real>(o::FACE, "DebyeLength");
  const auto larmorRadii = mesh.get_array<o::Real>(o::FACE, "LarmorRadius");
  const auto childLangmuirDists = mesh.get_array<o::Real>(o::FACE, "ChildLangmuirDist");
  const auto elDensity = mesh.get_array<o::Real>(o::FACE, "ElDensity");
  const auto elTemp = mesh.get_array<o::Real>(o::FACE, "ElTemp");
  auto* ptcls = gp.ptcls;
  auto pos_ps = ptcls->get<PTCL_POS>();
  auto efield_ps  = ptcls->get<PTCL_EFIELD>();
  auto pid_ps = ptcls->get<PTCL_ID>();

  auto charge_ps = ptcls->get<PTCL_CHARGE>();

  //NOTE arrays is based on pid, which is reset upon each rebuild
  const auto& closestPoints =  gp.closestPoints;
  const auto& faceIds = gp.closestBdryFaceIds;

  auto run = PS_LAMBDA(const int& elem, const int& pid, const int& mask) { 
    if(mask >0) {
      auto ptcl = pid_ps(pid);
      auto faceId = faceIds[pid];
      //if(faceId < 0) faceId = 0;
      if(faceId < 0) {
        efield_ps(pid, 0) = 0;
        efield_ps(pid, 1) = 0;
        efield_ps(pid, 2) = 0;
      } else {
        // TODO angle is between surface normal and magnetic field at center of face
        // If  face is long, BField is not accurate. Calculate at closest point ?
        auto angle = angles[faceId];
        auto pot = potentials[faceId];
        //TODO remove this after testing
        pot = biasPot;
        auto debyeLength = debyeLengths[faceId];
        auto larmorRadius = larmorRadii[faceId];
        auto childLangmuirDist = childLangmuirDists[faceId];
        auto pos = p::makeVector3(pid, pos_ps);
        
        o::Vector<3> closest;
        for(o::LO i=0; i<3; ++i)
          closest[i] = closestPoints[pid*3+i];
        
        //get Tel, Nel
        auto nelMesh = elDensity[faceId];
        auto telMesh = elTemp[faceId];
        auto bfel = p::elem_id_of_bdry_face_of_tet(faceId, f2rPtr, f2rElem);
        auto bface_coords = p::get_face_coords_of_tet(face_verts, coords, faceId);
        auto bmid = p::centroid_of_triangle(bface_coords);
        if(debug)
          printf("calcE0: ptcl %d ppos %.15e %.15e %.15e nelMesh %.15e TelMesh %.15e "
            "  bfidmid %.15e %.15e %.15e bfid %d bfel %d bface_verts %.15e %.15e %.15e ,"
            " %.15e %.15e %.15e, %.15e %.15e %.15e \n", 
            ptcl, pos[0], pos[1], pos[2], nelMesh, telMesh, bmid[0], bmid[1], bmid[2], 
            faceId, bfel, bface_coords[0][0], bface_coords[0][1], bface_coords[0][2],
            bface_coords[1][0], bface_coords[1][1], bface_coords[1][2],
            bface_coords[2][0], bface_coords[2][1], bface_coords[2][2]);

        auto distVector = closest - pos;
        auto dirUnitVector = o::normalize(distVector);
        auto d2bdry = o::norm(distVector);
        o::Real emag = 0;
        if(biasedSurface) {
          emag = pot/(2.0*childLangmuirDist)* exp(-d2bdry/(2.0*childLangmuirDist));
        } else { 
          o::Real fd = 0.98992 + 5.1220E-03 * angle - 7.0040E-04 * pow(angle,2.0) +
                       3.3591E-05 * pow(angle,3.0) - 8.2917E-07 * pow(angle,4.0) +
                       9.5856E-09 * pow(angle,5.0) - 4.2682E-11 * pow(angle,6.0);
          emag = pot*(fd/(2.0 * debyeLength)* exp(-d2bdry/(2.0 * debyeLength))+ 
                  (1.0 - fd)/(larmorRadius)* exp(-d2bdry/larmorRadius));
        }
        if(isnan(emag))
          emag = 0;
        if(debug)
          printf("calcE1- ptcl %d  distVec %.15e %.15e %.15e dirUnitVec %.15e %.15e %.15e\n",
           ptcl, distVector[0], distVector[1], distVector[2], dirUnitVector[0],
           dirUnitVector[1], dirUnitVector[2]);

        if(o::are_close(d2bdry, 0.0) || o::are_close(larmorRadius, 0.0)) {
          emag = 0.0;
          dirUnitVector[0] = dirUnitVector[1] = dirUnitVector[2] = 0;
        }
        auto exd = emag*dirUnitVector;
        for(int i=0; i<3; ++i)
          efield_ps(pid, i) = exd[i];

        if(debug){
          printf("calcE2: ptcl %d bdryface:%d  bfel %d emag %.15e "
              " pos %.15e %.15e %.15e closest %.15e %.15e %.15e distVec %.15e %.15e %.15e "
              " dirUnitVec %.15e %.15e %.15e \n", 
              ptcl, faceId, bfel, emag, pos[0], pos[1], pos[2],
              closest[0], closest[1], closest[2], distVector[0], distVector[1],
              distVector[2], dirUnitVector[0], dirUnitVector[1], dirUnitVector[2]);
        } 
        if(debug)
          printf("calcE_this:gitr ptcl %d timestep %d charge %d  dist2bdry %.15e"
             " CLD %.15e efield %.15e  %.15e  %.15e  CLD %.15e  Nel %.15e Tel %.15e \n", 
            ptcl, iTimeStep, charge_ps(pid), d2bdry, childLangmuirDist, 
            efield_ps(pid, 0), efield_ps(pid, 1), 
            efield_ps(pid, 2), childLangmuirDist, nelMesh, telMesh);
      } //faceId
    } //mask
  };
  ps::parallel_for(ptcls, run, "CalculateE");
}

inline void gitrm_borisMove(PS* ptcls, const GitrmMesh &gm, const o::Real dTime, 
  bool debug=false) {
  o::Mesh &mesh = gm.mesh;  
  const auto& coords = mesh.coords();
  const auto& mesh2verts = mesh.ask_elem_verts();
  auto amu = gitrm::PTCL_AMU;
  auto use3dField = USE3D_BFIELD;
  auto use2dInputFields = USE2D_INPUTFIELDS;
  auto useConstantBField = USE_CONSTANT_BFIELD;

  int iTimeStep = iTimePlusOne - 1;
  if(PISCESRUN)
    OMEGA_H_CHECK(useConstantBField);
  const auto BField = o::Reals(); //o::Reals(mesh.get_array<o::Real>(o::VERT, "BField"));
  const auto bX0 = gm.bGridX0;
  const auto bZ0 = gm.bGridZ0;
  const auto bDx = gm.bGridDx;
  const auto bDz = gm.bGridDz;
  const auto bGridNx = gm.bGridNx;
  const auto bGridNz = gm.bGridNz;
  const auto& BField_2d = gm.Bfield_2d;
  const auto eFieldConst_d = gitrm::getConstEField();
  //TODO crash using these variables
  const o::Real eQ = gitrm::ELECTRON_CHARGE;//1.60217662e-19;
  const o::Real pMass = gitrm::PROTON_MASS;//1.6737236e-27;
  auto pid_ps = ptcls->get<PTCL_ID>();
  auto tgt_ps = ptcls->get<PTCL_NEXT_POS>();
  auto efield_ps  = ptcls->get<PTCL_EFIELD>();
  auto pos_ps = ptcls->get<PTCL_POS>();
  auto vel_ps = ptcls->get<PTCL_VEL>();
  auto charge_ps = ptcls->get<PTCL_CHARGE>();
  auto boris = PS_LAMBDA(const int& elem, const int& pid, const int& mask) {
    if(mask >0) {
      auto ptcl = pid_ps(pid);
      auto vel = p::makeVector3(pid, vel_ps);
      auto pos = p::makeVector3(pid, pos_ps);
      auto charge = charge_ps(pid);
      auto bField_radial= o::zero_vector<3>();
      auto bField = o::zero_vector<3>();
      auto eField = p::makeVector3(pid, efield_ps);

      auto eField0 = o::zero_vector<3>();
      bool cylSymm = true;
      p::interp2dVector(eFieldConst_d, 0, 0, 0, 0, 1, 1, pos, eField0, cylSymm);
      eField += eField0;
      if(use3dField) {
        auto bcc = o::zero_vector<4>();
        p::findBCCoordsInTet(coords, mesh2verts, pos, elem, bcc);
        p::interpolate3dFieldTet(mesh2verts, BField, elem, bcc, bField);  
      } else if(useConstantBField || use2dInputFields) {
        p::interp2dVector(BField_2d, bX0, bZ0, bDx, bDz, bGridNx, bGridNz, pos,
          bField, cylSymm, &ptcl);
      }
      auto vel0 = vel;
      OMEGA_H_CHECK((amu >0) && (dTime>0));
      o::Real bFieldMag = o::norm(bField);
      //TODO crash replacing numbers
      o::Real qPrime = charge*1.60217662e-19/(amu* 1.6737236e-27) *dTime*0.5;
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
      auto vel_ = vel;
      vel = vel + qpE;
      
      if (false && ptcl==7)  {
        printf("Inside push the original position to which increment is t be made ptcl %d is %.15e %.15e %.15e\n",ptcl, pos[0],pos[1],pos[2]);
        printf("Inside push the velocity increment is to be made ptcl %d is %.15e %.15e %.15e\n", ptcl, vel[0],vel[1],vel[2]);
        printf("Magnetic field %.15e %.15e %.15e\n",bField[0], bField[1], bField[2]);
        printf("Electric field %.15e %.15e %.15e\n",eField[0], eField[1], eField[2]);
      
      }
      auto tgt = pos + vel * dTime;
      tgt_ps(pid, 0) = tgt[0];
      tgt_ps(pid, 1) = tgt[1];
      tgt_ps(pid, 2) = tgt[2];
      vel_ps(pid, 0) = vel[0];
      vel_ps(pid, 1) = vel[1];
      vel_ps(pid, 2) = vel[2];
      if(debug) {
        printf("Boris0 ptcl %d timestep %d eField %.15e %.15e %.15e bField %.15e %.15e %.15e "
          " qPrime %.15e coeff %.15e qpE %.15e %.15e %.15e vmxB %.15e %.15e %.15e "
          " qp_vmxB %.15e %.15e %.15e  v_prime %.15e %.15e %.15e vpxB %.15e %.15e %.15e "
          " c_vpxB %.15e %.15e %.15e  v_ %.15e %.15e %.15e\n", 
          ptcl, iTimeStep, eField[0], eField[1], eField[2], bField[0], bField[1], bField[2],  
          qPrime, coeff, qpE[0], qpE[1], qpE[2],vmxB[0], vmxB[1],vmxB[2], 
          qpVmxB[0], qpVmxB[1],  qpVmxB[2], vPrime[0], vPrime[1], vPrime[2] , 
          vpxB[0], vpxB[1], vpxB[2], cVpxB[0],cVpxB[1],cVpxB[2], vel_[0], vel_[1], vel_[2] );
      }

      if(debug) {
        printf("Boris1 ptcl %d timestep %d e %d charge %d pos %.15e %.15e %.15e =>  %.15e %.15e %.15e  "
          "vel %.15e %.15e %.15e =>  %.15e %.15e %.15e eField %.15e %.15e %.15e\n", ptcl, iTimeStep, 
          elem, charge, pos[0], pos[1], pos[2], tgt[0], tgt[1], tgt[2], 
          vel0[0], vel0[1], vel0[2], vel[0], vel[1], vel[2], eField[0], eField[1], eField[2]);
      }
    }// mask
  };
  ps::parallel_for(ptcls, boris, "BorisMove");
}

inline void neutralBorisMove(PS* ptcls,  const o::Real dTime) {
  auto vel_ps = ptcls->get<PTCL_VEL>();
  auto tgt_ps = ptcls->get<PTCL_NEXT_POS>();
  auto pos_ps = ptcls->get<PTCL_POS>();
  auto boris = PS_LAMBDA(const int& elem, const int& pid, const int& mask) {
    if(mask >0) {
      auto vel = p::makeVector3(pid, vel_ps);
      auto pos = p::makeVector3(pid, pos_ps);
      // Next position and velocity
      tgt_ps(pid, 0) = pos[0] + vel[0] * dTime;
      tgt_ps(pid, 1) = pos[1] + vel[1] * dTime;
      tgt_ps(pid, 2) = pos[2] + vel[2] * dTime;
      vel_ps(pid, 0) = vel[0];
      vel_ps(pid, 1) = vel[1];
      vel_ps(pid, 2) = vel[2];      
    }// mask
  };
  ps::parallel_for(ptcls, boris, "neutralBorisMove");
} 

inline void neutralBorisMove_float(PS* ptcls,  const o::Real dTime, bool debug = false) {
  auto vel_ps = ptcls->get<PTCL_VEL>();
  auto tgt_ps = ptcls->get<PTCL_NEXT_POS>();
  auto pos_ps = ptcls->get<PTCL_POS>();
  auto pid_ps = ptcls->get<PTCL_ID>();
  auto boris = PS_LAMBDA(const int& elem, const int& pid, const int& mask) {
    if(mask >0) {
      auto vel = p::makeVector3(pid, vel_ps);
      auto pos = p::makeVector3(pid, pos_ps);
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
        tgt_ps(pid, i) = val[i];
        vel_ps(pid, i) = v2[i];
      }
    
    }// mask
  };
  ps::parallel_for(ptcls, boris, "neutralBorisMove");
} 

#endif //define

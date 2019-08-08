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
  auto efield_scs  = scs->get<PTCL_EFIELD_PREV>();
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

        auto pos = p::makeVector3(pid, pos_scs);
        o::Vector<3> closest;
        for(o::LO i=0; i<3; ++i)
          closest[i] = closestPoints[pid*3+i];
        auto distVector = pos - closest; 
        auto dirUnitVector = o::normalize(distVector);
        auto md = p::osh_mag(distVector);
        o::Real Emag = 0;
        /*
        // Calculate angle at closest point, instead of using it for centroid of a long tet
        //TODO what about other quantities ???
        */
        if(debug){
          printf("CalcE: ptcl %d dist2bdry %g  bdryface:%d angle:%g pot:%g"
           " Deb:%g Larm:%g CLD:%g \n", pid_scs(pid), md, faceId, angle, 
           pot, debyeLength, larmorRadius, childLangmuirDist);
          p::print_osh_vector(pos, "ptclPos");
          p::print_osh_vector(closest, "calcE_closestPt");
          p::print_osh_vector(distVector, "calcE_distVector");
        }
        if(biasedSurface) {
          Emag = pot/(2.0*childLangmuirDist)* exp(-md/(2.0*childLangmuirDist));
          if(debug)
            printf("biased: Emag %g\n", Emag);
        } else { 
          o::Real fd = 0.98992 + 5.1220E-03 * angle - 7.0040E-04 * pow(angle,2.0) +
                       3.3591E-05 * pow(angle,3.0) - 8.2917E-07 * pow(angle,4.0) +
                       9.5856E-09 * pow(angle,5.0) - 4.2682E-11 * pow(angle,6.0);
          Emag = pot*(fd/(2.0 * debyeLength)* exp(-md/(2.0 * debyeLength))+ 
                  (1.0 - fd)/(larmorRadius)* exp(-md/larmorRadius));
          if(debug)
            printf("Non-biased: fd,Emag,md %g %g %g\n", fd, Emag, md);
        }
        if(std::isnan(Emag))
          Emag = 0;

        if(p::almost_equal(md, 0.0) || p::almost_equal(larmorRadius, 0.0)) {
          Emag = 0.0;
          dirUnitVector = o::zero_vector<3>();
        }
        auto exd = Emag*dirUnitVector;
        efield_scs(pid, 0) = exd[0];
        efield_scs(pid, 1) = exd[1];
        efield_scs(pid, 2) = exd[2];

        if(debug) {
              printf("efield %g %g %g :d2bdry %g \n", exd[0], exd[1], exd[2], md);
        }
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
  const GitrmMesh &gm, const o::Real dTime, bool debug=false) {
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

  const auto BField = o::Reals( mesh.get_array<o::Real>(o::VERT, "BField"));
  // Only if used 2D field read from file
  auto shiftB = gm.mesh2Bfield2Dshift;
  o::Real bxz[] = {gm.bGridX0, gm.bGridZ0, gm.bGridDx, gm.bGridDz};
  auto bGridNx = gm.bGridNx;
  auto bGridNz = gm.bGridNz;
  const auto &BField_2d = gm.Bfield_2d;
  auto pid_scs = scs->get<PTCL_ID>();
  auto pos_scs = scs->get<PTCL_POS>();
  auto efield_scs  = scs->get<PTCL_EFIELD_PREV>();
  auto prev_pos_scs = scs->get<PTCL_POS_PREV>();
  auto vel_scs = scs->get<PTCL_VEL>();
  auto charge_scs = scs->get<PTCL_CHARGE>();

  auto boris = SCS_LAMBDA(const int& elem, const int& pid, const int& mask) {
    if(mask >0) {
      auto ptcl = pid_scs(pid);
      auto vel = p::makeVector3(pid, vel_scs); //at current_pos
      auto posPrev = p::makeVector3(pid, prev_pos_scs);
      auto charge = charge_scs(pid);
      auto bField = o::zero_vector<3>(); //At previous_pos
      // for neutral tracking skip gitrm_calculateE()
      auto eField = p::makeVector3(pid, efield_scs); //at previous_pos
      // In GITR only constant EField is used
      eField += eFieldConst;
      
      if(useConstantBField) {
        bField = bFieldConst;
      } else if(use3dField) {
        auto bcc = o::zero_vector<4>();
        p::findBCCoordsInTet(coords, mesh2verts, posPrev, elem, bcc);
        // BField is 3 component array
        p::interpolate3dFieldTet(mesh2verts, BField, elem, bcc, bField);//At previous_pos      
      } else if(use2dInputFields) { //TODO for testing
        auto pos = o::zero_vector<3>();
        //cylindrical symmetry, height (z) is same.
        auto rad = sqrt(posPrev[0]*posPrev[0] + posPrev[1]*posPrev[1]);
        // projecting point to y=0 plane, since 2D data is on const-y plane.
        pos[2] = posPrev[2];
        pos[0] = rad + shiftB; // D3D 1.6955m.
        p::interp2dVector(BField_2d,  bxz[0], bxz[1], bxz[2], bxz[3], bGridNx,
          bGridNz, pos, bField, false); //At previous_pos
      }

      OMEGA_H_CHECK(amu >0 && dTime>0);
      o::Real bFieldMag = p::osh_mag(bField);
      o::Real qPrime = charge*1.60217662e-19/(amu*1.6737236e-27) *dTime*0.5;
      o::Real coeff = 2.0*qPrime/(1.0+(qPrime*bFieldMag)*(qPrime*bFieldMag));
      if(debug)
        printf("ptcl %d Bmag %g charge %d qPrime %g coeff %g \n", ptcl,
          bFieldMag, charge, qPrime, coeff );
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

      // Update particle data
      prev_pos_scs(pid, 0) = pos_scs(pid, 0);
      prev_pos_scs(pid, 1) = pos_scs(pid, 1);
      prev_pos_scs(pid, 2) = pos_scs(pid, 2);

      // Next position and velocity
      pos_scs(pid, 0) = posPrev[0] + vel[0] * dTime;
      pos_scs(pid, 1) = posPrev[1] + vel[1] * dTime;
      pos_scs(pid, 2) = posPrev[2] + vel[2] * dTime;
      vel_scs(pid, 0) = vel[0];
      vel_scs(pid, 1) = vel[1];
      vel_scs(pid, 2) = vel[2];
      
      if(debug){
        printf("e %d ptcl %d vel %.1f %.1f %.1f \n pos %g %g %g => %g %g %g \n", 
          elem, ptcl, vel[0], vel[1], vel[2], prev_pos_scs(pid, 0), prev_pos_scs(pid, 1),
          prev_pos_scs(pid, 2), pos_scs(pid, 0), pos_scs(pid, 1), pos_scs(pid, 2));
      }
    }// mask
  };
  scs->parallel_for(boris);
}

#endif //define

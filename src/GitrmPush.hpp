#ifndef GITRM_PUSH_HPP
#define GITRM_PUSH_HPP

#include "GitrmMesh.hpp"
#include "GitrmParticles.hpp" 

#include <SellCSigma.h>
#include <SCS_Macros.h>
#include <Kokkos_Core.hpp>  //direct use


#include <iostream>
#include <cmath>
#include <utility>

#include "Omega_h_adj.hpp"
#include "Omega_h_array_ops.hpp"
#include "Omega_h_scalar.hpp" //divide
#include "Omega_h_fail.hpp"

#include "pumipic_utils.hpp"
#include "pumipic_constants.hpp"


// Angle, DebyeLength etc were calculated at center of LONG tet, using BField.
//  Does BField vary along boundary ?
inline void gitrm_calculateE(particle_structs::SellCSigma<Particle>* scs, 
  o::Mesh &mesh) {

  const auto angles = o::Reals(mesh.get_array<o::Real>(o::FACE, "angleBdryBfield"));
  const auto potentials = o::Reals(mesh.get_array<o::Real>(o::FACE, "potential"));
  const auto debyeLengths = o::Reals(mesh.get_array<o::Real>(o::FACE, "DebyeLength"));
  const auto larmorRadii = o::Reals(mesh.get_array<o::Real>(o::FACE, "LarmorRadius"));
  const auto childLangmuirDists = o::Reals(mesh.get_array<o::Real>(
                                  o::FACE, "ChildLangmuirDist"));

  auto ptclPos_d = scs->get<PTCL_POS>();
  auto closestPoint_d = scs->get<PTCL_BDRY_CLOSEPT>();
  auto faceId_d = scs->get<PTCL_BDRY_FACEID>();
  auto efield_d  = scs->get<PTCL_EFIELD_PREV>();

  auto run = SCS_LAMBDA(const int &elem, const int &pid,
                                const int &mask) { 
    o::LO verbose = (elem %100 ==0)?3:0;

    auto faceId = faceId_d(pid);
    if(faceId < 0) {
      //TODO check
      efield_d(pid, 0) = 0;
      efield_d(pid, 1) = 0;
      efield_d(pid, 2) = 0;
      return;
    }

    // TODO angle is between surface normal and magnetic field at center of face
    // If  face is long, BField is not accurate. Calculate at closest point ?
    o::Real angle = angles[faceId];
    o::Real pot = potentials[faceId];
    o::Real debyeLength = debyeLengths[faceId];
    o::Real larmorRadius = larmorRadii[faceId];
    o::Real childLangmuirDist = childLangmuirDists[faceId];

    o::Vector<3> pos{ptclPos_d(pid,0), ptclPos_d(pid,1), ptclPos_d(pid,2)};
    o::Vector<3> closest{closestPoint_d(pid,0), closestPoint_d(pid,1), 
                         closestPoint_d(pid,2)};
    o::Vector<3> distVector = pos - closest; 
    o::Vector<3> dirUnitVector = o::normalize(distVector);
    o::Real md = p::osh_mag(distVector);
    o::Real Emag = 0;

    /*
    // Calculate angle at closest point, instead of using it for centroid of a long tet
    //TODO what about other quantities ????????????????
    */
    if(verbose >4){
      printf("CalcE: %.5f %.5f %.5f %.5f %.5f \n", angle, pot, debyeLength, 
        larmorRadius, childLangmuirDist);
      p::print_osh_vector(pos);
      p::print_osh_vector(closest);
      p::print_osh_vector(distVector);
    }

    if(BIASED_SURFACE) {
      Emag = pot/(2.0*childLangmuirDist)* exp(-md/(2.0*childLangmuirDist));
    }
    else { 
      o::Real fd = 0.98992 + 5.1220E-03 * angle - 7.0040E-04 * pow(angle,2.0) +
                   3.3591E-05 * pow(angle,3.0) - 8.2917E-07 * pow(angle,4.0) +
                   9.5856E-09 * pow(angle,5.0) - 4.2682E-11 * pow(angle,6.0);
      
      Emag = pot*(fd/(2.0 * debyeLength)* exp(-md/(2.0 * debyeLength))+ 
              (1.0 - fd)/(larmorRadius)* exp(-md/larmorRadius));

      if(verbose >4)
        printf("fd,Emag,md %.5f %.5f %.5f\n", fd, Emag, md);
    }


    if(p::almost_equal(md, 0.0) || p::almost_equal(larmorRadius, 0.0)) {
      Emag = 0.0;
      dirUnitVector = {0, 0, 0}; //TODO confirm
    }
    auto exd = Emag*dirUnitVector;
    efield_d(pid, 0) = exd[0];
    efield_d(pid, 1) = exd[1];
    efield_d(pid, 2) = exd[2];

    if(verbose >2) {
          printf("efield %.5f %.5f %.5f \n", exd[0], exd[1], exd[2]);
    }

  };
  scs->parallel_for(run);
}

// NOTE: for extruded mesh, TETs are too long that particle's projected position
// onto nearest poloidal plane is to be used to interpolate fields from 
// vertices of corresponding face of the TET. For a non-extruded mesh this projection
// is not possible and interpolation from all points of the TET is to be used.

inline void gitrm_borisMove(particle_structs::SellCSigma<Particle>* scs, 
  o::Mesh &mesh, const GitrmMesh &gm, const o::Real dtime) {

  const auto coords = mesh.coords();
  const auto mesh2verts = mesh.ask_elem_verts();


  // Only 3D field from mesh tags
  const o::LO USE_3D_FIELD = USE_3D_FIELDS;
  const auto BField = o::Reals( mesh.get_array<o::Real>(o::VERT, "BField"));
  const auto EField = o::Reals( mesh.get_array<o::Real>(o::VERT, "EField"));

  // Only if used 2D field read from file
  o::Real  exz[] = {gm.EGRIDX0, gm.EGRIDZ0, gm.EGRID_DX, gm.EGRID_DZ};
  o::Real  bxz[] = {gm.BGRIDX0, gm.BGRIDZ0, gm.BGRID_DX, gm.BGRID_DZ};
  o::LO ebn[] = {gm.EGRID_NX, gm.EGRID_NZ, gm.BGRID_NX, gm.BGRID_NZ};
  const auto &EField_2d = gm.Efield_2d;
  const auto &BField_2d = gm.Bfield_2d;

  auto ptclPos_d = scs->get<PTCL_POS>();
  auto efield_d  = scs->get<PTCL_EFIELD_PREV>();
  auto ptclPrevPos_d = scs->get<PTCL_POS_PREV>();
  auto vel_d = scs->get<PTCL_VEL>();

  auto boris = SCS_LAMBDA(const int &elem, const int &pid, const int &mask) {
    o::LO verbose = 4;//(elem%50==0)?4:0;
    
    o::Vector<3> vel{vel_d(pid,0), vel_d(pid,1), vel_d(pid,2)};  //at current_pos
    o::Vector<3> eField{efield_d(pid,0), efield_d(pid,1),efield_d(pid,2)}; //at previous_pos
    o::Vector<3> posPrev{ptclPrevPos_d(pid,0), ptclPrevPos_d(pid,1), ptclPrevPos_d(pid,2)};
    o::Vector<3> bField; //At previous_pos

    if(verbose >3) {
      printf("vel_current: %.3f %.3f %.3f \n", vel[0], vel[1], vel[2]);
    }
    
    o::Vector<4> bcc{{0}};
    if(USE_3D_FIELD) {
      p::findBCCoordsInTet(coords, mesh2verts, posPrev, elem, bcc);
    }

    if(USEPRESHEATHEFIELD) {
      o::Vector<3> psheathE;

      if(USE_3D_FIELD) { // for 2D field TODO
        p::interpolate3dFieldTet(EField, bcc, elem, psheathE);//At previous_pos
      } else {
        p::interp2dVector(EField_2d, exz[0], exz[1], exz[2], exz[3], ebn[0], ebn[1],
          posPrev, psheathE, true);
      }

      eField = eField + psheathE;
    }

    // BField is 3 component array
    if(USE_3D_FIELD) {
      p::interpolate3dFieldTet(BField, bcc, elem, bField);//At previous_pos      
    } else {
      p::interp2dVector(BField_2d,  bxz[0], bxz[1], bxz[2], bxz[3], ebn[2], ebn[3], 
         posPrev, bField, true); //At previous_pos
    }

//TODO
    o::Real charge = 1; //TODO get using speciesID using enum
    o::Real amu = 184.0; //TODO //impurity_amu = 184.0

    OMEGA_H_CHECK(amu >0 && dtime>0); //TODO dtime
    o::Real bFieldMag = p::osh_mag(bField);
    o::Real qPrime = charge*1.60217662e-19/(amu*1.6737236e-27) *dtime*0.5;
    o::Real coeff = 2.0*qPrime/(1.0+(qPrime*bFieldMag)*(qPrime*bFieldMag));

    //v_minus = v + q_prime*E;
    o::Vector<3> qpE = qPrime*eField;
    o::Vector<3> vMinus = vel - qpE;

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
    o::Vector<3> pre {ptclPrevPos_d(pid, 0), ptclPrevPos_d(pid, 1), 
                              ptclPrevPos_d(pid, 2)}; //prev pos
    ptclPrevPos_d(pid, 0) = ptclPos_d(pid, 0);
    ptclPrevPos_d(pid, 1) = ptclPos_d(pid, 1);
    ptclPrevPos_d(pid, 2) = ptclPos_d(pid, 2);

    // Next position and velocity
    ptclPos_d(pid, 0) = pre[0] + vel[0] * dtime;
    ptclPos_d(pid, 1) = pre[1] + vel[1] * dtime;
    ptclPos_d(pid, 2) = pre[2] + vel[2] * dtime;
    vel_d(pid, 0) = vel[0];
    vel_d(pid, 1) = vel[1];
    vel_d(pid, 2) = vel[2];
    
    if(verbose >3){
      printf(" e,p: %d %d :: pos: %.3f %.3f %.3f ::", elem, pid, ptclPos_d(pid, 0), 
        ptclPos_d(pid, 1), ptclPos_d(pid, 2));
      printf("prev_pos: %.3f %.3f %.3f ::", pre[0], pre[1], pre[2]);
      //printf("vel_next: %.3f %.3f %.3f \n", vel[0], vel[1], vel[2]);
    }
  };

  scs->parallel_for(boris);
}

#endif //define

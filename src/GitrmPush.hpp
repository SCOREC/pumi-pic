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



inline void gitrm_getE(particle_structs::SellCSigma<Particle>* scs, 
  const o::Mesh &mesh) {

  const auto angles = o::Reals( mesh.get_array<o::Real>(o::FACE, "angleBdryBfield"));
  const auto potentials = o::Reals(mesh.get_array<o::Real>(o::FACE, "potential"));
  const auto debyeLengths = o::Reals(mesh.get_array<o::Real>(o::FACE, "DebyeLength"));
  const auto larmorRadii = o::Reals(mesh.get_array<o::Real>(o::FACE, "LarmorRadius"));
  const auto childLangmuirDists = o::Reals(mesh.get_array<o::Real>(
                                  o::FACE, "ChildLangmuirDist"));

  scs->transferToDevice();
  kkFp3View ptclPos_d("ptclPos_d", scs->offsets[scs->num_slices]);
  hostToDeviceFp(ptclPos_d, scs->getSCS<PCL_POS>());
  
  kkFp3View closestPoint_d("closestPoint_d", scs->offsets[scs->num_slices]);
  hostToDeviceFp(closestPoint_d, scs->getSCS<PCL_BDRY_CLOSEPT>()); 

  kkLidView faceId_d("faceId_d", scs->offsets[scs->num_slices]);
  hostToDeviceLid(faceId_d, scs->getSCS<PCL_BDRY_FACEID>());
  
  kkFp3View efield_d("efield_d", scs->offsets[scs->num_slices]);
  hostToDeviceFp(efield_d, scs->getSCS<PCL_EFIELD_PREV>());

  auto run = SCS_LAMBDA(const int &elem, const int &pid,
                                const int &mask) { 
    o::LO verbose = (elem%1000==0)?3:0;

    auto faceId = faceId_d[pid];
    if(faceId < 0) {
      //TODO check
      efield_d(pid, 0) = 0;
      efield_d(pid, 1) = 0;
      efield_d(pid, 2) = 0;
      return;
    }

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

    if(BIASED_SURFACE) {
      Emag = pot/(2.0*childLangmuirDist)*
              exp(-md/(2.0*childLangmuirDist));
    }
    else { 
      o::Real fd = 0.98992 + 5.1220E-03 * angle - 7.0040E-04 * pow(angle,2.0) +
                   3.3591E-05 * pow(angle,3.0) - 8.2917E-07 * pow(angle,4.0) +
                   9.5856E-09 * pow(angle,5.0) - 4.2682E-11 * pow(angle,6.0);
      
      Emag = pot*(fd/(2.0 * debyeLength)* exp(-md/(2.0 * debyeLength))+ 
              (1.0 - fd)/(larmorRadius)* exp(-md/larmorRadius));
    }


    if(p::almost_equal(md, 0.0) || p::almost_equal(larmorRadius, 0.0)) {
      Emag = 0.0;
      dirUnitVector = {0, 0, 0}; //TODO confirm
    }
    auto exd = Emag*dirUnitVector;
    efield_d(pid, 0) = exd[0];
    efield_d(pid, 1) = exd[1];
    efield_d(pid, 2) = exd[2];

    if(verbose >2)
     printf("efield %.5f %.5f %.5f \n", efield_d(pid, 0), efield_d(pid, 1), efield_d(pid, 2));

  };
  scs->parallel_for(run);

  deviceToHostFp(efield_d, scs->getSCS<PCL_EFIELD_PREV>());
}


/** @brief Re-writing of interp2dCombined() in GITR
 *  @see https://github.com/ORNL-Fusion/GITR/blob/master/src/interp2d.cpp
 *  @param[in]  data, flat 3component array
 *  @param[in] comp, component, from degree of freedom
 *  @return value corresponding to comp
 */
OMEGA_H_INLINE o::Real interpolate2dField(const o::Reals &data, const o::LO comp, 
  const o::Real gridx0, const o::Real gridz0, const o::Real dx, const o::Real dz, 
  const o::LO nx, const o::LO nz, const o::Vector<3> &pos) {
  
  if(nx*nz == 1)
  {
    return data[comp+0];
  }

  o::Real x = pos[0];
  o::Real y = pos[1];
  o::Real z = pos[2];   

  o::Real fxz = 0;
  o::Real fx_z1 = 0;
  o::Real fx_z2 = 0; 
  o::Real dim1 = 0;

  if(USECYLSYMM > 0)
    dim1 = sqrt(x*x + y*y);
  else
    dim1 = x;
  
  o::LO i = floor((dim1 - gridx0)/dx);
  o::LO j = floor((z - gridz0)/dz);
  
  if (i < 0) i=0;
  if (j < 0) j=0;

  o::Real gridxi = gridx0 + i * dx;
  o::Real gridxip1 = gridx0 + (i+1) * dx;    
  o::Real gridzj = gridz0 + j * dz;
  o::Real gridzjp1 = gridz0 + (j+1) * dz; 

  if (i >=nx-1 && j>=nz-1) {
      fxz = data[(nx-1+(nz-1)*nx)*3+comp];  //TODO this is wrong, include comp
  }
  else if (i >=nx-1) {
      fx_z1 = data[(nx-1+j*nx)*3+comp];
      fx_z2 = data[(nx-1+(j+1)*nx)*3+comp];
      fxz = ((gridzjp1-z)*fx_z1+(z - gridzj)*fx_z2)/dz;
  }
  else if (j >=nz-1) {
      fx_z1 = data[(i+(nz-1)*nx)*3+comp];
      fx_z2 = data[(i+(nz-1)*nx)*3+comp];
      fxz = ((gridxip1-dim1)*fx_z1+(dim1 - gridxi)*fx_z2)/dx;
      
  }
  else {
    fx_z1 = ((gridxip1-dim1)*data[(i+j*nx)*3+comp] + 
            (dim1 - gridxi)*data[(i+1+j*nx)*3+comp])/dx;
    fx_z2 = ((gridxip1-dim1)*data[(i+(j+1)*nx)*3+comp] + 
            (dim1 - gridxi)*data[(i+1+(j+1)*nx)*3+comp])/dx; 
    fxz = ((gridzjp1-z)*fx_z1+(z - gridzj)*fx_z2)/dz;
  }
  
  return fxz;
}

OMEGA_H_INLINE void interp2dVector (const o::Reals &data3, o::Real gridx0, 
  o::Real gridz0, o::Real dx, o::Real dz, int nx, int nz,
  const o::Vector<3> &pos, o::Vector<3> &field) {

  o::Real Ar = interpolate2dField(data3, 0, gridx0, gridz0, dx, dz, nx, nz, pos);
  o::Real At = interpolate2dField(data3, 1, gridx0, gridz0, dx, dz, nx, nz, pos);
  field[2] = interpolate2dField(data3, 2, gridx0, gridz0, dx, dz, nx, nz, pos);
  if(USECYLSYMM > 0) {
    o::Real theta = atan2(pos[1], pos[0]);   
    field[0] = cos(theta)*Ar - sin(theta)*At;
    field[1] = sin(theta)*Ar + cos(theta)*At;
  }
  else {
    field[0] = Ar;
    field[1] = At;
  }
}


inline void gitrm_borisMove(particle_structs::SellCSigma<Particle>* scs, 
  const o::Mesh &mesh, const o::Real dtime) {

  const auto BField = o::Reals( mesh.get_array<o::Real>(o::VERT, "BField"));

  scs->transferToDevice();
  kkFp3View efield_d("efield_d", scs->offsets[scs->num_slices]);
  hostToDeviceFp(efield_d, scs->getSCS<PCL_EFIELD_PREV>());
  kkFp3View vel_d("vel_d", scs->offsets[scs->num_slices]);
  hostToDeviceFp(vel_d, scs->getSCS<PCL_VEL>());
  kkFp3View ptclPrevPos_d("ptclPrevPos_d", scs->offsets[scs->num_slices]);
  hostToDeviceFp(ptclPrevPos_d, scs->getSCS<PCL_POS_PREV>());
  kkFp3View ptclPos_d("ptclPos_d", scs->offsets[scs->num_slices]);
  hostToDeviceFp(ptclPos_d, scs->getSCS<PCL_POS>());

  auto boris = SCS_LAMBDA(const int &elem, const int &pid, const int &mask) {
    o::LO verbose = (elem%1000==0)?3:0;

    //TODO check
    o::Vector<3> vel{vel_d(pid,0), vel_d(pid,1), vel_d(pid,2)};  //at current_pos
    o::Vector<3> eField{efield_d(pid,0), efield_d(pid,1),efield_d(pid,2)}; //at previous_pos
    o::Vector<3> posPrev{ptclPrevPos_d(pid,0), ptclPrevPos_d(pid,1), ptclPrevPos_d(pid,2)};
    o::Vector<3> bField; //At previous_pos
    //TODO check BField shape
    interp2dVector(BField, BGRIDX0, BGRIDZ0, BGRID_DX, BGRID_DZ, BGRID_NX, BGRID_NZ, 
                    posPrev, bField); //At previous_pos

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

    //write
    o::Vector<3> pre = {ptclPrevPos_d(pid, 0), ptclPrevPos_d(pid, 1), 
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
    
    if(verbose >2){
      printf("prev_pos: %.3f %.3f %.3f :: ", ptclPrevPos_d(pid, 0), ptclPrevPos_d(pid, 1), 
        ptclPrevPos_d(pid, 2));
      printf("pre: %.3f %.3f %.3f \n", pre[0], pre[1], pre[2]);
      printf("pos: %.3f %.3f %.3f ::", ptclPos_d(pid, 0), ptclPos_d(pid, 1), ptclPos_d(pid, 2));
      printf("vel: %.3f %.3f %.3f \n", vel_d(pid, 0), vel_d(pid, 1), vel_d(pid, 2));
    }
  };

  scs->parallel_for(boris);
  deviceToHostFp(ptclPos_d, scs->getSCS<PCL_POS>());
  deviceToHostFp(ptclPrevPos_d, scs->getSCS<PCL_POS_PREV>());
  deviceToHostFp(vel_d, scs->getSCS<PCL_VEL>());
}



#endif //define
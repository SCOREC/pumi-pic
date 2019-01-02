#ifndef GITRM_PUSH_HPP_INCLUDED
#define GITRM_PUSH_HPP_INCLUDED

#include <iostream>
#include <cmath>
#include <utility>

#include "Omega_h_adj.hpp"
#include "Omega_h_array_ops.hpp"
#include "Omega_h_scalar.hpp" //divide
#include "Omega_h_fail.hpp" //assert

#include "gitrm_utils.hpp"
#include "gitrmPtcls.hpp" //NOT YET THERE !

//#define DEBUG 1

namespace GITRm
{

OMEGA_H_INLINE void doPushBoris( gitrmPtcls &ptcls, Omega_h::Write<Omega_h::Real> &bcc, Omega_h::LO elId)
{
// Fields ?
//B Fields ; br -0.1 to 0.3; bt = -3 to 3; bz= -1 to 1.5
// E = 2x10^11 V/m

}

// Option to select various EFields removed, since it can be done while writing them in
// previous step, in the search routine.
OMEGA_H_INLINE void pushBoris( gitrmPtcls &ptcls, Omega_h::Real timeSpan)
{
  Omega_h::Real dt = timeSpan;

  // TODO check if float is good, if so use it in other code as well
  Omega_h::Vector<3> eField{0.0f, 0.0f, 0.0f};
  Omega_h::Vector<3> bField{0.0f,0.0f,0.0f};

  Omega_h::LO nsteps = std::floor(timeSpan/dt + 0.5f);
  //previous position
  Omega_h::Vector<3> pos = {ptcls.xpre[indx], ptcls.ypre[indx], ptcls.zpre[indx]};

  for (int s=0; s<nsteps; s++ )
  {
    // TODO OMEGA_H_CHECK(ptcls.eFieldsPre.size() > index); indx and array_size

    for(Omega_h::LO i=0; i<3; ++i)
    {
      eField[i] = ptcls.eFieldsPre[indx+i];
    }

    for(Omega_h::LO i=0; i<3; ++i)
    {
      bField[i] = ptcls.bFieldsPre[indx+i];
    }

    Omega_h::Real bFieldMag = Omega_h::normalize(bField);
    Omega_h::Real qPrime = ptcls.charge[indx]*1.60217662e-19f/(ptcls.amu[indx]*1.6737236e-27f)*dt*0.5f;
    Omega_h::Real coeff = 2.0f*qPrime/(1.0f+(qPrime*bFieldMag)*(qPrime*bFieldMag));
    Omega_h::Vector<3> vel = {ptcls.vx[indx], ptcls.vy[indx], ptcls.vz[indx]};

    //v_minus = v + q_prime*E;
    Omega_h::Vector<3> qpE = dot(qPrime,eField);
    Omega_h::Vector<3> vMinus = vel - qpE;
    for(Omega_h::LO i=0; i<3; ++i)
      electricForce[i] = 2.0*qpE[i];  //?

   //v_prime = v_minus + q_prime*(v_minus x B)
    Omega_h::Vector<3> vmxB = Omega_h::cross(vMinus,bField);
    Omega_h::Vector<3> qpVmxB = dot(qPrime,vmxB);
    Omega_h::Vector<3> vPrime = vMinus + qpVmxB;
    for(Omega_h::LO i=0; i<3; ++i)
      magneticForce[i] = qpVmxB[i];  //?

    //v = v_minus + coeff*(v_prime x B)
    Omega_h::Vector<3> vpxB = Omega_h::cross(vPrime, bField);
    Omega_h::Vector<3> cVpxB = dot(coeff,vpxB);
    vel = vMinus + cVpxB;

    //v = v + q_prime*E
    vel = vel + qpE;
    int thisTmp = (ptcls.vx[indx]< 10.0 && vel[0] > 4.0e3) ?1:0; //?

    if(ptcls.hitWall[indx] == 0.0)
    {
      ptcls.x[indx] = pos[0] + vel[0] * dt;
      ptcls.y[indx] = pos[1] + vel[1] * dt;
      ptcls.z[indx] = pos[2] + vel[2] * dt;
      ptcls.vx[indx] = vel[0];
      ptcls.vy[indx] = vel[1];
      ptcls.vz[indx] = vel[2];
    }
  } //nsteps
}

// TODO add RK4 method



} //namespace
#endif // GITRM_PUSH_HPP_INCLUDED

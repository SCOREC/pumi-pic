#ifndef PUMIPIC_PUSH_HPP_INCLUDED
#define PUMIPIC_PUSH_HPP_INCLUDED

#include <iostream>
#include <cmath>
#include <utility>

#include "Omega_h_adj.hpp"
#include "Omega_h_array_ops.hpp"
#include "Omega_h_scalar.hpp" //divide
#include "Omega_h_fail.hpp"

#include "pumipic_utils.hpp"
#include "pumipic_constants.hpp"

namespace pumipic
{
OMEGA_H_INLINE void pushBoris(Omega_h::LO nelems, Omega_h::Write<Omega_h::Real> &x,
 Omega_h::Write<Omega_h::Real> &y, Omega_h::Write<Omega_h::Real> &z, Omega_h::Write<Omega_h::Real> &xp,
 Omega_h::Write<Omega_h::Real> &yp, Omega_h::Write<Omega_h::Real> &zp, Omega_h::Write<Omega_h::Real> &vx,
 Omega_h::Write<Omega_h::Real> &vy, Omega_h::Write<Omega_h::Real> &vz, Omega_h::Write<Omega_h::Real> &eFld0x,
 Omega_h::Write<Omega_h::Real> &eFld0y, Omega_h::Write<Omega_h::Real> &eFld0z,
 Omega_h::Write<Omega_h::Real> &bFld0r, Omega_h::Write<Omega_h::Real> &bFld0t,
 Omega_h::Write<Omega_h::Real> &bFld0z, Omega_h::Write<Omega_h::LO> &part_flags, Omega_h::Real dt)
{
  auto pushPtcl = OMEGA_H_LAMBDA( Omega_h::LO ielem)
  {
    Omega_h::LO pid = ielem; //TODO Replace by ptcl loop

    Omega_h::Vector<3> vel{vx[pid], vy[pid], vz[pid]}; //current
    Omega_h::Vector<3> eField{eFld0x[pid], eFld0y[pid], eFld0z[pid]}; //previous
    Omega_h::Vector<3> bField{bFld0r[pid], bFld0t[pid], bFld0z[pid]}; //previous

    Omega_h::Real charge = 1; //TODO get using speciesID using enum
    Omega_h::Real amu = 10; //TODO "

    OMEGA_H_CHECK(amu >0 && dt>0);
    Omega_h::Real bFieldMag = osh_mag(bField);
    Omega_h::Real qPrime = charge*1.60217662e-19/(amu*1.6737236e-27) *dt*0.5;
    Omega_h::Real coeff = 2.0*qPrime/(1.0+(qPrime*bFieldMag)*(qPrime*bFieldMag));

      //v_minus = v + q_prime*E;
    Omega_h::Vector<3> qpE = qPrime*eField;
    Omega_h::Vector<3> vMinus = vel - qpE;

    //v_prime = v_minus + q_prime*(v_minus x B)
    Omega_h::Vector<3> vmxB = Omega_h::cross(vMinus,bField);
    Omega_h::Vector<3> qpVmxB = qPrime*vmxB;
    Omega_h::Vector<3> vPrime = vMinus + qpVmxB;

    //v = v_minus + coeff*(v_prime x B)
    Omega_h::Vector<3> vpxB = Omega_h::cross(vPrime, bField);
    Omega_h::Vector<3> cVpxB = coeff*vpxB;
    vel = vMinus + cVpxB;

    //v = v + q_prime*E
    vel = vel + qpE;

    //write
    Omega_h::Vector<3> pre = {xp[pid], yp[pid], zp[pid]}; //prev pos
    xp[pid] = x[pid];
    yp[pid] = y[pid];
    zp[pid] = z[pid];

    // Next position and velocity
    x[pid] = pre[0] + vel[0] * dt;
    y[pid] = pre[1] + vel[1] * dt;
    z[pid] = pre[2] + vel[2] * dt;
    vx[pid] = vel[0];
    vy[pid] = vel[1];
    vz[pid] = vel[2];
  };

  Omega_h::parallel_for(1,  pushPtcl, "push");
}

} //namespace
#endif // PUMIPIC_PUSH_HPP_INCLUDED

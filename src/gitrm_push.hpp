#ifndef GITRM_PUSH_HPP_INCLUDED
#define GITRM_PUSH_HPP_INCLUDED

#include <iostream>
#include <cmath>
#include <utility>

#include "Omega_h_adj.hpp"
#include "Omega_h_array_ops.hpp"
#include "Omega_h_scalar.hpp" //divide
#include "Omega_h_fail.hpp"

#include "gitrm_utils.hpp"
#include "gitrm_constants.hpp"

namespace GITRm
{

//Temporary particle data. See how it is used in particle push in pushBoris(..)
class gitrmParticles
{
  public:
    Omega_h::LO np;
    gitrmParticles(Omega_h::LO np_):
      np(np_), x(np,0), y(np,0), z(np,0), xpre(np,0), ypre(np,0), zpre(np,0),
      vx(np,0), vy(np,0), vz(np,0), eFieldsxPre(np,0), eFieldsyPre(np,0),
      eFieldszPre(np,0), bFieldsrPre(np,0), bFieldstPre(np,0), bFieldszPre(np,0),
      hitWall(np,1), charge(np,-1), amu(np,1.66e-27)
    {}
    Omega_h::Write<Omega_h::Real> x,y,z;
    Omega_h::Write<Omega_h::Real> xpre, ypre, zpre;
    Omega_h::Write<Omega_h::Real> vx, vy, vz;
    Omega_h::Write<Omega_h::Real> eFieldsxPre, eFieldsyPre, eFieldszPre;
    Omega_h::Write<Omega_h::Real> bFieldsrPre, bFieldstPre, bFieldszPre;
    // br = -0.1 to 0.3, bt = -3 to 3, bz = -1 to 1.5
    Omega_h::Write<Omega_h::Real> hitWall, charge, amu; //const
};


// TODO what is nsteps ?
// Option to select various EFields removed, since it can be done while writing them in
// previous step, in the search routine.
OMEGA_H_INLINE void pushBoris(Omega_h::LO nelems, gitrmParticles &ptcl, Omega_h::Real dt)
{

  auto pushPtcl = OMEGA_H_LAMBDA( Omega_h::LO elID)
  {
    // B Fields ; br -0.1 to 0.3; bt = -3 to 3; bz= -1 to 1.5
    if(ptcl.hitWall[elID] == 0)
      return;

    //previous position
    Omega_h::Vector<3> pre = {ptcl.xpre[elID], ptcl.ypre[elID], ptcl.zpre[elID]};

    // TODO check if float is good, if so use it in other parts
    Omega_h::Vector<3> eField;
    Omega_h::Vector<3> bField;

    eField[0] = ptcl.eFieldsxPre[elID];
    eField[1] = ptcl.eFieldsyPre[elID];
    eField[2] = ptcl.eFieldszPre[elID];
    bField[0] = ptcl.bFieldsrPre[elID];
    bField[1] = ptcl.bFieldstPre[elID];
    bField[2] = ptcl.bFieldszPre[elID];

    Omega_h::Real bFieldMag = osh_mag(bField);
    OMEGA_H_CHECK(ptcl.amu[elID] >0 && dt>0);
    Omega_h::Real qPrime = ptcl.charge[elID]*1.60217662e-19/(ptcl.amu[elID]*1.6737236e-27)*dt*0.5;
    Omega_h::Real coeff = 2.0*qPrime/(1.0+(qPrime*bFieldMag)*(qPrime*bFieldMag));
    Omega_h::Vector<3> vel = {ptcl.vx[elID], ptcl.vy[elID], ptcl.vz[elID]};

      //v_minus = v + q_prime*E;
    Omega_h::Vector<3> qpE = qPrime*eField;
    Omega_h::Vector<3> vMinus = vel - qpE;
    //for(Omega_h::LO i=0; i<3; ++i)
    //  electricForce[i] = 2.0*qpE[i];  //?
    //v_prime = v_minus + q_prime*(v_minus x B)
    Omega_h::Vector<3> vmxB = Omega_h::cross(vMinus,bField);
    Omega_h::Vector<3> qpVmxB = qPrime*vmxB;
    Omega_h::Vector<3> vPrime = vMinus + qpVmxB;
    //for(Omega_h::LO i=0; i<3; ++i)
    //  magneticForce[i] = qpVmxB[i];  //?

    //v = v_minus + coeff*(v_prime x B)
    Omega_h::Vector<3> vpxB = Omega_h::cross(vPrime, bField);
    Omega_h::Vector<3> cVpxB = coeff*vpxB;
    vel = vMinus + cVpxB;

    //v = v + q_prime*E
    vel = vel + qpE;
    int thisTmp = (ptcl.vx[elID]< 10.0 && vel[0] > 4.0e3)?1:0; //?

    ptcl.x[elID] = pre[0] + vel[0] * dt;
    ptcl.y[elID] = pre[1] + vel[1] * dt;
    ptcl.z[elID] = pre[2] + vel[2] * dt;
    ptcl.vx[elID] = vel[0];
    ptcl.vy[elID] = vel[1];
    ptcl.vz[elID] = vel[2];
//#if DEBUG >0
    std::cout << elID << " " << ptcl.x[elID] << " " << ptcl.y[elID] << " " <<  ptcl.z[elID] << "\n";
//#endif // DEBUG
    };

    Omega_h::parallel_for(1,  pushPtcl, "push");

}

// TODO add RK4 method



} //namespace
#endif // GITRM_PUSH_HPP_INCLUDED

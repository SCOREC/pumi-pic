#ifndef THERMAL_FORCE_H
#define THERMAL_FORCE_H
#include "GitrmMesh.hpp"
#include "GitrmParticles.hpp"
void gitrm_thermal_force(PS* ptcls, int *iteration, const GitrmMesh& gm,
const GitrmParticles& gp, double dt)
{

  bool debug= 1;
  auto pid_ps = ptcls->get<PTCL_ID>();
  auto x_ps_d = ptcls->get<PTCL_POS>();
  auto xtgt_ps_d = ptcls->get<PTCL_NEXT_POS>();
  auto efield_ps_d  = ptcls->get<PTCL_EFIELD>();
  auto vel_ps_d = ptcls->get<PTCL_VEL>();
  auto charge_ps_d = ptcls->get<PTCL_CHARGE>();
  printf("Entering Thermal Force Routine\n");


  //Mesh data regarding the gradient of temperatures
   const auto &gradTi_d = gm.gradTi_d ;

  auto grTiX0 = gm.gradTiX0;   
  auto grTiZ0 = gm.gradTiZ0; 
  auto grTiNX = gm.gradTiNx; 
  auto grTiNZ = gm.gradTiNz;
  auto grTiDX = gm.gradTiDx;
  auto grTiDZ = gm.gradTiDz;

  
  const auto &gradTe_d = gm.gradTe_d;

  auto grTeX0 = gm.gradTeX0;   
  auto grTeZ0 = gm.gradTeZ0; 
  auto grTeNX = gm.gradTeNx; 
  auto grTeNZ = gm.gradTeNz;
  auto grTeDX = gm.gradTeDx;
  auto grTeDZ = gm.gradTeDz;



  o::Reals bFieldConst(3); 
  if(USE_CONSTANT_BFIELD) {
    bFieldConst = o::Reals(o::HostWrite<o::Real>({CONSTANT_BFIELD0,
        CONSTANT_BFIELD1, CONSTANT_BFIELD2}).write());
  }
  const o::Real amu = gitrm::PTCL_AMU; 
  const o::Real background_Z = BACKGROUND_Z;
  const o::Real background_amu = gitrm::BACKGROUND_AMU;
  const auto MI     = 1.6737236e-27;


  auto update_thermal = PS_LAMBDA(const int& e, const int& pid, const bool& mask) 
	{ if(mask > 0 )
    	{	
    		auto posit          = p::makeVector3(pid, x_ps_d);
        auto ptcl           = pid_ps(pid);
        auto charge         = charge_ps_d(pid);
          if(!charge)
            return;

        auto posit_next     = p::makeVector3(pid, xtgt_ps_d);
        auto eField         = p::makeVector3(pid, efield_ps_d);
        auto vel            = p::makeVector3(pid, vel_ps_d);
        auto bField         = o::zero_vector<3>();


        for(auto i=0; i<3; ++i)
            bField[i] = bFieldConst[i];
        auto b_mag  = Omega_h::norm(bField);
        Omega_h::Vector<3> b_unit = bField/b_mag;

        auto pos2D          = o::zero_vector<3>();
    		pos2D[0]            = sqrt(posit[0]*posit[0] + posit[1]*posit[1]);
    		pos2D[1]            = 0;
    		pos2D[2]            = posit[2];

        auto gradti         = o::zero_vector<3>();
        auto gradte         = o::zero_vector<3>();

    		//find out the gradients of the electron and ion temperatures at that particle poin
        p::interp2dVector(gradTi_d, grTiX0, grTiZ0, grTiDX, grTiDZ, grTiNX, grTiNX, pos2D, gradti, false);
        p::interp2dVector(gradTe_d, grTeX0, grTeZ0, grTeDX, grTeDZ, grTeNX, grTeNX, pos2D, gradte, false);

    		o::Real mu = amu /(background_amu+amu);
        o::Real alpha = 0.71*charge*charge;
        o::Real beta = 3 * (mu + 5*sqrt(2.0)*charge*charge*(1.1*pow(mu, (5 / 2))-0.35*pow(mu,(3/2)))-1)/
        (2.6 - 2*mu + 5.4*std::pow(mu, 2));
        auto dv_ITG= o::zero_vector<3>();
        dv_ITG[0] =1.602e-19*dt/(amu*MI)*beta*gradti[0]*b_unit[0];
        dv_ITG[1] =1.602e-19*dt/(amu*MI)*beta*gradti[1]*b_unit[1];
        dv_ITG[2] =1.602e-19*dt/(amu*MI)*beta*gradti[2]*b_unit[2];

        vel_ps_d(pid,0)=posit[0]+dv_ITG[0];
        vel_ps_d(pid,1)=posit[1]+dv_ITG[0];
        vel_ps_d(pid,2)=posit[2]+dv_ITG[0]; 
        

    	}
    };
    ps::parallel_for(ptcls, update_thermal);
    
}
#endif
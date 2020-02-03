#ifndef THERMAL_FORCE_H
#define THERMAL_FORCE_H
#include "GitrmMesh.hpp"
#include "GitrmParticles.hpp"
void gitrm_thermal_force(PS* ptcls, int *iteration, const GitrmMesh& gm,
const GitrmParticles& gp, double dt)
{

  bool debug= 0;
  auto pid_ps = ptcls->get<PTCL_ID>();
  auto x_ps_d = ptcls->get<PTCL_POS>();
  auto xtgt_ps_d = ptcls->get<PTCL_NEXT_POS>();
  auto efield_ps_d  = ptcls->get<PTCL_EFIELD>();
  auto vel_ps_d = ptcls->get<PTCL_VEL>();
  auto charge_ps_d = ptcls->get<PTCL_CHARGE>();
  printf("Entering Thermal Force Routine\n");
  

  const auto &gradTi_d = gm.gradTi_d ;

  //Mesh data regarding the gradient of temperatures

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
  const double amu = 184; 
  //const o::Real background_Z = BACKGROUND_Z;
  const double background_amu = 4;
  const auto MI     = 1.6737236e-27;

  const auto iTimeStep = iTimePlusOne - 1;


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
        auto bField_radial  = o::zero_vector<3>();
        auto bField         = o::zero_vector<3>();

        
        for(auto i=0; i<3; ++i)
            bField_radial[i] = bFieldConst[i];

          //Transformation of B field from cylindrical to cartesian

            o::Real theta = atan2(posit_next[1], posit_next[0]);  
            bField[0] = cos(theta)*bField_radial[0] - sin(theta)*bField_radial[1];
            bField[1] = sin(theta)*bField_radial[0] + cos(theta)*bField_radial[1];
            bField[2] = bField_radial[2];



        auto b_mag  = Omega_h::norm(bField);
        Omega_h::Vector<3> b_unit = bField/b_mag;

        auto pos2D          = o::zero_vector<3>();
    		pos2D[0]            = sqrt(posit_next[0]*posit_next[0] + posit_next[1]*posit_next[1]);
    		pos2D[1]            = 0;
    		pos2D[2]            = posit_next[2];

        auto gradti         = o::zero_vector<3>();
        auto gradte         = o::zero_vector<3>();

    		//find out the gradients of the electron and ion temperatures at that particle poin
        p::interp2dVector(gradTi_d, grTiX0, grTiZ0, grTiDX, grTiDZ, grTiNX, grTiNZ, posit_next, gradti, true);
        p::interp2dVector(gradTe_d, grTeX0, grTeZ0, grTeDX, grTeDZ, grTeNX, grTeNZ, posit_next, gradte, true);
        

    

    		o::Real mu = amu /(background_amu+amu);
        o::Real alpha = 0.71*charge*charge;
        o::Real beta = 3 * (mu + 5*sqrt(2.0)*charge*charge*(1.1*pow(mu, (5 / 2))-0.35*pow(mu,(3/2)))-1)/
        (2.6 - 2*mu + 5.4*pow(mu, 2));
        auto dv_ITG= o::zero_vector<3>();
        dv_ITG[0] =1.602e-19*dt/(amu*MI)*beta*gradti[0]*b_unit[0];
        dv_ITG[1] =1.602e-19*dt/(amu*MI)*beta*gradti[1]*b_unit[1];
        dv_ITG[2] =1.602e-19*dt/(amu*MI)*beta*gradti[2]*b_unit[2];

        if (debug){
          printf("Ion_temp_grad particle %d timestep %d: %.16e %.16e %.16e \n",ptcl,iTimeStep, gradti[0], gradti[1], gradti[2]);
          printf("El_temp_grad particle %d timestep %.d: %.16e %.16e %.16e \n",ptcl,iTimeStep,gradte[0], gradte[1], gradte[2] );
          printf("Deltavs particle %d timestep %d: %.16e %.16e %.16e \n", ptcl,iTimeStep, dv_ITG[0], dv_ITG[1], dv_ITG[2]);
          printf("Positions particle %d timestep %d: %1.16e %.16e %.16e \n", ptcl,iTimeStep, posit[0], posit[1], posit[2]);
          printf("Mu Beta Magnetic_fields particle %d timestep %d: %.16e %.16e %.16e %.16e %.16e\n", ptcl,iTimeStep, mu, beta, b_unit[0], b_unit[1],b_unit[2]);
          printf("amu background_amu particle %d timestep %d: %.16e %.16e \n",ptcl,iTimeStep, amu, background_amu);
          printf("Charge particle %d timestep %d: %d \n", ptcl,iTimeStep, charge);
        }

        if(1){
     
        //printf("Position partcle %d for timestep %d is %.15e %.15e %.15e \n",iTimeStep, ptcl, posit[0],posit[1],posit[2]);
        printf("Position partcle %d timestep %d is %.15e %.15e %.15e \n",ptcl, iTimeStep, posit_next[0],posit_next[1],posit_next[2]);
        printf("The velocities partcle %d timestep %dare %.15e %.15e %.15e \n", iTimeStep, ptcl, vel[0],vel[1],vel[2]); 
        //printf("vPartNorm nuEdt partcle %d for timestep %d is %.15e %.15e\n", ptcl, iTimeStep,Omega_h::norm(vel), nuEdt);
        //printf("coeff_par parallel_direction partcle %d for timestep %d is %.15e %.15e %.15e %.15e\n", ptcl, iTimeStep, coeff_par,parallel_dir[0],parallel_dir[1],parallel_dir[2]);
        //printf("coeff_perp1 coeff_perp2 partcle %d timestep %d is %.15e %.15e \n", ptcl, iTimeStep, coeff_perp1,coeff_perp2);
        //printf("Perpendicuar directions1 partcle %d timestep %d is %.15e %.15e %.15e \n", ptcl, iTimeStep, perp1_dir[0],perp1_dir[1],perp1_dir[2]);      
        //printf("Perpendicuar directions2 partcle %d timestep %d is %.15e %.15e %.15e \n", ptcl, iTimeStep, perp2_dir[0],perp2_dir[1],perp2_dir[2]);
        //printf("velocity Collision partcle %d timestep %d is %.15e %.15e %.15e \n", ptcl, iTimeStep, drag*relvel[0]/velocityNorm,drag*relvel[1]/velocityNorm,drag*relvel[2]/velocityNorm);
        //printf("n2 is %.15e\n", n2);
      
        }


        vel_ps_d(pid,0)=vel[0]+dv_ITG[0];
        vel_ps_d(pid,1)=vel[1]+dv_ITG[1];
        vel_ps_d(pid,2)=vel[2]+dv_ITG[2];

        printf("The velocities after updation THERMAL_COLSN partcle %d timestep %d are %.15e %.15e %.15e \n", ptcl, iTimeStep, vel_ps_d(pid,0),vel_ps_d(pid,1),vel_ps_d(pid,2)); 
     
        

    	}
    };
    ps::parallel_for(ptcls, update_thermal);
    
}
#endif
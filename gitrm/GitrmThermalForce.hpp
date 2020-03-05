#ifndef THERMAL_FORCE_H
#define THERMAL_FORCE_H
#include "GitrmMesh.hpp"
#include "GitrmParticles.hpp"
void gitrm_thermal_force(PS* ptcls, int *iteration, const GitrmMesh& gm,
const GitrmParticles& gp, double dt, o::Write<o::LO>& elm_ids)
{

  bool debug= 0;
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

  //Setting up of 2D magnetic field data 
  const auto& BField_2d = gm.Bfield_2d;
  const auto bX0 = gm.bGridX0;
  const auto bZ0 = gm.bGridZ0;
  const auto bDx = gm.bGridDx;
  const auto bDz = gm.bGridDz;
  const auto bGridNx = gm.bGridNx;
  const auto bGridNz = gm.bGridNz;

  auto& mesh = gm.mesh;
  const auto coords = mesh.coords();
  const auto mesh2verts = mesh.ask_elem_verts();
  const auto gradTionVtx = mesh.get_array<o::Real>(o::VERT, "gradTiVtx");
  const auto gradTelVtx   = mesh.get_array<o::Real>(o::VERT, "gradTeVtx");
  const auto BField = o::Reals(); //o::Reals(mesh.get_array<o::Real>(o::VERT, "BField"));



  auto useConstantBField = USE_CONSTANT_BFIELD;
  auto use2dInputFields = USE2D_INPUTFIELDS;
  auto use3dField = USE3D_BFIELD;
  bool cylSymm = true;

  const double amu = 184; 
  //const o::Real background_Z = BACKGROUND_Z;
  const double background_amu = 4;
  const auto MI     = 1.6737236e-27;

  const auto iTimeStep = iTimePlusOne - 1;
  auto& xfaces =gp.wallCollisionFaceIds;

  auto update_thermal = PS_LAMBDA(const int& e, const int& pid, const bool& mask) 
	{ if(mask > 0 && elm_ids[pid] >= 0)
    	{	
    		o::LO el = elm_ids[pid];
        auto posit          = p::makeVector3(pid, x_ps_d);
        auto ptcl           = pid_ps(pid);
        auto charge         = charge_ps_d(pid);
        auto fid            = xfaces[ptcl];
          if(!charge || fid >=0)
             return;

        auto posit_next     = p::makeVector3(pid, xtgt_ps_d);
        auto eField         = p::makeVector3(pid, efield_ps_d);
        auto vel            = p::makeVector3(pid, vel_ps_d);
        auto bField         = o::zero_vector<3>();

        
        if (use2dInputFields || useConstantBField){

            p::interp2dVector(BField_2d, bX0, bZ0, bDx, bDz, bGridNx, bGridNz, posit_next, bField, cylSymm, &ptcl);
            
        }

        else if (use3dField){

            auto bcc = o::zero_vector<4>();
            p::findBCCoordsInTet(coords, mesh2verts, posit_next, el, bcc);
            p::interpolate3dFieldTet(mesh2verts, BField, el, bcc, bField); 

        }



        auto b_mag  = Omega_h::norm(bField);
        Omega_h::Vector<3> b_unit = bField/b_mag;

        auto pos2D          = o::zero_vector<3>();
    		pos2D[0]            = sqrt(posit_next[0]*posit_next[0] + posit_next[1]*posit_next[1]);
    		pos2D[1]            = 0;
    		pos2D[2]            = posit_next[2];

        auto gradti         = o::zero_vector<3>();
        auto gradte         = o::zero_vector<3>();


        //find out the gradients of the electron and ion temperatures at that particle poin

        if (use2dInputFields){
        p::interp2dVector(gradTi_d, grTiX0, grTiZ0, grTiDX, grTiDZ, grTiNX, grTiNZ, posit_next, gradti, true);
        p::interp2dVector(gradTe_d, grTeX0, grTeZ0, grTeDX, grTeDZ, grTeNX, grTeNZ, posit_next, gradte, true);
        }
  
        else if (use3dField){
          auto bcc = o::zero_vector<4>();
          p::findBCCoordsInTet(coords, mesh2verts, posit_next, el, bcc);
          p::interpolate3dFieldTet(mesh2verts, gradTionVtx, el, bcc, gradti);
          p::interpolate3dFieldTet(mesh2verts, gradTelVtx, el, bcc, gradte);
        }  

        
    		o::Real mu = amu /(background_amu+amu);
        //o::Real alpha = 0.71*charge*charge;
        o::Real beta = 3 * (mu + 5*sqrt(2.0)*charge*charge*(1.1*pow(mu, (5 / 2))-0.35*pow(mu,(3/2)))-1)/
        (2.6 - 2*mu + 5.4*pow(mu, 2));
        auto dv_ITG= o::zero_vector<3>();
        dv_ITG[0] =1.602e-19*dt/(amu*MI)*beta*gradti[0]*b_unit[0];
        dv_ITG[1] =1.602e-19*dt/(amu*MI)*beta*gradti[1]*b_unit[1];
        dv_ITG[2] =1.602e-19*dt/(amu*MI)*beta*gradti[2]*b_unit[2];

        if (debug && ptcl==4){

          printf("Ion_temp_grad particle %d timestep %d: %.16e %.16e %.16e \n",ptcl,iTimeStep, gradti[0], gradti[1], gradti[2]);
          printf("El_temp_grad particle %d timestep %.d: %.16e %.16e %.16e \n",ptcl,iTimeStep,gradte[0], gradte[1], gradte[2] );
          printf("Deltavs particle %d timestep %d: %.16e %.16e %.16e \n", ptcl,iTimeStep, dv_ITG[0], dv_ITG[1], dv_ITG[2]);
          printf("Positions particle %d timestep %d: %1.16e %.16e %.16e \n", ptcl,iTimeStep, posit[0], posit[1], posit[2]);
          printf("Mu Beta Magnetic_fields particle %d timestep %d: %.16e %.16e %.16e %.16e %.16e\n", ptcl,iTimeStep, mu, beta, b_unit[0], b_unit[1],b_unit[2]);
          printf("amu background_amu particle %d timestep %d: %.16e %.16e \n",ptcl,iTimeStep, amu, background_amu);
          printf("Charge particle %d timestep %d: %d \n", ptcl,iTimeStep, charge);
          printf("Position partcle %d timestep %d is %.15e %.15e %.15e \n",ptcl, iTimeStep, posit_next[0],posit_next[1],posit_next[2]);
          printf("The velocities partcle %d timestep %dare %.15f %.15f %.15f \n", iTimeStep, ptcl, vel[0],vel[1],vel[2]); 
        }


        vel_ps_d(pid,0)=vel[0]+dv_ITG[0];
        vel_ps_d(pid,1)=vel[1]+dv_ITG[1];
        vel_ps_d(pid,2)=vel[2]+dv_ITG[2];
        
        if (debug && ptcl==4){
          printf("The velocities after updation THERMAL_COLSN partcle %d timestep %d are %.15f %.15f %.15f \n", ptcl, iTimeStep, vel_ps_d(pid,0),vel_ps_d(pid,1),vel_ps_d(pid,2));
        } 
    	}
    };
    ps::parallel_for(ptcls, update_thermal);
    
}
#endif

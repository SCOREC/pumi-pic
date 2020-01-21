#ifndef THERMAL_FORCE_H
#define THERMAL_FORCE_H

void calculate_thermal_force(SCS* scs, int *iteration, const GitrmMesh& gm, const GitrmParticles& gp, double dt)
{

  int j=*iteration;
  bool debug= 1;
  auto pid_scs = scs->get<PTCL_ID>();
  auto x_scs_d = scs->get<PTCL_POS>();
  auto xtgt_scs_d = scs->get<PTCL_NEXT_POS>();
  auto efield_scs_d  = scs->get<PTCL_EFIELD>();
  auto vel_scs_d = scs->get<PTCL_VEL>();
  auto charge_scs_d = scs->get<PTCL_CHARGE>();

  //Mesh data regarding the gradient of temperatures
   const auto &gradTiR_d = gm.gradTiR_d ;
   const auto &gradTiT_d = gm.gradTiT_d ;
   const auto &gradTiZ_d = gm.gradTiZ_d ;
   const auto &gradTeR_d = gm.gradTiR_d ;
   const auto &gradTeT_d = gm.gradTiT_d ;
   const auto &gradTeZ_d = gm.gradTiZ_d ;

  auto gradTX0 = gm.gradTiRX0; //THis is just 1 set of data. There will be 5 more. 
  auto gradTZ0 = gm.gradTiRZ0; // However if the grid (mumber and daa points) are the same,
  auto gradTNX = gm.gradTiRNx; // they can be ignored.
  auto gradTNZ = gm.gradTiRNz;
  auto gradTDX = gm.gradTiRDx;
  auto gradTDZ = gm.gradTiRDz;



  auto update_thermal = SCS_LAMBDA(const int& e, const int& pid, const bool& mask) 
	{ if(mask > 0 )
    	{	
    		auto posit          = p::makeVector3(pid, x_scs_d);
    		auto ptcl           = pid_scs(pid);
    		auto charge         = charge_scs_d(pid);
    			if(!charge)
      			return;

    		auto posit_next     = p::makeVector3(pid, xtgt_scs_d);
    		auto eField         = p::makeVector3(pid, efield_scs_d);
    		auto vel            = p::makeVector3(pid, vel_scs_d);
    		pos2D[0]            = sqrt(posit[0]*posit[0] + posit[1]*posit[1]);
    		pos2D[1]            = 0;
    		pos2D[2]            = posit[2];
    		//find out the gradients of the electron and ion temperatures at that particle point
    		auto gradTiR_ptcl = p::interpolate2dField(gradTiR_d, gradTX0, gradTZ0, gradTNX, 
            gradTNZ, gradTDX, gradTDZ, pos2D, true,1,0,false);

            auto gradTiT_ptcl = p::interpolate2dField(gradTiT_d, gradTX0, gradTZ0, gradTNX, 
            gradTNZ, gradTDX, gradTDZ, pos2D, true,1,0,false);

            auto gradTiZ_ptcl = p::interpolate2dField(gradTiZ_d, gradTX0, gradTZ0, gradTNX, 
            gradTNZ, gradTDX, gradTDZ, pos2D, true,1,0,false);

            auto gradTeR_ptcl = p::interpolate2dField(gradTeR_d, gradTX0, gradTZ0, gradTNX, 
            gradTNZ, gradTDX, gradTDZ, pos2D, true,1,0,false);

            auto gradTeT_ptcl = p::interpolate2dField(gradTeT_d, gradTX0, gradTZ0, gradTNX, 
            gradTNZ, gradTDX, gradTDZ, pos2D, true,1,0,false);

            auto gradTeZ_ptcl = p::interpolate2dField(gradTeZ_d, gradTX0, gradTZ0, gradTNX, 
            gradTNZ, gradTDX, gradTDZ, pos2D, true,1,0,false);

    		//mu=particle amu /(back_ground_amu+particle amu);
    		//alpha= 0.71*particle chare^2;
    		//beta = 3*(mu+%*sqrt
    		//find the magnetic field B[0], B[1],B[2]
    	}
    };
    scs->parallel_for(update_thermal);
}
#endif
#ifndef CROSS_DIFFUSION_H
#define CROSS_DIFFUSION_H
#include "GitrmMesh.hpp"
#include "GitrmParticles.hpp"


void gitrm_cross_diffusion(PS* ptcls, int *iteration, const GitrmMesh& gm,
const GitrmParticles& gp, double dt)
{

  bool debug= 0;
  auto pid_ps = ptcls->get<PTCL_ID>();
  auto x_ps_d = ptcls->get<PTCL_POS>();
  auto xtgt_ps_d = ptcls->get<PTCL_NEXT_POS>();
  auto vel_ps_d = ptcls->get<PTCL_VEL>();
  auto charge_ps_d = ptcls->get<PTCL_CHARGE>();

  printf("Entering CROSS DIFFUSION Routine\n");
  
  o::Reals bFieldConst(3); 
  if(USE_CONSTANT_BFIELD) {
    bFieldConst = o::Reals(o::HostWrite<o::Real>({CONSTANT_BFIELD0,
        CONSTANT_BFIELD1, CONSTANT_BFIELD2}).write());
  }

  const int USEPERPDIFFUSION = 1;
  const double diffusionCoefficient=1;

  const auto& testGitrPtclStepData = gp.testGitrPtclStepData;
  const auto testGDof = gp.testGitrStepDataDof;
  const auto testGNT = gp.testGitrStepDataNumTsteps;
  const auto iTimeStep = iTimePlusOne - 1;
  const auto diff_rnd1 = gp.testGitrCrossFieldDiffRndInd;
  auto& xfaces =gp.wallCollisionFaceIds;

  auto update_diffusion = PS_LAMBDA(const int& e, const int& pid, const bool& mask) 
	{ if(mask > 0 )
    	{	

        auto ptcl           = pid_ps(pid);
        auto charge         = charge_ps_d(pid);
        auto fid            = xfaces[ptcl];
          if(!charge || fid >=0)
             return;
        
        auto posit          = p::makeVector3(pid, x_ps_d);
        auto posit_next     = p::makeVector3(pid, xtgt_ps_d);
        auto vel            = p::makeVector3(pid, vel_ps_d);
        auto bField_radial  = o::zero_vector<3>();
        auto bField         = o::zero_vector<3>();
        auto bField_plus    = o::zero_vector<3>();
        auto bField_deriv   = o::zero_vector<3>(); 
        o::Real phi_random;

        for(auto i=0; i<3; ++i)
            bField_radial[i] = bFieldConst[i];

        
        //Transformation of B field from cylindrical to cartesian
        o::Real theta = atan2(posit_next[1], posit_next[0]);  
        bField[0] = cos(theta)*bField_radial[0] - sin(theta)*bField_radial[1];
        bField[1] = sin(theta)*bField_radial[0] + cos(theta)*bField_radial[1];
        bField[2] = bField_radial[2];

        auto perpVector         = o::zero_vector<3>();

        auto b_mag  = Omega_h::norm(bField);
        Omega_h::Vector<3> b_unit = bField/b_mag;

    		double r3 = 0.5;
        //o::Real r4 = 0.75;

        double step=sqrt(6*diffusionCoefficient*dt);

      /*
      if(USEPERPDIFFUSION>1){
        double plus_minus1=floor(r4+0.5)*2-1.0;
        double h=0.001;

        auto posit_next_plus  = o::zero_vector<3>();
        posit_next_plus=posit_next_plus + h*bField;

        o::Real theta_plus = atan2(posit_next_plus[1], posit_next_plus[0]);  
        bField_plus[0] = cos(theta_plus)*bField_radial[0] - sin(theta_plus)*bField_radial[1];
        bField_plus[1] = sin(theta_plus)*bField_radial[0] + cos(theta_plus)*bField_radial[1];
        bField_plus[2] = bField_radial[2];
        auto b_plus_mag  = Omega_h::norm(bField_plus);

        bField_deriv=(bField_plus-bField)/h;
        auto denom = Omega_h::norm(bField_deriv);
        double R = 1.0e4;
        if(( abs(denom) > 1e-10) & ( abs(denom) < 1e10) )
        {
            R = b_mag/denom;
        }


        double initial_guess_theta = 3.14159265359*0.5;
        double eps = 0.01;
        double error = 2.0;
        double s = step;
        double drand = r3;
        double theta0 = initial_guess_theta;
        double theta1 = 0.0;
        double f = 0.0;
        double f_prime = 0.0;
        int nloops = 0;

         if(R > 1.0e-4){

              while ((error > eps)&(nloops<10)){

                f = (2*R*theta0-s*sin(theta0))/(2*3.14159265359*R) - drand;
                f_prime = (2*R-s*cos(theta0))/(2*3.14159265359*R); 
                theta1 = theta0 - f/f_prime;
                error = abs(theta1-theta0);
                theta0=theta1;

                nloops++;
             }

              if(nloops > 9){
                theta0 = 2*3.14159265359*drand;
              }
          }
          
          else{
              
              R = 1.0e-4;
              theta0 = 2*3.14159265359*drand;
          }

          //TO USE CONDITIONAL STATEMENT

            if(plus_minus1 < 0){

                theta0 = 2*3.14159265359-theta0; 
            }
        

            perpVector              = bField_deriv/Omega_h::norm(bField_deriv);
            auto y_dir              = o::zero_vector<3>();
            y_dir                   = Omega_h::cross(bField, bField_deriv);

            double x_comp = s*cos(theta0);
            double y_comp = s*sin(theta0);

            auto transform         = o::zero_vector<3>();
            transform              = x_comp*perpVector+y_comp*y_dir;
            if (abs(denom) > 1.0e-8)
            
              {
                
                xtgt_ps_d(pid,0)=posit_next[0]+transform[0];
                xtgt_ps_d(pid,1)=posit_next[1]+transform[1];
                xtgt_ps_d(pid,2)=posit_next[2]+transform[2];
                //exit(0);
              }

        }*/


        if( USEPERPDIFFUSION==1){

                

                r3  = testGitrPtclStepData[ptcl*testGNT*testGDof + iTimeStep*testGDof + diff_rnd1];
 
                phi_random = 2*3.14159265*r3;
                perpVector[0] =  cos(phi_random);
                perpVector[1] =  sin(phi_random);
                perpVector[2] = (-perpVector[0]*b_unit[0] - perpVector[1]*b_unit[1])/b_unit[2];
              
                if (b_unit[2] == 0){
                    perpVector[2] = perpVector[1];
                    perpVector[1] = (-perpVector[0]*b_unit[0] - perpVector[2]*b_unit[2])/b_unit[1];
                }
                
      
                if ((b_unit[0] == 1.0 && b_unit[1] ==0.0 && b_unit[2] ==0.0) || (b_unit[0] == -1.0 && b_unit[1] ==0.0 && b_unit[2] ==0.0)){
                    perpVector[2] = perpVector[0];
                    perpVector[0] = 0;
                    perpVector[1] = sin(phi_random);
                  // cout << "perp Vector " << perpVector[0] << " " << perpVector[1] << " " << perpVector[2]<< endl;
                }

                else if ((b_unit[0] == 0.0 && b_unit[1] ==1.0 && b_unit[2] ==0.0) || (b_unit[0] == 0.0 && b_unit[1] ==-1.0 && b_unit[2] ==0.0)){

                    perpVector[1] = 0.0;
                }

                else if ((b_unit[0] == 0.0 && b_unit[1] ==0.0 && b_unit[2] ==1.0) || (b_unit[0] == 0.0 && b_unit[1] ==0.0 && b_unit[2] ==-1.0)){

                perpVector[2] = 0;
                
                }

                perpVector = perpVector/Omega_h::norm(perpVector);

                xtgt_ps_d(pid,0)=posit_next[0]+step*perpVector[0];
                xtgt_ps_d(pid,1)=posit_next[1]+step*perpVector[1];
                xtgt_ps_d(pid,2)=posit_next[2]+step*perpVector[2];

        } 
    if (debug){      
    printf("The positions before updation CROSS_DIFFUSION partcle %d timestep %d are %.15f %.15f %.15f \n", ptcl, iTimeStep, posit_next[0], posit_next[1], posit_next[2]); 
    printf("The perpVectors are %.15f %0.15f %0.15f \n",perpVector[0],perpVector[1],perpVector[2]);
    printf("The random number is %.15f\n",r3);
    printf("The numbers %d  %d  %d  %d  %d  %d \n",ptcl,testGNT, testGDof, iTimeStep, testGDof, diff_rnd1);
    printf("The diffusion coefficient and step are %.15f,%.15f \n",diffusionCoefficient,step);
    printf("The positions after updation CROSS_DIFFUSION partcle %d timestep %d are %.15f %.15f %.15f \n", ptcl, iTimeStep, xtgt_ps_d(pid,0), xtgt_ps_d(pid,1), xtgt_ps_d(pid,2)); 
    }

    }
    
  };
    ps::parallel_for(ptcls, update_diffusion);
    
}
#endif

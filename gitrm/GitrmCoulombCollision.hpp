#ifndef GITRM_COULOMB_COLLISION_H
#define GITRM_COULOMB_COLLISION_H
#include "GitrmParticles.hpp"

OMEGA_H_DEVICE void get_drag(const Omega_h::Vector<3> &vel, const Omega_h::Vector<3> &flowVelocity,
double& nu_friction,double& nu_deflection, double& nu_parallel,double& nu_energy, const double temp,
const double temp_el, const double dens, const double charge, int ptcl_t, int iTimeStep_t, int debug)
{
  
  const double Q    = 1.60217662e-19;
  const double EPS0 = 8.854187e-12;
  const double pi   = 3.14159265;
  const auto MI     = 1.6737236e-27;
  const auto ME     = 9.10938356e-31;
  const double amu  = 184; 
  const double background_amu=4;
  const double background_Z = 1;
    
    Omega_h::Vector<3>relvel = vel - flowVelocity;
    auto velocityNorm   = Omega_h::norm(relvel);
    //parallel_dir=relvel/velocityNorm;
    
    auto lam_d = sqrt(EPS0*temp_el/(dens*pow(background_Z,2)*Q));
    auto lam = 4*pi*dens*pow(lam_d,3);

    auto gam_electron_background = 0.238762895*pow(charge,2)*log(lam)/(amu*amu);

    if(gam_electron_background < 0.0){

        gam_electron_background=0.0;

    }

    auto gam_ion_background = 0.238762895*pow(charge,2)*pow(background_Z,2)*log(lam)/(amu*amu);

    if(gam_ion_background < 0.0){

      gam_ion_background=0.0;
      
    }
      
    auto nu_0_i = gam_electron_background*dens/pow(velocityNorm,3);      
    auto nu_0_e = gam_ion_background*dens/pow(velocityNorm,3);

    auto xx_i= pow(velocityNorm,2)*background_amu*MI/(2*temp*Q);
    auto xx_e= pow(velocityNorm,2)*ME/(2*temp_el*Q);
        
    auto psi_i       = 0.75225278*pow(xx_i,1.5);
    auto psi_prime_i = 1.128379*sqrt(xx_i);
    //auto psi_psiprime_i   = psi_i+psi_prime_i;
    auto psi_psiprime_psi2x_i = 1.128379*sqrt(xx_i)*exp(-xx_i);
      
    auto psi_e = 0.75225278*pow(xx_e,1.5); 
    auto psi_prime_e = 1.128379*sqrt(xx_e);
    //auto psi_psiprime_e = psi_e+psi_prime_e;
    auto psi_psiprime_psi2x_e = 1.128379*sqrt(xx_e)*exp(-xx_e);
   
    auto nu_friction_i=((1+amu/background_amu)*psi_i)*nu_0_i;
    auto nu_deflection_i = 2*(psi_psiprime_psi2x_i)*nu_0_i;
    auto nu_parallel_i = psi_i/xx_i*nu_0_i;
    auto nu_energy_i = 2*(amu/background_amu*psi_i - psi_prime_i)*nu_0_i;

    auto nu_friction_e=((1+amu*MI/ME)*psi_e)*nu_0_e;
    auto nu_deflection_e = 2*(psi_psiprime_psi2x_e)*nu_0_e;
    auto nu_parallel_e = psi_e/xx_e*nu_0_e;
    auto nu_energy_e = 2*(amu/(ME/MI)*psi_e - psi_prime_e)*nu_0_e;


    nu_friction = nu_friction_i + nu_friction_e;
    nu_deflection = nu_deflection_i + nu_deflection_e;
    nu_parallel = nu_parallel_i + nu_parallel_e;
    nu_energy = nu_energy_i + nu_energy_e;

    if (temp<=0.0|| temp_el<=0.0){

      nu_friction=0.0;
      nu_deflection = 0.0;
      nu_parallel = 0.0;
      nu_energy = 0.0;
    }

    if (dens<=0){

      nu_friction=0.0;
      nu_deflection = 0.0;
      nu_parallel = 0.0;
      nu_energy = 0.0;
    }

    if(debug){

      printf("NU_friction  for timestep %d is %g \n",iTimeStep_t, nu_friction);
      printf("NU_deflection  for timestep %d is %g \n",iTimeStep_t, nu_deflection);
      printf("NU_parallel  for timestep %d is %g \n",iTimeStep_t, nu_parallel);
      printf("NU_energy  for timestep %d is %g \n",iTimeStep_t, nu_energy);
      printf("Ion temperature, electron temperature and ion density for timestep %d are %g %g %g \n",iTimeStep_t, temp, temp_el,dens);
      
    }

    
}

OMEGA_H_DEVICE void get_direc(const Omega_h::Vector<3> &vel, const Omega_h::Vector<3> &flowVelocity,
  const Omega_h::Vector<3> &b_field, Omega_h::Vector<3> &parallel_dir, Omega_h::Vector<3> &perp1_dir,
  Omega_h::Vector<3> &perp2_dir, int ptcl_t, int iTimeStep_t, int debug){

    auto b_mag  = Omega_h::norm(b_field);
    Omega_h::Vector<3> b_unit = b_field/b_mag;
   
    if(b_mag == 0.0){

            b_unit[0] = 1.0;
            b_unit[1] = 0.0;
            b_unit[2] = 0.0;
            b_mag     = 1.0;
    }
    


    Omega_h::Vector<3> relvel1 = vel ;// - flowVelocity;
    auto velocityNorm   = Omega_h::norm(relvel1);
    parallel_dir=relvel1/velocityNorm;
    
    auto s1 = Omega_h::inner_product(parallel_dir, b_unit);
    auto s2 = sqrt(1.0-s1*s1);
    if(abs(s1) >=1.0) s2=0;
    perp1_dir=1.0/s2*(s1*parallel_dir-b_unit);
    perp2_dir=1.0/s2*(Omega_h::cross(parallel_dir, b_unit));

    if(debug){

      printf("s1  for timestep %d is %g \n",iTimeStep_t, s1);
      printf("s2  for timestep %d is %g \n",iTimeStep_t, s2);
      printf("parallel_dir  for timestep %d is %g %g %g \n",iTimeStep_t, parallel_dir[0], parallel_dir[1], parallel_dir[2]);
      printf("b_field  for timestep %d is %g %g %g \n",iTimeStep_t, b_field[0], b_field[1],b_field[2]);
      printf("b_unit  for timestep %d is %g %g %g \n",iTimeStep_t, b_unit[0], b_unit[1],b_unit[2]);
      printf("perp1_dir for timestep %d is %g %g %g \n",iTimeStep_t, perp1_dir[0], perp1_dir[1], perp1_dir[2]);
      printf("perp2_dir for timestep %d is %g %g %g \n",iTimeStep_t, perp2_dir[0], perp2_dir[1], perp2_dir[2]);
      
    }  

    if (s2 == 0.0){
                
      perp1_dir[0] =  s1;//why no idea?
      perp1_dir[1] =  s2;//why no idea?

      perp2_dir[0] = parallel_dir[2];
      perp2_dir[1] = parallel_dir[0];
      perp2_dir[2] = parallel_dir[1];

           
      s1 = Omega_h::inner_product(parallel_dir, perp2_dir);
      s2 = sqrt(1.0-s1*s1);
      perp1_dir = -1.0/s2*Omega_h::cross(parallel_dir,perp2_dir);
               
    }

}

inline void gitrm_coulomb_collision(PS* ptcls, int *iteration, const GitrmMesh& gm,
 const GitrmParticles& gp, double dt, o::Write<o::LO>& elm_ids) {
  bool debug = 0;
  auto pid_ps = ptcls->get<PTCL_ID>();
  auto x_ps_d = ptcls->get<PTCL_POS>();
  auto xtgt_ps_d = ptcls->get<1>();
  auto efield_ps_d  = ptcls->get<PTCL_EFIELD>();
  auto vel_ps_d = ptcls->get<PTCL_VEL>();
  auto charge_ps_d = ptcls->get<PTCL_CHARGE>();
  printf("Entering coulomb_collision routine\n");

  auto useConstantFlowVelocity=USE_CONSTANT_FLOW_VELOCITY;
  auto useConstantBField = USE_CONSTANT_BFIELD;
  auto use2dInputFields = USE2D_INPUTFIELDS;
  auto use3dField = USE3D_BFIELD;
  bool cylSymm = true;

  
  o::Reals constFlowVelocity(3); 
  if(useConstantFlowVelocity) {
    constFlowVelocity = o::Reals(o::HostWrite<o::Real>({CONSTANT_FLOW_VELOCITY0,
    CONSTANT_FLOW_VELOCITY1, CONSTANT_FLOW_VELOCITY2}).write());
  }


  //Setting up of 2D magnetic field data 
  const auto& BField_2d = gm.Bfield_2d;
  const auto bX0 = gm.bGridX0;
  const auto bZ0 = gm.bGridZ0;
  const auto bDx = gm.bGridDx;
  const auto bDz = gm.bGridDz;
  const auto bGridNx = gm.bGridNx;
  const auto bGridNz = gm.bGridNz;
  

  //Setting up of 2D ion temperature data
  const auto& temIon_d = gm.temIon_d;
  auto x0Temp = gm.tempIonX0;
  auto z0Temp = gm.tempIonZ0;
  auto nxTemp = gm.tempIonNx;
  auto nzTemp = gm.tempIonNz;
  auto dxTemp = gm.tempIonDx;
  auto dzTemp = gm.tempIonDz;


  //Setting up of 2D ion density data
  const auto& densIon_d =gm.densIon_d;
  auto x0Dens = gm.densIonX0;
  auto z0Dens = gm.densIonZ0;
  auto nxDens = gm.densIonNx;
  auto nzDens = gm.densIonNz;
  auto dxDens = gm.densIonDx;
  auto dzDens = gm.densIonDz;

  ////Setting up of 2D electron temperature data
  const auto& temEl_d  = gm.temEl_d;
  auto x0Temp_el=gm.tempElX0;
  auto z0Temp_el=gm.tempElZ0;
  auto nxTemp_el=gm.tempElNx;
  auto nzTemp_el=gm.tempElNz;
  auto dxTemp_el=gm.tempElDx;
  auto dzTemp_el=gm.tempElDz;

  
  //Data for 3D interpolation
  auto& mesh = gm.mesh;
  const auto coords = mesh.coords();
  const auto mesh2verts = mesh.ask_elem_verts();
  const auto tIonVtx = mesh.get_array<o::Real>(o::VERT, "IonTempVtx");
  const auto densVtx = mesh.get_array<o::Real>(o::VERT, "IonDensityVtx");
  const auto tElVtx  = mesh.get_array<o::Real>(o::VERT, "ElTempVtx");
  const auto BField = o::Reals(); //o::Reals(mesh.get_array<o::Real>(o::VERT, "BField"));

  //Use of GITR generated random numbers
  const auto& testGitrPtclStepData = gp.testGitrPtclStepData;
  const auto testGDof = gp.testGitrStepDataDof;
  const auto testGNT = gp.testGitrStepDataNumTsteps;
  //const auto testGIind = gp.testGitrDataIoniRandInd;
  const auto iTimeStep = iTimePlusOne - 1;
  const auto collisionIndex1 = gp.testGitrCollisionRndn1Ind;
  const auto collisionIndex2 = gp.testGitrCollisionRndn2Ind;
  const auto collisionIndex3 = gp.testGitrCollisionRndxsiInd;
  auto& xfaces =gp.wallCollisionFaceIds;


  auto updatePtclPos = PS_LAMBDA(const int& e, const int& pid, const bool& mask){ 
    
    if(mask > 0&& elm_ids[pid] >= 0){
    
        o::LO el = elm_ids[pid];
        auto ptcl           = pid_ps(pid);
        auto charge         = charge_ps_d(pid);
        auto fid            = xfaces[ptcl];
    
        if(!charge || fid >=0)
          return; 

  	    auto posit          = p::makeVector3(pid, x_ps_d);
        auto posit_next     = p::makeVector3(pid, xtgt_ps_d);
        auto eField         = p::makeVector3(pid, efield_ps_d);
        auto vel            = p::makeVector3(pid, vel_ps_d);
        auto relvel         = o::zero_vector<3>();
        auto parallel_dir   = o::zero_vector<3>();
        auto perp1_dir      = o::zero_vector<3>();
        auto perp2_dir      = o::zero_vector<3>();
        auto pos2D          = o::zero_vector<3>();
        pos2D[0]            = sqrt(posit_next[0]*posit_next[0] + posit_next[1]*posit_next[1]);
        pos2D[1]            = 0;
        pos2D[2]            = posit_next[2];

        Omega_h::Vector<3> flowVelocity_radial;
        Omega_h::Vector<3> flowVelocity;
        Omega_h::Vector<3> b_field;

        auto dens=0.0;
        auto temp=0.0;
        auto temp_el=0.0;
        auto velIn = vel;

        if (useConstantFlowVelocity){

            for(auto i=0; i<3; ++i)
            flowVelocity_radial[i] = constFlowVelocity[i];

            o::Real theta = atan2(posit_next[1], posit_next[0]);

            flowVelocity[0] = cos(theta)*flowVelocity_radial[0] - sin(theta)*flowVelocity_radial[1];
            flowVelocity[1] = sin(theta)*flowVelocity_radial[0] + cos(theta)*flowVelocity_radial[1];
            flowVelocity[2] = flowVelocity_radial[2];

        }


        if (use2dInputFields || useConstantBField){

            p::interp2dVector(BField_2d, bX0, bZ0, bDx, bDz, bGridNx, bGridNz, posit_next, b_field, cylSymm, &ptcl);
      

        }

        else if (use3dField){

            auto bcc = o::zero_vector<4>();
            p::findBCCoordsInTet(coords, mesh2verts, posit_next, el, bcc);
            p::interpolate3dFieldTet(mesh2verts, BField, el, bcc, b_field); 

        }


        if (use2dInputFields){

            dens = p::interpolate2dField(densIon_d, x0Dens, z0Dens, dxDens, 
      	           dzDens, nxDens, nzDens, pos2D, true,1,0,false);
            temp = p::interpolate2dField(temIon_d, x0Temp, z0Temp, dxTemp,
                   dzTemp, nxTemp, nzTemp, pos2D, true, 1,0,false);
            temp_el= p::interpolate2dField(temEl_d, x0Temp_el, z0Temp_el, dxTemp_el,
                   dzTemp_el, nxTemp_el, nzTemp_el, pos2D, true, 1,0,false);

          //printf("density,ion_temp, el_temp 2D %.15f, %.15f, %.15f \n",dens, temp, temp_el);

        }
    
        else if (use3dField){
   
            auto bcc = o::zero_vector<4>();
            p::findBCCoordsInTet(coords, mesh2verts, posit_next, el, bcc);
            dens    = p::interpolateTetVtx(mesh2verts, densVtx, el, bcc, 1);
            temp    = p::interpolateTetVtx(mesh2verts, tIonVtx, el, bcc, 1);
            temp_el = p::interpolateTetVtx(mesh2verts, tElVtx, el, bcc, 1);

          //printf("density,ion_temp, el_temp 3D %.15f, %.15f, %.15f \n",dens, temp, temp_el);

        }   

        //Funvtion to test accuracy of 3D interpolation
        //p::test3Dinterpfunction(coords, mesh2verts,el,densVtx);

        auto nu_friction   =0.0;
        auto nu_deflection =0.0;
        auto nu_parallel   =0.0;
        auto nu_energy     =0.0;
        
        get_drag(vel, flowVelocity, nu_friction, nu_deflection, nu_parallel, nu_energy, temp,temp_el,
                dens,charge, ptcl, iTimeStep, debug);

        get_direc( vel,flowVelocity,b_field, parallel_dir,perp1_dir, perp2_dir,ptcl, iTimeStep, debug);
   
        relvel=vel-flowVelocity;
        auto velocityNorm   = Omega_h::norm(relvel);
        double drag  = -dt*nu_friction*velocityNorm/1.2;

        double n1  = 0.0;
        double n2  = 0.0;
        double xsi = 0.0;

        n1  = testGitrPtclStepData[ptcl*testGNT*testGDof + iTimeStep*testGDof + collisionIndex1];
        n2  = testGitrPtclStepData[ptcl*testGNT*testGDof + iTimeStep*testGDof + collisionIndex2];
        xsi = testGitrPtclStepData[ptcl*testGNT*testGDof + iTimeStep*testGDof + collisionIndex3];

        double coeff_par = 1.4142 * n1 * sqrt(nu_parallel * dt);
        double cosXsi = cos(2.0 * 3.14159265 * xsi);
        double sinXsi = sin(2.0 * 3.14159265 * xsi);
        double coeff_perp1 = cosXsi * sqrt(nu_deflection * dt * 0.5);
        double coeff_perp2 = sinXsi * sqrt(nu_deflection * dt * 0.5);

        double nuEdt=nu_energy*dt;
        if (nuEdt < -1.0)
            nuEdt = -1.0;

        if(debug){

            printf("coeff_par  for timestep %d is %g \n",iTimeStep, coeff_par);
            printf("cosXsi  for timestep %d is %g \n",iTimeStep, cosXsi);
            printf("sinXsi  for timestep %d is %g \n",iTimeStep, sinXsi);
            printf("coeff_perp1  for timestep %d is %g \n",iTimeStep, coeff_perp1);
            printf("coeff_perp2  for timestep %d is %g \n",iTimeStep, coeff_perp2);
            printf("n2  for timestep %d is %g \n",iTimeStep, n2);
            printf("Position partcle %d timestep %d is %.15e %.15e %.15e \n",ptcl, iTimeStep, posit_next[0],posit_next[1],posit_next[2]);
            printf("The velocities partcle %d timestep %dare %.15e %.15e %.15e \n", iTimeStep, ptcl, vel[0],vel[1],vel[2]);  

        }

          //vel=Omega_h::norm(vel)*parallel_dir+drag*relvel/velocityNorm; // This was if only drag force is implemented.
          vel=Omega_h::norm(vel)*(1-0.5*nuEdt)*((1+coeff_par)*parallel_dir+abs(n2) * 
              (coeff_perp1 * perp1_dir + coeff_perp2 * perp2_dir))+drag*relvel/velocityNorm;

        if (debug){ 

            printf("The velocities after updation from COULOMB collision partcle %d timestep %dare %.15e %.15e %.15e \n", ptcl, iTimeStep, vel[0],vel[1],vel[2]); 
          
        }

        vel_ps_d(pid,0)=vel[0];
        vel_ps_d(pid,1)=vel[1];
        vel_ps_d(pid,2)=vel[2];  
    }
  };
  ps::parallel_for(ptcls, updatePtclPos);

}

#endif

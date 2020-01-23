#ifndef GITRM_COULOMB_COLLISION_H
#define GITRM_COULOMB_COLLISION_H
#include "GitrmParticles.hpp"

OMEGA_H_DEVICE void get_drag(const Omega_h::Vector<3> &vel, const Omega_h::Vector<3> &flowVelocity,
double& nu_friction,double& nu_deflection, double& nu_parallel,double& nu_energy, const double temp,
const double temp_el, const double dens, const double charge, int ptcl_t, int iTimeStep_t, int debug)
{
  
  printf("Working on particle %d\n",ptcl_t);
  const double Q    = 1.60217662e-19;
  const double EPS0 = 8.854187e-12;
  const double pi   = 3.14159265;
  const auto MI     = 1.6737236e-27;
  const auto ME     = 9.10938356e-31;
  const double amu  = 184; 
  const double background_amu=4;
  const double background_Z = 1;
  //const double charge=1;
    
    Omega_h::Vector<3>relvel = vel - flowVelocity;
    auto velocityNorm   = Omega_h::norm(relvel);
    //parallel_dir=relvel/velocityNorm;
    
    auto lam_d = sqrt(EPS0*temp_el/(dens*pow(background_Z,2)*Q));
    auto lam = 4*pi*dens*pow(lam_d,3);

    auto gam_electron_background = 0.238762895*pow(charge,2)*log(lam)/(amu*amu);
     if(gam_electron_background < 0.0)
     {
        gam_electron_background=0.0;
     }  
    auto gam_ion_background = 0.238762895*pow(charge,2)*pow(background_Z,2)*log(lam)/(amu*amu);
     if(gam_ion_background < 0.0)
    {
      gam_ion_background=0.0;
    }
      
    auto nu_0_i = gam_electron_background*dens/pow(velocityNorm,3);      
    auto nu_0_e = gam_ion_background*dens/pow(velocityNorm,3);

    auto xx_i= pow(velocityNorm,2)*background_amu*MI/(2*temp*Q);
    auto xx_e= pow(velocityNorm,2)*ME/(2*temp_el*Q);
  
      
    auto psi_i       = 0.75225278*pow(xx_i,1.5);
    auto psi_prime_i = 1.128379*sqrt(xx_i);
    //auto psi_psiprime_i   = psi_i+psi_prime_i;
    auto psi_psiprime_psi2x_i = 1.128379*sqrt(xx_i)*expf(-xx_i);
      
    auto psi_e = 0.75225278*pow(xx_e,1.5); 
    auto psi_prime_e = 1.128379*sqrt(xx_e);
    //auto psi_psiprime_e = psi_e+psi_prime_e;
    auto psi_psiprime_psi2x_e = 1.128379*sqrt(xx_e)*expf(-xx_e);

    
    

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

    if (temp<=0.0|| temp_el<=0.0)
    {
      nu_friction=0.0;
      nu_deflection = 0.0;
      nu_parallel = 0.0;
      nu_energy = 0.0;
    }
    if (dens<=0)
    {
      nu_friction=0.0;
      nu_deflection = 0.0;
      nu_parallel = 0.0;
      nu_energy = 0.0;
    }

    if(debug && ptcl_t==6)
    {
      //printf("gam_electron_background is %g \n",gam_electron_background);
      //printf("gam_ion_background is %g \n",gam_ion_background);
      //printf("nu_0_i %g \n",nu_0_i);
      //printf("nu_0_e %g \n",nu_0_e);
      //printf("xx_i is %g \n",xx_i);
      //printf("xx_e is %g \n",xx_e);

      //printf("psi_i for timestep %d is %g \n",iTimeStep_t, psi_i);
      //printf("psi_prime_i for timestep %d is %g \n",iTimeStep_t, psi_prime_i);
      //printf("psi_psiprime_psi2x_i for timestep %d is %g \n",iTimeStep_t, psi_psiprime_psi2x_i);

      //printf("psi_e for timestep %d is %g \n",iTimeStep_t, psi_e);
      //printf("psi_prime_e  for timestep %d is %g \n",iTimeStep_t, psi_prime_e);
      //printf("psi_psiprime_psi2x_e  for timestep %d is %g \n",iTimeStep_t, psi_psiprime_psi2x_e);

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
   
        if(b_mag == 0.0)
        {   
            b_unit[0] = 1.0;
            b_unit[1] = 0.0;
            b_unit[2] = 0.0;
            b_mag     = 1.0;
        }
    


    Omega_h::Vector<3> relvel1 = vel ;// - flowVelocity;
    auto velocityNorm   = Omega_h::norm(relvel1);
    auto velocityNorm1  = p::osh_mag(relvel1);
    printf("The normal velocities are %g %g \n", velocityNorm, velocityNorm1);
    parallel_dir=relvel1/velocityNorm;
    
    auto s1 = p::osh_dot(parallel_dir, b_unit);
    auto s2 = sqrt(1.0-s1*s1);
    if(abs(s1) >=1.0) s2=0;
    perp1_dir=1.0/s2*(s1*parallel_dir-b_unit);
    perp2_dir=1.0/s2*(Omega_h::cross(parallel_dir, b_unit));
    //perp2_dir[0] = (1.0/s2)*(parallel_dir[1]*b_unit[2] - parallel_dir[2]*b_unit[1]);
    //perp2_dir[1] = (1.0/s2)*(parallel_dir[2]*b_unit[0] - parallel_dir[0]*b_unit[2]);
    //perp2_dir[2] = (1.0/s2)*(parallel_dir[0]*b_unit[1] - parallel_dir[1]*b_unit[0]);

      if(debug && ptcl_t==6)
    
    {

      printf("s1  for timestep %d is %g \n",iTimeStep_t, s1);
      printf("s2  for timestep %d is %g \n",iTimeStep_t, s2);
      printf("parallel_dir  for timestep %d is %g %g %g \n",iTimeStep_t, parallel_dir[0], parallel_dir[1], parallel_dir[2]);
      printf("b_field  for timestep %d is %g %g %g \n",iTimeStep_t, b_field[0], b_field[1],b_field[2]);
      printf("b_unit  for timestep %d is %g %g %g \n",iTimeStep_t, b_unit[0], b_unit[1],b_unit[2]);
      printf("perp1_dir for timestep %d is %g %g %g \n",iTimeStep_t, perp1_dir[0], perp1_dir[1], perp1_dir[2]);
      printf("perp2_dir for timestep %d is %g %g %g \n",iTimeStep_t, perp2_dir[0], perp2_dir[1], perp2_dir[2]);
      
    }  

    if (s2 == 0.0)
            {
                printf("Entering the inner s2 loop");
                perp1_dir[0] =  s1;//why no idea?
                perp1_dir[1] =  s2;//why no idea?

                perp2_dir[0] = parallel_dir[2];
                perp2_dir[1] = parallel_dir[0];
                perp2_dir[2] = parallel_dir[1];

                
                s1 = p::osh_dot(parallel_dir, perp2_dir);
                s2 = sqrt(1.0-s1*s1);
                perp1_dir = -1.0/s2*Omega_h::cross(parallel_dir,perp2_dir);
               
            }

}

inline void gitrm_coulomb_collision(PS* ptcls, int *iteration, const GitrmMesh& gm,
 const GitrmParticles& gp, double dt) {
  bool debug = 0;
  auto pid_ps = ptcls->get<PTCL_ID>();
  auto x_ps_d = ptcls->get<PTCL_POS>();
  auto xtgt_ps_d = ptcls->get<1>();
  auto efield_ps_d  = ptcls->get<PTCL_EFIELD>();
  auto vel_ps_d = ptcls->get<PTCL_VEL>();
  auto charge_ps_d = ptcls->get<PTCL_CHARGE>();
  //const auto& BField_2d = gm.Bfield_2d;
  printf("Entering coulomb_collision routine\n");
  //constexpr double CONSTANT_BFIELD[] = {0,0,-0.08};

  ///* 
  o::Reals bFieldConst(3); 
  if(USE_CONSTANT_BFIELD) {
    bFieldConst = o::Reals(o::HostWrite<o::Real>({CONSTANT_BFIELD0,
        CONSTANT_BFIELD1, CONSTANT_BFIELD2}).write());
  }
  //*/

  const auto& temIon_d = gm.temIon_d;
  auto x0Temp = gm.tempIonX0;
  auto z0Temp = gm.tempIonZ0;
  auto nxTemp = gm.tempIonNx;
  auto nzTemp = gm.tempIonNz;
  auto dxTemp = gm.tempIonDx;
  auto dzTemp = gm.tempIonDz;
  
  const auto& densIon_d =gm.densIon_d;
  auto x0Dens = gm.densIonX0;
  auto z0Dens = gm.densIonZ0;
  auto nxDens = gm.densIonNx;
  auto nzDens = gm.densIonNz;
  auto dxDens = gm.densIonDx;
  auto dzDens = gm.densIonDz;

  const auto& temEl_d  = gm.temEl_d;
  auto x0Temp_el=gm.tempElX0;
  auto z0Temp_el=gm.tempElZ0;
  auto nxTemp_el=gm.tempElNx;
  auto nzTemp_el=gm.tempElNz;
  auto dxTemp_el=gm.tempElDx;
  auto dzTemp_el=gm.tempElDz;
  //o::Write<o::Real> angle_d(mesh.nfaces());

  const auto& testGitrPtclStepData = gp.testGitrPtclStepData;
  const auto testGDof = gp.testGitrStepDataDof;
  const auto testGNT = gp.testGitrStepDataNumTsteps;
  //const auto testGIind = gp.testGitrDataIoniRandInd;
  const auto iTimeStep = iTimePlusOne - 1;
  const auto collisionIndex1 = gp.testGitrCollisionRndn1Ind;
  const auto collisionIndex2 = gp.testGitrCollisionRndn2Ind;
  const auto collisionIndex3 = gp.testGitrCollisionRndxsiInd;

  printf("The numbers are %d, %d %d %d %d %d \n", testGDof, testGNT, iTimeStep, collisionIndex1, 
    collisionIndex2, collisionIndex3);
  //auto  collisionIndex1 = gp.testGitrCollisionRndn1Ind;
  //auto  collisionIndex2 = collisionIndex1 + 1;
  //auto  collisionIndex3 = collisionIndex1 + 2;


  auto updatePtclPos = PS_LAMBDA(const int& e, const int& pid, const bool& mask) 
  { if(mask > 0 )
    {
    auto posit          = p::makeVector3(pid, x_ps_d);
    auto ptcl           = pid_ps(pid);
    auto charge         = charge_ps_d(pid);


    Omega_h::Vector<3> b_field;
    for(auto i=0; i<3; ++i)
      b_field[i] = bFieldConst[i];
    
    if(!charge)
      return; 

    auto posit_next     = p::makeVector3(pid, xtgt_ps_d);
    auto eField         = p::makeVector3(pid, efield_ps_d);
    auto vel            = p::makeVector3(pid, vel_ps_d);
    //auto bField       = o::zero_vector<3>();
    //auto b_field        = p::makeVector3FromArray (CONSTANT_BFIELD);
    auto flowVelocity   = p::makeVector3FromArray({0, 0, -20000.0});
    auto relvel         = o::zero_vector<3>();
    auto parallel_dir   = o::zero_vector<3>();
    auto perp1_dir      = o::zero_vector<3>();
    auto perp2_dir      = o::zero_vector<3>();
    auto pos2D          = o::zero_vector<3>();
    pos2D[0]            = sqrt(posit[0]*posit[0] + posit[1]*posit[1]);
    pos2D[1]            = 0;
    pos2D[2]            = posit[2];

    auto velIn = vel;

    auto dens = p::interpolate2dField(densIon_d, x0Dens, z0Dens, dxDens, 
            dzDens, nxDens, nzDens, pos2D, true,1,0,false);

    auto temp = p::interpolate2dField(temIon_d, x0Temp, z0Temp, dxTemp,
          dzTemp, nxTemp, nzTemp, pos2D, true, 1,0,false);

    auto temp_el= p::interpolate2dField(temEl_d, x0Temp_el, z0Temp_el, dxTemp_el,
          dzTemp_el, nxTemp_el, nzTemp_el, pos2D, true, 1,0,false);

    
    //printf("The components of magnetic fields are %g %g %g \n ", b_field[0], b_field[1], b_field[2]);

    auto nu_friction   =0.0;
    auto nu_deflection =0.0;
    auto nu_parallel   =0.0;
    auto nu_energy     =0.0;
    
    get_drag(vel, flowVelocity, nu_friction, nu_deflection, nu_parallel, nu_energy, temp,temp_el,
      dens,charge, ptcl, iTimeStep, debug);
    get_direc( vel,flowVelocity,b_field, parallel_dir,perp1_dir, perp2_dir,ptcl, iTimeStep, debug);
    // printf("\n The velocity in %d iteration %d are %g  %g %g %g %d \n",j,pid, vel[0], vel[1], 
    //vel[2],nu_friction,charge);

    relvel=vel-flowVelocity;
    auto velocityNorm   = Omega_h::norm(relvel);
    double drag  = -dt*nu_friction*velocityNorm/1.2;

    //Temporary
    // From IOnization recombination, use of indexing
    //if(useGitrRnd) {
    //    randGitr = testGitrPtclStepData[ptcl*testGNT*testGDof + iTimeStep*testGDof + testGIind];
    //    randn = randGitr;        
    //} 
    //else
    //    randn = rands[pid];
    double n1  = 0.5;
    double n2  = 0.5;
    double xsi = 0.75;

    n1  = testGitrPtclStepData[ptcl*testGNT*testGDof + iTimeStep*testGDof + collisionIndex1];
    n2  = testGitrPtclStepData[ptcl*testGNT*testGDof + iTimeStep*testGDof + collisionIndex2];
    xsi = testGitrPtclStepData[ptcl*testGNT*testGDof + iTimeStep*testGDof + collisionIndex3];

    double coeff_par = 1.4142 * n1 * sqrt(nu_parallel * dt);
    double cosXsi = cos(2.0 * 3.14159265 * xsi);
    double sinXsi = sin(2.0 * 3.14159265 * xsi);
    double coeff_perp1 = cosXsi * sqrt(nu_deflection * dt * 0.5);
    double coeff_perp2 = sinXsi * sqrt(nu_deflection * dt * 0.5);

        if(debug && ptcl==6)
    {


      printf("coeff_par  for timestep %d is %g \n",iTimeStep, coeff_par);
      printf("cosXsi  for timestep %d is %g \n",iTimeStep, cosXsi);
      printf("sinXsi  for timestep %d is %g \n",iTimeStep, sinXsi);
      printf("coeff_perp1  for timestep %d is %g \n",iTimeStep, coeff_perp1);
      printf("coeff_perp2  for timestep %d is %g \n",iTimeStep, coeff_perp2);
      printf("n2  for timestep %d is %g \n",iTimeStep, n2);
      

      
    }
    double nuEdt=nu_energy*dt;
          if (nuEdt < -1.0)
              nuEdt = -1.0;
    //vel=Omega_h::norm(vel)*parallel_dir+drag*relvel/velocityNorm;// This was if only drag force is implemented.
    vel=Omega_h::norm(vel)*(1-0.5*nuEdt)*((1+coeff_par)*parallel_dir+abs(n2) * 
      (coeff_perp1 * perp1_dir + coeff_perp2 * perp2_dir))+drag*relvel/velocityNorm;
    // printf("\n The velocity in %d iteration %d post collision are are %f %f %f \n",
    //j,pid, vel[0], vel[1], vel[2],nu_friction);

    vel_ps_d(pid,0)=vel[0];
    vel_ps_d(pid,1)=vel[1];
    vel_ps_d(pid,2)=vel[2];  
    }
  };
  ps::parallel_for(ptcls, updatePtclPos);

}

#endif

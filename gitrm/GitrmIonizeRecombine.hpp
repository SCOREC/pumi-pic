#ifndef GITRM_IONIZE_RECOMBINE_HPP
#define GITRM_IONIZE_RECOMBINE_HPP

#include "GitrmParticles.hpp"
#include <cstdlib>
#include <ctime>
#include <fstream>


class GitrmIonizeRecombine {
public:
  GitrmIonizeRecombine(const std::string &fName, bool charged=true);
  void initIonizeRecombRateData(const std::string &, int debug=0);
  bool chargedPtclTracking = true;
  o::Real ionizeTempGridMin = 0;
  o::Real ionizeDensGridMin = 0;
  o::Real ionizeTempGridDT = 0;
  o::Real ionizeDensGridDn = 0;
  o::LO ionizeTempGridN = 0;
  o::LO ionizeDensGridN = 0;
  o::Real recombTempGridMin = 0;
  o::Real recombDensGridMin = 0;
  o::Real recombTempGridDT = 0;
  o::Real recombDensGridDn = 0;
  o::LO recombTempGridN = 0;
  o::LO recombDensGridN = 0;
  
  o::Reals ionizationRates;
  o::Reals recombinationRates;
  o::Reals gridTempIonize;
  o::Reals gridDensIonize;
  o::Reals gridTempRec;
  o::Reals gridDensRec;
};

//passed in Temperature is in log, but density is not
// stored as 3component data, but interpolated from 2D grid in log scale
OMEGA_H_DEVICE o::Real interpolateRateCoeff(const o::Reals &data, 
  const o::Reals &gridTemp, const o::Reals &gridDens,
  const o::Real tem, const o::Real dens, const o::Real gridT0, 
  const o::Real gridD0, const o::Real dT, const o::Real dD, 
  const o::LO nT,  const o::LO nD, o::LO charge) {

  int debug = 0;
  o::LO indT = floor( (log10(tem) - gridT0)/dT );
  //o::LO indT = floor( (tem - gridT0)/dT );
  o::LO indN = floor( (log10(dens) - gridD0)/dD );
  if(indT < 0 || indT > nT-2)
    indT = 0;
  if(indN < 0 || indN > nD-2)
    indN = 0;
  //TODO
  if(charge > 74-1) //only Tungston
    {charge = 0;} 
  // TODO use log interpolation. This will lose accuracy 
  auto gridTnext = gridTemp[indT+1] ;//gridT0 + (indT+1)*dT;
  o::Real aT = pow(10.0, gridTnext) - tem;
  //o::Real aT = pow(10.0, gridTnext) - pow(10.0, tem);
  auto gridT = gridTemp[indT] ;//gridT0 + indT*dT;
  o::Real bT = tem - pow(10.0, gridT);
  //o::Real bT = pow(10.0, tem) - pow(10.0, gridT);
  o::Real abT = aT+bT;
  auto gridDnext = gridDens[indN+1] ;//gridD0 + (indN+1)*dD;
  o::Real aN = pow(10.0, gridDnext) - dens;
  auto gridD = gridDens[indN]; //gridD0 + indN*dD;
  o::Real bN = dens - pow(10.0, gridD);
  o::Real abN = aN + bN;
  if(debug)
    printf("InterpRate: gridTnext %g pow(10.0, gridTnext) %g  aT %g bT %g "
        "tem %g abT %g aN %g bN %g dens %g abN %g \n", 
         gridTnext, pow(10.0, gridTnext), aT, bT, tem, abT, aN, bN, dens, abN);
  o::Real fx_z1 = (aN*pow(10.0, data[charge*nT*nD + indT*nD + indN]) 
          + bN*pow(10.0, data[charge*nT*nD + indT*nD + indN + 1]))/abN;
  
  o::Real fx_z2 = (aN*pow(10.0, data[charge*nT*nD + (indT+1)*nD + indN]) 
          + bN*pow(10.0, data[charge*nT*nD + (indT+1)*nD + indN+1]))/abN;
  o::Real RClocal = (aT*fx_z1+bT*fx_z2)/abT;
  
  if(debug)
    printf("  interpRate: t %g n %g gridT0 %g log10(tem) %g indT %d  gridD0 %g "
      "log10(dens) %g indN %d dT %g dD %g nT %d nD %d charge %d fx_z1 %g fx_z2 %g "
      "RClocal(fxz) %g\n", tem,dens,gridT0, log10(tem), indT, gridD0, log10(dens),
      indN, dT,dD,nT,nD,charge, fx_z1, fx_z2, RClocal);

  o::Real rate;
  if(o::are_close(tem,0) || o::are_close(dens, 0))
    rate = 1.0e12;
  else {
    //double eps = 1e-30; //TODO
    //if(o::are_close(RClocal,0, eps,eps))
    //  debug = 1;
    //OMEGA_H_CHECK(!o::are_close(RClocal,0, eps,eps));

    rate = 1/(RClocal*dens);
    OMEGA_H_CHECK(!isnan(rate));
  }
  OMEGA_H_CHECK(!isnan(rate));

  return rate;
}

inline void gitrm_ionize(PS* ptcls, const GitrmIonizeRecombine& gir, 
  GitrmParticles& gp, const GitrmMesh& gm, o::Write<o::LO>& elm_ids, 
  bool debug = false) {
  auto& mesh = gm.mesh;
  auto use2DRatesData = USE_2DREADIN_IONI_REC_RATES;
  auto& densIon_d = gm.densIon_d;
  auto& temIon_d = gm.temIon_d;
  auto x0Dens = gm.densIonX0;
  auto z0Dens = gm.densIonZ0;
  auto nxDens = gm.densIonNx;
  auto nzDens = gm.densIonNz;
  auto dxDens = gm.densIonDx;
  auto dzDens = gm.densIonDz;
  auto x0Temp = gm.tempIonX0;
  auto z0Temp = gm.tempIonZ0;
  auto nxTemp = gm.tempIonNx;
  auto nzTemp = gm.tempIonNz;
  auto dxTemp = gm.tempIonDx;
  auto dzTemp = gm.tempIonDz;

  bool useGitrRnd = USE_GITR_RND_NUMS;
  //#ifdef USE_GITR_RND_NUMS
  if(!gp.ranIonization)
    gp.ranIonization = true;
  const auto& testGitrPtclStepData = gp.testGitrPtclStepData;
  const auto testGDof = gp.testGitrStepDataDof;
  const auto testGNT = gp.testGitrStepDataNumTsteps;
  const auto testGIind = gp.testGitrDataIoniRandInd;
  const auto iTimeStep = iTimePlusOne - 1;
  if(useGitrRnd)
    OMEGA_H_CHECK(gp.testGitrOptIoniRec);
  else
    OMEGA_H_CHECK(!gp.testGitrOptIoniRec);

  //#endif

  auto& xfaces_d = gp.wallCollisionFaceIds;
  auto dt = gp.timeStep;
  auto gridT0 = gir.ionizeTempGridMin;
  auto gridD0 = gir.ionizeDensGridMin;
  auto dTem = gir.ionizeTempGridDT;
  auto dDens = gir.ionizeDensGridDn;
  auto nTRates = gir.ionizeTempGridN;
  auto nDRates = gir.ionizeDensGridN;
  const auto& iRates = gir.ionizationRates; 
  const auto& gridTemp = gir.gridTempIonize;
  const auto& gridDens = gir.gridDensIonize;
  const auto coords = mesh.coords();
  const auto mesh2verts = mesh.ask_elem_verts();
  const auto tIonVtx = mesh.get_array<o::Real>(o::VERT, "IonTempVtx");
  const auto densVtx = mesh.get_array<o::Real>(o::VERT, "IonDensityVtx"); 
  
  auto tempWrite = o::deep_copy(mesh.get_array<o::Real>(o::VERT, "IonTempVtx"));

  //const auto& tIonVtx = gm.tempIonVtx_d;
  //const auto& densVtx = gm.densIonVtx_d;
  auto pid_ps = ptcls->get<PTCL_ID>();
  auto new_pos = ptcls->get<PTCL_NEXT_POS>();
  auto charge_ps = ptcls->get<PTCL_CHARGE>();
  auto first_ionizeZ_ps = ptcls->get<PTCL_FIRST_IONIZEZ>();
  auto prev_ionize_ps = ptcls->get<PTCL_PREV_IONIZE>();
  auto first_ionizeT_ps = ptcls->get<PTCL_FIRST_IONIZET>();
  auto psCapacity = ptcls->capacity();
  //delete later
  auto pos_prev=ptcls->get<0>();

  //TODO replace by Kokkos random
  o::HostWrite<o::Real> rnd_h(psCapacity);
  for(auto i=0; i<psCapacity; ++i) {
    rnd_h[i] = (double)(std::rand())/RAND_MAX;
  }
  auto rands = o::Reals(o::Write<o::Real>(rnd_h));

  auto lambda = PS_LAMBDA(const int &elem, const int &pid, const int &mask) {
    // invalid elem_ids init to -1
    if(mask > 0 && elm_ids[pid] >= 0) {
     // element of next_pos
      o::LO el = elm_ids[pid];
      auto ptcl = pid_ps(pid);
      auto pos = p::makeVector3(pid, new_pos);
      auto charge = charge_ps(pid);


      auto pos_previous = p::makeVector3(pid, pos_prev);

      //printf("IONIZE particle %d timestep %d positions %.15e, %.15e, %.15e\n", ptcl, iTimeStep, pos_previous[0], pos_previous[1], pos_previous[2]);
      //printf("IONIZE particle %d timestep %d positions %.15e, %.15e, %.15e\n", ptcl, iTimeStep, pos[0], pos[1], pos[2]);
      o::Real tlocal = 0;
      o::Real nlocal = 0;
      if(!use2DRatesData) {
        auto bcc = o::zero_vector<4>();
        p::findBCCoordsInTet(coords, mesh2verts, pos, el, bcc);
        tlocal = p::interpolateTetVtx(mesh2verts, tIonVtx, el, bcc, 1);
        nlocal = p::interpolateTetVtx(mesh2verts, densVtx, el, bcc, 1);
      }
      if(charge > 74-1) //W=74, charge index=73
        charge = 0;
      // from data array
      if(use2DRatesData) {
        // projecting point to y=0 plane, since 2D data is on const-y plane,
        // same as a 2D plane through the particle position, with x'=r=(sqrt(x^2+y^2).
        bool cylSymm = true;
        auto dens = p::interpolate2dField(densIon_d, x0Dens, z0Dens, dxDens, 
          dzDens, nxDens, nzDens, pos, cylSymm, 1,0,false);
        auto temp = p::interpolate2dField(temIon_d, x0Temp, z0Temp, dxTemp,
          dzTemp, nxTemp, nzTemp, pos, cylSymm, 1,0,false);
        
        if(debug)
          printf("Ionization point: ptcl %d dens2D %g temp2D %g tlocal %g nlocal %g "
            " pos %g %g %g nxTemp %d nzTemp %d\n", 
            ptcl, dens, temp, tlocal, nlocal, pos[0], pos[1], pos[2], nxTemp, nzTemp);
        nlocal = dens;
        tlocal = temp;
      } 
      o::Real rateIon = interpolateRateCoeff(iRates, gridTemp, gridDens, tlocal, 
        nlocal, gridT0, gridD0, dTem, dDens, nTRates, nDRates, charge);

      o::Real P1 = 1.0 - exp(-dt/rateIon);
      double randn = 0;
      int gitrInd = -1; 
      double randGitr = 0;
      if(useGitrRnd) {
        randGitr = testGitrPtclStepData[ptcl*testGNT*testGDof + iTimeStep*testGDof + testGIind];
        randn = randGitr;        
      } else
        randn = rands[pid];
      
      auto xfid = xfaces_d[ptcl];
      auto first_iz = first_ionizeZ_ps(pid);
      if(xfid < 0) {
        if(randn <= P1)
          charge_ps(pid) = charge+1;
      }  prev_ionize_ps(pid) = 1;
        // Z=0 unitialized or specific z(height) value ? question
      if(o::are_close(first_iz, 0)) {
          first_ionizeZ_ps(pid) = pos[2]; // z
      } else if(o::are_close(first_iz, 0)) {
        auto fit = first_ionizeT_ps(pid);
        first_ionizeT_ps(pid) = fit + dt;
      } 
      if(useGitrRnd && debug)
        gitrInd = ptcl*testGNT*testGDof + iTimeStep*testGDof+testGIind;
      if(debug)
        printf("ionizable %d ptcl %d charge %d randn %g P1 %g rateIon %g dt %g @ %d\n",
          xfid<0, ptcl, charge_ps(pid), randn, P1, rateIon, dt, gitrInd);
      
    } //mask 
  };
  ps::parallel_for(ptcls,lambda, "ionizeKernel");
} 


inline void gitrm_recombine(PS* ptcls, const GitrmIonizeRecombine& gir, 
   GitrmParticles& gp, const GitrmMesh& gm, o::Write<o::LO>& elm_ids, 
   bool debug = false) {
  auto& mesh = gm.mesh;
  auto& densIon_d = gm.densIon_d;
  auto& temIon_d = gm.temIon_d;
  auto x0Dens = gm.densIonX0;
  auto z0Dens = gm.densIonZ0;
  auto nxDens = gm.densIonNx;
  auto nzDens = gm.densIonNz;
  auto dxDens = gm.densIonDx;
  auto dzDens = gm.densIonDz;
  auto x0Temp = gm.tempIonX0;
  auto z0Temp = gm.tempIonZ0;
  auto nxTemp = gm.tempIonNx;
  auto nzTemp = gm.tempIonNz;
  auto dxTemp = gm.tempIonDx;
  auto dzTemp = gm.tempIonDz;

  auto useGitrRnd = USE_GITR_RND_NUMS;
  if(!gp.ranRecombination)
    gp.ranRecombination = true;
  //#ifdef USE_GITR_RND_NUMS
  const auto& testGitrPtclStepData = gp.testGitrPtclStepData;
  const auto testGDof = gp.testGitrStepDataDof;
  const auto testGrecInd = gp.testGitrDataRecRandInd;
  const auto testGNT = gp.testGitrStepDataNumTsteps;
  const auto iTimeStep = iTimePlusOne - 1;
  if(useGitrRnd)
    OMEGA_H_CHECK(gp.testGitrOptIoniRec);
  else
    OMEGA_H_CHECK(!gp.testGitrOptIoniRec);
  //#endif

  auto use2DRatesData = USE_2DREADIN_IONI_REC_RATES;
  auto& xfaces_d = gp.wallCollisionFaceIds;
  auto dt = gp.timeStep;
  auto gridT0 = gir.recombTempGridMin;
  auto gridD0 = gir.recombDensGridMin;
  auto dTem = gir.recombTempGridDT;
  auto dDens = gir.recombDensGridDn;
  auto nTRates = gir.recombTempGridN;
  auto nDRates = gir.recombDensGridN;
  const auto& rRates = gir.recombinationRates; 
  const auto& gridTemp = gir.gridTempRec;
  const auto& gridDens = gir.gridDensRec;  
  const auto coords = mesh.coords();
  const auto mesh2verts = mesh.ask_elem_verts();
  const auto tIonVtx = mesh.get_array<o::Real>(o::VERT, "IonTempVtx");
  const auto densVtx = mesh.get_array<o::Real>(o::VERT, "IonDensityVtx");
  //const auto& tIonVtx = gm.densElVtx_d;
  //const auto& densVtx = gm.tempElVtx_d;
  auto pid_ps = ptcls->get<PTCL_ID>();
  auto new_pos = ptcls->get<PTCL_NEXT_POS>();
  auto charge_ps = ptcls->get<PTCL_CHARGE>();
  auto first_ionizeZ_ps = ptcls->get<PTCL_FIRST_IONIZEZ>();
  auto prev_recomb_ps = ptcls->get<PTCL_PREV_RECOMBINE>();
  auto psCapacity = ptcls->capacity();

  //TODO FIXME replace by Kokkos random
  o::HostWrite<o::Real> rnd_h(psCapacity);
  for(auto i=0; i<psCapacity; ++i) {
    rnd_h[i] = (double)(std::rand())/RAND_MAX;
  }
  auto rands = o::Reals(o::Write<o::Real>(rnd_h));


  // is elm_ids[pid] >= 0 make sure ptcl not intersected bdry ?????????
  auto lambda = PS_LAMBDA(const int &elem, const int &pid, const int &mask) {
    if(mask > 0 && elm_ids[pid] >= 0) {
      auto el = elm_ids[pid];
      auto ptcl = pid_ps(pid);
      auto charge = charge_ps(pid);
      auto pos = p::makeVector3(pid, new_pos);
      o::Real rateRecomb = 0;
      o::Real P1 = 0;
      if(charge > 0) {
        o::Real tlocal = 0;
        o::Real nlocal = 0;
        if(!use2DRatesData) {
          auto bcc = o::zero_vector<4>();
          p::findBCCoordsInTet(coords, mesh2verts, pos, el, bcc);
          // from tags
          tlocal = p::interpolateTetVtx(mesh2verts, tIonVtx, el, bcc, 1);
          nlocal = p::interpolateTetVtx(mesh2verts, densVtx, el, bcc, 1);
        }
        // from data array
        if(use2DRatesData) {
          //cylindrical symmetry, height (z) is same.
          // projecting point to y=0 plane, since 2D data is on const-y plane.
          bool cylSymm = true;
          auto dens = p::interpolate2dField(densIon_d, x0Dens, z0Dens, dxDens, 
            dzDens, nxDens, nzDens, pos, cylSymm,1,0,false);
          auto temp = p::interpolate2dField(temIon_d, x0Temp, z0Temp, dxTemp,
            dzTemp, nxTemp, nzTemp, pos, cylSymm,1,0,false);
          
          if(debug)
            printf("Recomb Dens: ptcl %d x0 %g z0 %g dx %g dz %g nx %d " 
            " nz %d \n", ptcl, x0Dens, z0Dens, dxDens, dzDens, nxDens, nzDens);
          if(debug)
            printf("Recomb Temp: ptcl %d x0 %g z0 %g dx %g dz %g nx %d " 
            " nz %d \n", ptcl, x0Temp, z0Temp, dxTemp,dzTemp, nxTemp, nzTemp);  
          if(debug)
            printf("Recomb point: ptcl %d temp2D %g dens2D %g t3D %g d3D %g pos %g %g %g \n", 
              ptcl, temp, dens, tlocal, nlocal, pos[0], pos[1], pos[2]);
          nlocal = dens;
          tlocal = temp;
        }
        // rate is from global data
        rateRecomb = interpolateRateCoeff(rRates, gridTemp, gridDens, tlocal,
         nlocal, gridT0, gridD0, dTem, dDens, nTRates, nDRates, charge);
        P1 = 1.0 - exp(-dt/rateRecomb);

        double randn = 0;
        double randGitr = 0;
        int gitrInd = -1;
        if(useGitrRnd) {
          randGitr = testGitrPtclStepData[ptcl*testGNT*testGDof + 
            iTimeStep*testGDof + testGrecInd];
          randn = randGitr;
        }
        else 
          randn = rands[pid];

        auto xfid = xfaces_d[ptcl];
        auto first_iz = first_ionizeZ_ps(pid);
        if(xfid < 0 && randn <= P1) {
          charge_ps(pid) = charge-1;
          prev_recomb_ps(pid) = 1;
        }

        if(useGitrRnd && debug)
          gitrInd = ptcl*testGNT*testGDof+iTimeStep*testGDof+testGrecInd;
        
        if(debug)
          printf("recomb %d ptcl %d charge %d randn %g P1 %g rateRecomb %g @ %d\n", 
            xfid<0, ptcl, charge_ps(pid), randn, P1, rateRecomb, gitrInd);
      }
    } //mask 
  };
  ps::parallel_for(ptcls, lambda, "RecombineKernel");
} 

#endif

#ifndef GITRM_IONIZE_RECOMBINE_HPP
#define GITRM_IONIZE_RECOMBINE_HPP

#include "GitrmParticles.hpp"
#include <cstdlib>
#include <ctime>
#include <fstream>

class GitrmIonizeRecombine {
public:
  GitrmIonizeRecombine(const std::string &fName, bool charged=true);
  void initIonizeRecombRateData(const std::string &fName, int debug=0);
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
  
  //TODO set to 0 after testing
  o::LO useReadInRatesData = 0;

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
  const o::LO nT,  const o::LO nD, const o::LO charge) {

  o::LO indT = floor( (log10(tem) - gridT0)/dT );
  //o::LO indT = floor( (tem - gridT0)/dT );
  o::LO indN = floor( (log10(dens) - gridD0)/dD );
  if(indT < 0 || indT > nT-2)
    indT = 0;
  if(indN < 0 || indN > nD-2)
    indN = 0;
  // TODO use log interpolation. This will loose accuracy 
  auto gridTnext = gridTemp[indT+1] ;//gridT0 + (indT+1)*dT;
  //o::Real aT = pow(10.0, gridTnext) - tem;
  o::Real aT = pow(10.0, gridTnext) - pow(10.0, tem);
  auto gridT = gridTemp[indT] ;//gridT0 + indT*dT;
  //o::Real bT = tem - pow(10.0, gridT);
  o::Real bT = pow(10.0, tem) - pow(10.0, gridT);
  o::Real abT = aT+bT;
  auto gridDnext = gridDens[indN+1] ;//gridD0 + (indN+1)*dD;
  o::Real aN = pow(10.0, gridDnext) - dens;
  auto gridD = gridDens[indN]; //gridD0 + indN*dD;
  o::Real bN = dens - pow(10.0, gridD);
  o::Real abN = aN + bN;

  o::Real fx_z1 = (aN*pow(10.0, data[charge*nT*nD + indT*nD + indN]) 
          + bN*pow(10.0, data[charge*nT*nD + indT*nD + indN + 1]))/abN;
  
  o::Real fx_z2 = (aN*pow(10.0, data[charge*nT*nD + (indT+1)*nD + indN]) 
          + bN*pow(10.0, data[charge*nT*nD + (indT+1)*nD + indN+1]))/abN;
  o::Real rate = (aT*fx_z1+bT*fx_z2)/abT;

  return rate;
}

// TODO split 
// dt is timestep
inline void gitrm_ionize(SCS* scs, const GitrmIonizeRecombine& gir, 
  const GitrmParticles& gp, const GitrmMesh& gm, o::Write<o::LO>& elm_ids, 
  bool debug = false) {
  auto& mesh = gm.mesh;
  auto use2DRatesData = gir.useReadInRatesData;
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

  auto& xfaces_d = gp.collisionPointFaceIds;
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
  //const auto& tIonVtx = gm.tempIonVtx_d;
  //const auto& densVtx = gm.densIonVtx_d;
  auto pid_scs = scs->get<PTCL_ID>();
  auto new_pos = scs->get<PTCL_NEXT_POS>();
  auto charge_scs = scs->get<PTCL_CHARGE>();
  auto first_ionizeZ_scs = scs->get<PTCL_FIRST_IONIZEZ>();
  auto prev_ionize_scs = scs->get<PTCL_PREV_IONIZE>();
  auto first_ionizeT_scs = scs->get<PTCL_FIRST_IONIZET>();
  auto scsCapacity = scs->capacity();
  o::HostWrite<o::Real> rnd1(scsCapacity);
  std::srand(time(NULL));
  for(auto i=0; i<scsCapacity; ++i) {
    rnd1[i] = (double)(std::rand())/RAND_MAX;
  }
  o::Write<o::Real>rand1(rnd1);
  auto rands = o::Reals(rand1);
  auto lambda = SCS_LAMBDA(const int &elem, const int &pid, const int &mask) {
    // invalid elem_ids init to -1
    if(mask > 0 && elm_ids[pid] >= 0) {
     // element of next_pos
      o::LO el = elm_ids[pid];
      auto ptcl = pid_scs(pid);
      auto pos = p::makeVector3(pid, new_pos);
      auto charge = charge_scs(pid);
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
        //TODO move this to a unit test
        auto pos2D = o::zero_vector<3>();
        //cylindrical symmetry, height (z) is same.
        pos2D[0] = sqrt(pos[0]*pos[0] + pos[1]*pos[1]);
        // projecting point to y=0 plane, since 2D data is on const-y plane.
        // meaningless to include non-zero y coord of target plane.
        pos2D[1] = 0;
        auto dens = p::interpolate2dField(densIon_d, x0Dens, z0Dens, dxDens, 
          dzDens, nxDens, nzDens, pos2D, false, 1,0,debug);
        auto temp = p::interpolate2dField(temIon_d, x0Temp, z0Temp, dxTemp,
          dzTemp, nxTemp, nzTemp, pos2D, false, 1,0,debug);

        if(debug)
          printf("Ioni Dens: x0 %g z0 %g dx %g dz %g nx %d " 
          " nz %d \n", x0Dens, z0Dens, dxDens, dzDens, nxDens, nzDens);
        if(debug)
          printf("Ioni Temp: x0 %g z0 %g dx %g dz %g nx %d " 
          " nz %d \n", x0Temp, z0Temp, dxTemp,dzTemp, nxTemp, nzTemp);        
        if(debug)
          printf("Ionization point: temp2D %g dens2D %g t3D %g d3D %g "
            " pos2D %g %g %g nxTemp %d nzTemp %d\n", 
            temp, dens, tlocal, nlocal, pos2D[0], pos2D[1], pos2D[2], 
              nxTemp, nzTemp);
        nlocal = dens;
        tlocal = temp;
        
      }
      
      o::Real rate = interpolateRateCoeff(iRates, gridTemp, gridDens, tlocal, 
        nlocal, gridT0, gridD0, dTem, dDens, nTRates, nDRates, charge);

      //TODO check rate !=0
      //OMEGA_H_CHECK(!p::almost_equal(rate,0));
			o::Real rateIon = 1/(rate*nlocal);
      OMEGA_H_CHECK(!std::isnan(rateIon));
			if(p::almost_equal(tlocal,0) || p::almost_equal(nlocal, 0))
				rateIon = 1.0e12;

    	o::Real P1 = 1.0 - exp(-dt/rateIon);
      auto randn = rands[pid];
      auto xfid = xfaces_d[pid];
      auto first_iz = first_ionizeZ_scs(pid);
    	if(xfid < 0) {
		    if(randn <= P1)
			  	charge_scs(pid) = charge+1;
				prev_ionize_scs(pid) = 1;
        // Z=0 unitialized or specific z(height) value ? question
		    if(p::almost_equal(first_iz, 0))
		      first_ionizeZ_scs(pid) = pos[2]; // z
    	}	else if(p::almost_equal(first_iz, 0)) {
        auto fit = first_ionizeT_scs(pid);
	      first_ionizeT_scs(pid) = fit + dt;
      }
      if(debug)
        printf("ionizable %d ptcl%d charge %d randn %g P1 %g rateIon %g rateInterp %g dt %g\n",
          xfid<0, ptcl, charge_scs(pid), randn, P1, rateIon, rate, dt);
	  } //mask 
	};
  scs->parallel_for(lambda);
} 


inline void gitrm_recombine(SCS* scs, const GitrmIonizeRecombine& gir, 
   const GitrmParticles& gp, const GitrmMesh& gm, o::Write<o::LO>& elm_ids, 
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

  auto use2DRatesData = gir.useReadInRatesData;
  auto& xfaces_d = gp.collisionPointFaceIds;
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
  auto pid_scs = scs->get<PTCL_ID>();
  auto new_pos = scs->get<PTCL_NEXT_POS>();
  auto charge_scs = scs->get<PTCL_CHARGE>();
  auto first_ionizeZ_scs = scs->get<PTCL_FIRST_IONIZEZ>();
  auto prev_recombination_scs = scs->get<PTCL_PREV_RECOMBINE>();
  auto scsCapacity = scs->capacity();
  o::HostWrite<o::Real> rnd1(scsCapacity);
  std::srand(time(NULL));
  for(auto i=0; i<scsCapacity; ++i) {
    rnd1[i] = (double)(std::rand())/RAND_MAX;
  }
  o::Write<o::Real>rand1(rnd1);
  auto rands = o::Reals(rand1);

  auto lambda = SCS_LAMBDA(const int &elem, const int &pid, const int &mask) {
    if(mask > 0 && elm_ids[pid] >= 0) {
      auto el = elm_ids[pid];
      auto ptcl = pid_scs(pid);
      auto charge = charge_scs(pid);
      auto pos = p::makeVector3(pid, new_pos);

      o::Real rateRecomb = 0;
      o::Real rate = 0;
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
          //TODO move this to a unit test
          auto pos2D = o::zero_vector<3>();
          //cylindrical symmetry, height (z) is same.
          pos2D[0] = sqrt(pos[0]*pos[0] + pos[1]*pos[1]);
          // projecting point to y=0 plane, since 2D data is on const-y plane.
          // meaningless to include non-zero y coord of target plane.
          pos2D[1] = 0;
          auto dens = p::interpolate2dField(densIon_d, x0Dens, z0Dens, dxDens, 
            dzDens, nxDens, nzDens, pos2D, false,1,0,debug);
          auto temp = p::interpolate2dField(temIon_d, x0Temp, z0Temp, dxTemp,
            dzTemp, nxTemp, nzTemp, pos2D, false,1,0,debug);

          if(debug)
            printf("Recomb Dens: x0 %g z0 %g dx %g dz %g nx %d " 
            " nz %d \n", x0Dens, z0Dens, dxDens, dzDens, nxDens, nzDens);
          if(debug)
            printf("Recomb Temp: x0 %g z0 %g dx %g dz %g nx %d " 
            " nz %d \n", x0Temp, z0Temp, dxTemp,dzTemp, nxTemp, nzTemp);  
          if(debug)
            printf("Recomb point: temp2D %g dens2D %g t3D %g d3D %g pos2D %g %g %g \n", 
              temp, dens, tlocal, nlocal, pos2D[0], pos2D[1], pos2D[2]);
          nlocal = dens;
          tlocal = temp;
        }
        // rate is from global data
        rate = interpolateRateCoeff(rRates, gridTemp, gridDens, tlocal,
         nlocal, gridT0, gridD0, dTem, dDens, nTRates, nDRates, charge);
        rateRecomb = 1/(rate*nlocal);
        if(p::almost_equal(tlocal,0) || p::almost_equal(nlocal, 0)) 
          rateRecomb = 1.0e12;

        P1 = 1.0 - exp(-dt/rateRecomb);
      }

      auto randn = rands[pid];
      auto xfid = xfaces_d[pid];
      auto first_iz = first_ionizeZ_scs(pid);
      if(xfid < 0 && randn <= P1) {
        charge_scs(pid) = charge-1;
        prev_recombination_scs(pid) = 1;
      }

      if(debug)
        printf("recomb %d ptcl %d charge %d randn %g P1 %g rateRecomb %g rateInterp %g\n", 
          xfid<0, ptcl, charge_scs(pid), randn, P1, rateRecomb, rate);
    } //mask 
  };
  scs->parallel_for(lambda);
} 

#endif
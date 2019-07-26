#ifndef GITRM_IONIZE_RECOMBINE_HPP
#define GITRM_IONIZE_RECOMBINE_HPP

#include "GitrmParticles.hpp"
#include <cstdlib>
#include <ctime>
#include <fstream>

class GitrmIonizeRecombine {
public:
  GitrmIonizeRecombine(const std::string &fName);
  void initIonizeRecombRateData(const std::string &fName);
  o::Real IONIZE_TEM_GRID_MIN = 0;
  o::Real IONIZE_DENS_GRID_MIN = 0;
  o::Real IONIZE_TEM_GRID_UNIT = 0;
  o::Real IONIZE_DENSITY_GRID_UNIT = 0;
  o::LO IONIZE_TEM_GRID_SIZE = 0;
  o::LO IONIZE_DENSITY_GRID_SIZE = 0;
  o::Real RECOMBINE_TEM_GRID_MIN = 0;
  o::Real RECOMBINE_DENS_GRID_MIN = 0;
  o::Real RECOMBINE_TEM_GRID_UNIT = 0;
  o::Real RECOMBINE_DENSITY_GRID_UNIT = 0;
  o::LO RECOMBINE_TEM_GRID_SIZE = 0;
  o::LO RECOMBINE_DENSITY_GRID_SIZE = 0;

  o::Reals ionizationRates;
  o::Reals recombinationRates;
  o::Reals gridTempIonize;
  o::Reals gridDensIonize;
  o::Reals gridTempRec;
  o::Reals gridDensRec;
};


// stored as 3component data, but interpolated from 2D grid in log scale
OMEGA_H_DEVICE o::Real interpolateRateCoeff(const o::Reals &data, 
  const o::Reals &gridTemp, const o::Reals &gridDens,
  const o::Real tem, const o::Real dens, const o::Real gridT0, 
  const o::Real gridD0, const o::Real dT, const o::Real dD, 
  const o::LO nT,  const o::LO nD, const o::LO charge) {

  o::LO indT = floor( (log10(tem) - gridT0)/dT );
  o::LO indN = floor( (log10(dens) - gridD0)/dD );
  if(indT < 0 || indT > nT-2)
    indT = 0;
  if(indN < 0 || indN > nD-2)
    indN = 0;
  // TODO use log interpolation. This will loose accuracy 
  auto gridTnext = gridTemp[indT+1] ;//gridT0 + (indT+1)*dT;
  o::Real aT = pow(10.0, gridTnext) - tem;
  auto gridT = gridTemp[indT] ;//gridT0 + indT*dT;
  o::Real bT = tem - pow(10.0, gridT);
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

// dt is timestep
inline void gitrm_ionize(o::Mesh& mesh, SCS* scs, const GitrmIonizeRecombine& gir, 
  const GitrmParticles& gp, o::LOs& elm_ids) {
  auto& xfaces_d = gp.collisionPointFaceIds;
  auto dt = gp.timeStep;
  auto gridT0 = gir.IONIZE_TEM_GRID_MIN;
  auto gridD0 = gir.IONIZE_DENS_GRID_MIN;
  auto dTem = gir.IONIZE_TEM_GRID_UNIT;
  auto dDens = gir.IONIZE_DENSITY_GRID_UNIT;
  auto nTRates = gir.IONIZE_TEM_GRID_SIZE;
  auto nDRates = gir.IONIZE_DENSITY_GRID_SIZE;
  const auto& iRates = gir.ionizationRates; 
  const auto& gridTemp = gir.gridTempIonize;
  const auto& gridDens = gir.gridDensIonize;
  const auto coords = mesh.coords();
  const auto mesh2verts = mesh.ask_elem_verts();
  const auto tIonVtx = mesh.get_array<o::Real>(o::VERT, "TionVtx");
  const auto densVtx = mesh.get_array<o::Real>(o::VERT, "densityVtx");
  auto pos_scs = scs->get<PTCL_POS>();
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
    o::LO el = elm_ids[pid];
    if(mask > 0 && el >= 0) {
      //o::LO debug = 1;
      auto pos = p::makeVector3(pid, pos_scs);
      auto charge = charge_scs(pid);
  	  auto bcc = o::zero_vector<4>();
      p::findBCCoordsInTet(coords, mesh2verts, pos, el, bcc);
      // from tags
      o::Real tlocal = p::interpolateTet(tIonVtx, bcc, mesh2verts, el, 1);
	    o::Real nlocal = p::interpolateTet(densVtx, bcc, mesh2verts, el, 1);

      if(charge > 74-1) //W=74, charge index=73
        charge = 0;
	    // from data array
			o::Real rate = interpolateRateCoeff(iRates, gridTemp, gridDens, tlocal, 
        nlocal, gridT0, gridD0, dTem, dDens, nTRates, nDRates, charge);
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
	  } //mask 
	};
  scs->parallel_for(lambda);
} 


inline void gitrm_recombine(o::Mesh& mesh, SCS* scs, const GitrmIonizeRecombine& gir, 
   const GitrmParticles& gp, o::LOs& elm_ids) {
  auto& xfaces_d = gp.collisionPointFaceIds;
  auto dt = gp.timeStep;
  auto gridT0 = gir.RECOMBINE_TEM_GRID_MIN;
  auto gridD0 = gir.RECOMBINE_DENS_GRID_MIN;
  auto dTem = gir.RECOMBINE_TEM_GRID_UNIT;
  auto dDens = gir.RECOMBINE_DENSITY_GRID_UNIT;
  auto nTRates = gir.RECOMBINE_TEM_GRID_SIZE;
  auto nDRates = gir.RECOMBINE_DENSITY_GRID_SIZE;
  const auto& rRates = gir.recombinationRates; 
  const auto& gridTemp = gir.gridTempRec;
  const auto& gridDens = gir.gridDensRec;  
  const auto coords = mesh.coords();
  const auto mesh2verts = mesh.ask_elem_verts();
  const auto tIonVtx = mesh.get_array<o::Real>(o::VERT, "TionVtx");
  const auto densVtx = mesh.get_array<o::Real>(o::VERT, "densityVtx");
  auto pos_scs = scs->get<PTCL_POS>();
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
    auto el = elm_ids[pid];
    if(mask > 0 && el >= 0) {
      //o::LO debug = 1;
      auto charge = charge_scs(pid);
      auto pos = p::makeVector3(pid, pos_scs);

      o::Real P1 = 0;
      if(charge > 0) {
        auto bcc = o::zero_vector<4>();
        p::findBCCoordsInTet(coords, mesh2verts, pos, el, bcc);
        // from tags
        o::Real tlocal = p::interpolateTet(tIonVtx, bcc, mesh2verts, el, 1);
        o::Real nlocal = p::interpolateTet(densVtx, bcc, mesh2verts, el, 1);
        // from data array
        o::Real rate = interpolateRateCoeff(rRates, gridTemp, gridDens, tlocal,
         nlocal, gridT0, gridD0, dTem, dDens, nTRates, nDRates, charge);
        o::Real rateIon = 1/(rate*nlocal);
        if(p::almost_equal(tlocal,0) || p::almost_equal(nlocal, 0)) 
          rateIon = 1.0e12;

        P1 = 1.0 - exp(-dt/rateIon);
      }

      auto randn = rands[pid];
      auto xfid = xfaces_d[pid];
      auto first_iz = first_ionizeZ_scs(pid);
      if(xfid < 0 && randn <= P1) {
        charge_scs(pid) = charge-1;
        prev_recombination_scs(pid) = 1;
      }
    } //mask 
  };
  scs->parallel_for(lambda);
} 


#endif
#include "GitrmIonizeRecombine.hpp"
#include "GitrmInputOutput.hpp"
#include "Omega_h_for.hpp" //testing

GitrmIonizeRecombine::GitrmIonizeRecombine(const std::string &fName,
  bool chargedTracking) {
  chargedPtclTracking = chargedTracking;
  initIonizeRecombRateData(fName);
}

// ADAS_Rates_W_structure for Tungsten W(z=74)
void GitrmIonizeRecombine::initIonizeRecombRateData( const std::string &fName, int debug) {
  std::cout<< "Loading Ionization data from " << fName << "\n" ;
  // not reading: gridChargeState_Ionization
  // unread grids should appear last in gridnames. Grid and its names in same order.
  Field3StructInput ioni({"IonizationRateCoeff"}, 
    {"gridTemperature_Ionization", "gridDensity_Ionization", 
    "gridChargeState_Ionization"}, {"n_Temperatures_Ionize", 
    "n_Densities_Ionize", "n_ChargeStates_Ionize"}, 2);
  
  readInputDataNcFileFS3(fName, ioni);

  //processFieldFileFS3(fName, ioni, debug);
  ionizeTempGridMin = ioni.getGridMin(0);
  ionizeDensGridMin = ioni.getGridMin(1);
  ionizeTempGridDT = ioni.getGridDelta(0);
  ionizeDensGridDn = ioni.getGridDelta(1);
  ionizeTempGridN = ioni.getNumGrids(0);
  ionizeDensGridN = ioni.getNumGrids(1);
  /*
  // require handles passed in or use temporary. Why ? TODO
  auto ion1 = o::Reals(ioni.data);
  auto ion2 = o::Reals(ioni.grid1);
  auto ion3 = o::Reals(ioni.grid2);
  ionizationRates = ion1;
  gridTempIonize = ion2;
  gridDensIonize = ion3;
  */
  ionizationRates = o::Reals(ioni.data);
  gridTempIonize = o::Reals(ioni.grid1);
  gridDensIonize  = o::Reals(ioni.grid2);
  std::cout<< "Loading Recombination data from " << fName << "\n" ;
  // not reading: , gridChargeState_Recombination
  Field3StructInput rec({"RecombinationRateCoeff"},
    {"gridTemperature_Recombination", "gridDensity_Recombination", 
    "gridChargeState_Recombination"}, {"n_Temperatures_Recombine",
    "n_Densities_Recombine", "n_ChargeStates_Recombine"}, 2);
  //processFieldFileFS3(fName, rec, debug);
  readInputDataNcFileFS3(fName, rec);

  recombTempGridMin = rec.getGridMin(0);
  recombDensGridMin = rec.getGridMin(1);
  recombTempGridDT = rec.getGridDelta(0);
  recombDensGridDn = rec.getGridDelta(1);
  recombTempGridN = rec.getNumGrids(0);
  recombDensGridN = rec.getNumGrids(1);
  recombinationRates = o::Reals(rec.data);
  gridTempRec = o::Reals(rec.grid1);
  gridDensRec = o::Reals(rec.grid2);
 printf("recombDensGridN %d \n", recombDensGridN);
}


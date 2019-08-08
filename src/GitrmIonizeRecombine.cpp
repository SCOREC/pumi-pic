#include "GitrmIonizeRecombine.hpp"

GitrmIonizeRecombine::GitrmIonizeRecombine(const std::string &fName,
  bool chargedTracking) {
  chargedPtclTracking = chargedTracking;
  initIonizeRecombRateData(fName);
}

// ADAS_Rates_W_structure for Tungsten W(z=74)
void GitrmIonizeRecombine::initIonizeRecombRateData(
  const std::string &fName, int debug) {
  std::cout<< "Loading ionize/recombine data from " << fName << "\n" ;
  // not reading: gridChargeState_Ionization
  // unread grids should appear last in gridnames. Grid and its names in same order.
  FieldStruct3 ioni("IonizRate", "IonizationRateCoeff", "", "", 
    "gridTemperature_Ionization", "gridDensity_Ionization", 
    "gridChargeState_Ionization", "n_Temperatures_Ionize", 
    "n_Densities_Ionize", "n_ChargeStates_Ionize", 1, 3, 2);

  processFieldFileFS3(fName, ioni, debug);
  ionizeTempGridMin = ioni.gr1Min;
  ionizeDensGridMin = ioni.gr2Min;
  ionizeTempGridDT = (ioni.gr1Max - ioni.gr1Min)/ioni.nGrid1;
  ionizeDensGridDn = (ioni.gr2Max - ioni.gr2Min)/ioni.nGrid2;
  ionizeTempGridN = ioni.nGrid1;
  ionizeDensGridN = ioni.nGrid2;
  ionizationRates = o::Reals(*ioni.data);
  gridTempIonize = o::Reals(*ioni.grid1);
  gridDensIonize = o::Reals(*ioni.grid2);

  // not reading: , gridChargeState_Recombination
  FieldStruct3 rec("Recomb", "RecombinationRateCoeff", "", "",
    "gridTemperature_Recombination", "gridDensity_Recombination", 
    "gridChargeState_Recombination", "n_Temperatures_Recombine",
    "n_Densities_Recombine", "n_ChargeStates_Recombine", 1, 3, 2);
  processFieldFileFS3(fName, rec, debug);
  recombTempGridMin = rec.gr1Min;
  recombDensGridMin = rec.gr2Min;
  recombTempGridDT = (rec.gr1Max - rec.gr1Min)/rec.nGrid1;
  recombDensGridDn = (rec.gr2Max - rec.gr2Min)/rec.nGrid2;
  recombTempGridN = rec.nGrid1;
  recombDensGridN = rec.nGrid2;
  recombinationRates = o::Reals(*rec.data);
  gridTempRec = o::Reals(*rec.grid1);
  gridDensRec = o::Reals(*rec.grid2);
}


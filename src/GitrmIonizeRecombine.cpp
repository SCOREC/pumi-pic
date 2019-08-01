#include "GitrmIonizeRecombine.hpp"

GitrmIonizeRecombine::GitrmIonizeRecombine(const std::string &fName) {
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

// The rest is outdated
struct IonizeRecombStruct {
    IonizeRecombStruct(std::string sn1, std::string sn2, std::string sn3,
      std::string nm, std::string sg1, std::string sg2, std::string sg3):
      nGrid1str(sn1), nGrid2str(sn2), nGrid3str(sn3), name(nm), 
      grid1str(sg1), grid2str(sg2), grid3str(sg3)
    {}
  std::string nGrid1str;
  std::string nGrid2str;
  std::string nGrid3str;
  std::string name;
  std::string grid1str;
  std::string grid2str;
  std::string grid3str;
  // 3rd grid not read
  int nData = 3;
  int nGridNames = 3;
  double gr1Min = 0;
  double gr1Max = 0;
  double gr2Min = 0;
  double gr2Max = 0;
  double gr3Min = 0;
  double gr3Max = 0;
  int nGrid1 = 0;
  int nGrid2 = 0;
  int nGrid3 = 0;
  o::HostWrite<o::Real> data;
  o::HostWrite<o::Real> grid1;
  o::HostWrite<o::Real> grid2;
};

void load3DGridScalarFieldFromFile(const std::string& fName, 
  IonizeRecombStruct& fs) {
  std::ifstream ifs(fName);
  if (!ifs.good())
    Omega_h_fail("Error opening Field file %s \n",fName.c_str() );
  // note: grid data are not stored in the main data array
  auto nComp = fs.nData;
  std::string gridNames[fs.nGridNames];
  bool foundComp[nComp], dataLine[nComp], eq=false, dataInit=false;
  int foundNums=0, ind[nComp];
  std::set<int> nans, nans1, nans2;
  for(int i = 0; i < nComp; ++i) {
    ind[i] = 0;
    foundComp[i] = dataLine[i] = false;
  }
  gridNames[0] = fs.nGrid1str;
  gridNames[1] = fs.nGrid2str;
  gridNames[2] = fs.nGrid3str;

  std::string line, s1, s2, s3;
  while(std::getline(ifs, line)) {
    bool semi = (line.find(';') != std::string::npos);
    std::replace (line.begin(), line.end(), ',' , ' ');
    std::replace (line.begin(), line.end(), ';' , ' ');
    std::stringstream ss(line);
    // first string or number of EACH LINE is got here
    ss >> s1;
    if(s1.find_first_not_of(' ') == std::string::npos) {
      s1 = "";
      if(!semi)
       continue;
    }    
    //grid names
    if(foundNums < fs.nGridNames) {
      for(int i=0; i<fs.nGridNames; ++i) {
        if(s1 == gridNames[i]) {
          ss >> s2 >> s3;
          OMEGA_H_CHECK(s2 == "=");
          int num = std::stoi(s3);
          OMEGA_H_CHECK(!std::isnan(num));
          if(i==0)
            fs.nGrid1 = num;
          else if(i==1)
            fs.nGrid2 = num;
          else if(i==2)
            fs.nGrid3 = num;        
          ++foundNums;
        }
      }
    }

    if(!dataInit && foundNums==3) {
      fs.grid1 = o::HostWrite<o::Real>(fs.nGrid1);
      fs.grid2 = o::HostWrite<o::Real>(fs.nGrid2);
      // no grid dat afor third grid
      fs.data = o::HostWrite<o::Real>(fs.nGrid1*fs.nGrid2*fs.nGrid3);
      dataInit = true;
    }

    if(dataInit) {
      if(!(dataLine[1] || dataLine[2]))
        parseFileFieldData(ss, s1, fs.name, semi, fs.data, ind[0], 
          dataLine[0], nans, eq, 0, 1, 0);

      if(!(dataLine[0] || dataLine[2]))
        parseFileFieldData(ss, s1, fs.grid1str, semi, fs.grid1, ind[1], 
          dataLine[1], nans1, eq, 0, 1, 0);

      if(!(dataLine[0] || dataLine[1]))
        parseFileFieldData(ss, s1, fs.grid2str, semi, fs.grid2, ind[2], 
          dataLine[2], nans2, eq, 0, 1, 0);

      for(int i=0; i<3; ++i)
        if(!foundComp[i] && dataLine[i]) {
          foundComp[i] = true;
        }
    }
    s1 = s2 = s3 = "";
  } //while
  OMEGA_H_CHECK(dataInit);
 // for(int i=0; i<nComp; ++i){
    // if ; on first line, dataLine is reset before reaching back
    //OMEGA_H_CHECK(foundComp[i]);
 // }
  if(ifs.is_open()) {
    ifs.close();
  }
  if(nans.size() > 0 || nans1.size() > 0 || nans2.size() > 0) 
    std::cout << "ERROR: NaN in ADAS file/grid\n";
}

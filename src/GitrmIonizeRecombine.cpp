#include "GitrmIonizeRecombine.hpp"

GitrmIonizeRecombine::GitrmIonizeRecombine(const std::string &fName) {
  initIonizeRecombRateData(fName);
}

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

// ADAS_Rates_W_structure for Tungsten W(z=74)
void GitrmIonizeRecombine::initIonizeRecombRateData(
  const std::string &fName) {
  std::cout<< "Loading ionize/recombine data from " << fName << "\n" ;
  // NOT read: gridChargeState_Ionization, gridChargeState_Recombination
  IonizeRecombStruct ist {"n_Temperatures_Ionize", 
    "n_Densities_Ionize", "n_ChargeStates_Ionize", 
    "IonizationRateCoeff", "gridTemperature_Ionization", 
    "gridDensity_Ionization", "gridChargeState_Ionization"};
  load3DGridScalarFieldFromFile(fName, ist);

  IONIZE_TEM_GRID_MIN = ist.gr1Min;
  IONIZE_DENS_GRID_MIN = ist.gr2Min;
  IONIZE_TEM_GRID_UNIT = (int)(ist.gr1Max - ist.gr1Min)/ist.nGrid1;
  IONIZE_DENSITY_GRID_UNIT = (int)(ist.gr2Max - ist.gr2Min)/ist.nGrid2;
  IONIZE_TEM_GRID_SIZE = ist.nGrid1;
  IONIZE_DENSITY_GRID_SIZE = ist.nGrid2;
  ionizationRates = o::Reals(ist.data);
  gridTempIonize = o::Reals(ist.grid1);
  gridDensIonize = o::Reals(ist.grid2);

  o::HostWrite<o::Real> rdata;
  o::HostWrite<o::Real> gridRTemp;
  o::HostWrite<o::Real> gridRDens;
  IonizeRecombStruct rst{"n_Temperatures_Recombine", 
    "n_Densities_Recombine", "n_ChargeStates_Recombine", 
    "RecombinationRateCoeff", "gridTemperature_Recombination", 
    "gridDensity_Recombination", "gridChargeState_Recombination"};
  load3DGridScalarFieldFromFile(fName, rst);
  RECOMBINE_TEM_GRID_MIN = rst.gr1Min;
  RECOMBINE_DENS_GRID_MIN = rst.gr2Min;
  RECOMBINE_TEM_GRID_UNIT = (int)(rst.gr1Max - rst.gr1Min)/rst.nGrid1;
  RECOMBINE_DENSITY_GRID_UNIT = (int)(rst.gr2Max - rst.gr2Min)/rst.nGrid2;
  RECOMBINE_TEM_GRID_SIZE = rst.nGrid1;
  RECOMBINE_DENSITY_GRID_SIZE = rst.nGrid2;
  recombinationRates = o::Reals(rst.data);
  gridTempRec = o::Reals(rst.grid1);
  gridDensRec = o::Reals(rst.grid2);
}

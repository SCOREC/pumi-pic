#include "GitrmIonizeRecombine.hpp"

GitrmIonizeRecombine::GitrmIonizeRecombine(const std::string &fName) {
    initIonizeRecombRateData(fName);
}

struct IonizeRecombStruct {
    IonizeRecombStruct(std::string n, std::string sn1, std::string sn2, 
      std::string sn3, std::string sg1,std::string sg2, std::string sg3):
      name(n), nGrid1str(sn1), nGrid2str(sn2), nGrid3str(sn3), grid1str(sg1),
      grid2str(sg2), grid3str(sg3) {}
  std::string name;
  std::string nGrid1str;
  std::string nGrid2str;
  std::string nGrid3str;
  std::string grid1str;
  std::string grid2str;
  std::string grid3str;
  o::Real gr1Min = 0;
  o::Real gr1Max = 0;
  o::Real gr2Min = 0;
  o::Real gr2Max = 0;
  o::Real gr3Min = 0;
  o::Real gr3Max = 0;
  o::LO nGrid1 = 0;
  o::LO nGrid2 = 0;
  o::LO nGrid3 = 0;
};

// ADAS_Rates_W_structure for Tungsten W(z=74)
void GitrmIonizeRecombine::initIonizeRecombRateData(const std::string &fName) {
  std::cout<< "Loading ionize/recomine data from " << file << "\n" ;
  o::HostWrite<o::Real> idata;
  IonizeRecombStruct ist {"IonizationRateCoeff", "n_Temperatures_Ionize", 
    "n_Densities_Ionize", "n_ChargeStates_Ionize", 
    "gridTemperature_Ionization", "gridDensity_Ionization", 
    "gridChargeState_Ionization"};
  load3DGridScalarFieldFromFile(fName, ist, idata);
  IONIZE_TEM_GRID_MIN = ist.gr1Min;
  IONIZE_DENS_GRID_MIN = ist.gr2Min;
  IONIZE_TEM_GRID_UNIT = (o::LO)(ist.gr1Max - ist.gr1Min)/ist.nGrid1;
  IONIZE_DENSITY_GRID_UNIT = (o::LO)(ist.gr2Max - ist.gr2Min)/ist.nGrid2;
  IONIZE_TEM_GRID_SIZE = ist.nGrid1;
  IONIZE_DENSITY_GRID_SIZE = ist.nGrid2;
  ionizationRates = o::Reals(idata);

  o::HostWrite<o::Real> rdata;
  IonizeRecombStruct rst{"RecombinationRateCoeff", "n_Temperatures_Recombine", 
    "n_Densities_Recombine", "n_ChargeStates_Recombine", 
    "gridTemperature_Recombination", "gridDensity_Recombination", 
    "gridChargeState_Recombination"};
  load3DGridScalarFieldFromFile(fName, rst, rdata);
  RECOMBINE_TEM_GRID_MIN = rst.gr1Min;
  RECOMBINE_DENS_GRID_MIN = rst.gr2Min;
  RECOMBINE_TEM_GRID_UNIT = (o::LO)(rst.gr1Max - rst.gr1Min)/rst.nGrid1;
  RECOMBINE_DENSITY_GRID_UNIT = (o::LO)(rst.gr2Max - rst.gr2Min)/rst.nGrid2;
  RECOMBINE_TEM_GRID_SIZE = rst.nGrid1;
  RECOMBINE_DENSITY_GRID_SIZE = rst.nGrid2;
  recombinationRates = o::Reals(rdata);
}


void load3DGridScalarFieldFromFile(const std::string& file, 
  FieldStruct3DGrid& fs, o::HostWrite<o::Real>& data) {
  std::ifstream ifs(fName);
  if (!ifs.good())
    Omega_h_fail("Error opening Field file %s \n",fName );

  std::string fieldNames[nComp];
  int ind[nComp];
  bool expectEqual = false;
  bool foundComp[nComp], dataLine[nComp]; //6=x,y,z,vx,vy,vz
  std::string fieldNames[nComp];
  int foundNums=0;
  bool expectEqual = false, dataInit = false;
  int ind[nComp];
  std::set<int> nans;
  for(int i = 0; i < nComp; ++i) {
    ind[i] = 0;
    foundComp[i] = dataLine[i] = false;
  }
  fieldNames[0] = fs.name;
  fieldNames[1] = fs.nGrid1str;
  fieldNames[2] = fs.nGrid2str;
  fieldNames[3] = fs.nGrid3str;
  fieldNames[4] = fs.grid1str;
  fieldNames[5] = fs.grid2str;
  fieldNames[6] = fs.grid3str;

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
    if(foundNums < 3) {
      for(int i=1; i<4; ++i) {
        if(s1 == fieldNames[i]) {
          ss >> s2 >> s3;
          OMEGA_H_CHECK(s2 == "=");
          int num = std::stoi(s3);
          if(i==1)
            fs.nGrid1 = num;
          else if(i==2)
            fs.nGrid2 = num;
          else if(i==3)
            fs.nGrid3 = num;        
          ++foundNums;
        }
      }
    }

    if(!dataInit && foundNums==3) {
      data = o::HostWrite<o::Real>(fs.nGrid1*fs.nGrid2*fs.nGrid3);
      dataInit = true;
    }
    int compBeg = 0, compEnd = nComp;
    if(dataInit) {
      for(int iComp = compBeg; iComp<compEnd; ++iComp) {
        parseFileFieldData(ss, s1, fieldNames[iComp], semi, data, ind[iComp], 
          dataLine[iComp], nans, expectEqual, iComp, nComp, numPtcls);
        
        if(!foundComp[iComp] && dataLine[iComp])
          foundComp[iComp] = true;
      }
    }
    s1 = s2 = s3 = "";
  } //while
  OMEGA_H_CHECK(dataInit && foundNums==3);
  for(int i=0; i<6; ++i)
    OMEGA_H_CHECK(foundComp[i]==true);
  if(ifs.is_open()) {
    ifs.close();
  }
  if(nans.size() > 0) 
    std::cout << "ERROR: NaN in ADAS file" << nans.size() << "\n";
}


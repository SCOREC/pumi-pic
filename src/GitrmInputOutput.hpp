#ifndef GITRM_INPUT_OUTPUT_HPP
#define GITRM_INPUT_OUTPUT_HPP
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <initializer_list>
#include "Omega_h_for.hpp"

namespace o = Omega_h;

struct Field3StructInput;
struct OutputNcFileFieldStruct;

int verifyNetcdfFile(const std::string& ncFileName, int nc_err=1);
int readParticleSourceNcFile(std::string ncFileName,
  o::HostWrite<o::Real>& data, int& maxNPtcls, int& numPtclsRead, 
  bool replaceNaN=false);

int readInputDataNcFileFS3(const std::string& ncFileName,
  Field3StructInput& fs, bool debug=false);
// numPtclsRead to know data shape, if input file organized differently
int readInputDataNcFileFS3(const std::string& ncFileName,
  Field3StructInput& fs, int& maxNPtcls, int& numPtclsRead, 
  std::string nPstr="nP", bool debug=false);

void writeOutputNcFile( o::Write<o::Real>& ptclsHistoryData, int numPtcls, 
  int dof, OutputNcFileFieldStruct& st, std::string outNcFileName);
void writeOutBdryFaceCoordsNcFile(const std::string& outFileName, 
    o::Write<o::Real>& xd, o::Write<o::Real>& yd, o::Write<o::Real>& zd, const int nf); 
void writeOutputCsrFile(const std::string& outFileName, 
    const std::vector<std::string>& vars, const std::vector<std::string>& datNames, 
    o::LOs& ptrs_d, o::LOs& data_d, int* valExtra=nullptr);

int readCsrFile(const std::string& ncFileName,  
  const std::vector<std::string>& vars,
  const std::vector<std::string>& datNames, o::LOs& ptrs, o::LOs& data);

struct Field3StructInput {
  static constexpr int MAX_SIZE = 3; // add grid data before resizing
  Field3StructInput(std::initializer_list<std::string> compNames_in,
    std::initializer_list<std::string> gridNames_in,
    std::initializer_list<std::string> nGridNames_in, int nGridRead_in=-1,
    std::initializer_list<std::string> nVarNames_in = {}) {
    compNames = compNames_in;
    gridNames = gridNames_in;
    nGridNames = nGridNames_in;
    nGridRead = (nGridRead_in < 0) ? gridNames.size(): nGridRead_in;
    nVarNames = nVarNames_in;
    nComp = compNames.size();
    assert(nComp <= MAX_SIZE);
    assert(gridNames.size() <= MAX_SIZE);
    assert(nGridNames.size() <= MAX_SIZE);
  }
  std::vector<std::string> compNames;
  std::vector<std::string> gridNames;
  std::vector<std::string> nGridNames;
  std::vector<std::string> nVarNames;

  int nComp = 0;
  //Only first nGridRead grids are read out of gridNames_in
  int nGridRead = 0;
  std::vector<int> nGridVec;
  std::vector<int> nVarVec;
  double getGridDelta(int ind);
  int getNumGrids(int ind);
  double getGridMin(int ind);
  double getGridMax(int ind);
  int getIntValueOf(std::string name);
  // All grids are doubles; otherwise use template
  // grid data stored per grid, to use in interpolation routine
  o::HostWrite<o::Real> data;
  o::HostWrite<o::Real> grid1;
  o::HostWrite<o::Real> grid2;
  o::HostWrite<o::Real> grid3;
};

struct OutputNcFileFieldStruct {
  //require nDims' names first in init-list names
  OutputNcFileFieldStruct(std::initializer_list<std::string> numStringsIn,
    std::initializer_list<std::string> namesIn, 
    std::initializer_list<int> nDimsIn) {
    numStrings = numStringsIn;
    fieldNames = namesIn;
    nDims = nDimsIn;
  }
  std::vector<std::string> numStrings;
  std::vector<std::string> fieldNames;
  std::vector<int> nDims;
};

#endif

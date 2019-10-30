#include <fstream>
#include <cmath>
#include <utility>
#include <sstream>
#include "Omega_h_fail.hpp"
#include "Omega_h_array.hpp"
#include "Omega_h_array_ops.hpp"
#include <netcdf>
#include <cstddef>
#include "GitrmInputOutput.hpp"

double Field3StructInput::getGridDelta(int ind) {
  assert(ind < MAX_SIZE && ind >=0);
  assert(nGridVec.size() > ind && nGridVec[ind] > 0);
  return (getGridMax(ind) - getGridMin(ind))/(o::Real)nGridVec[ind];
}

int Field3StructInput::getNumGrids(int ind) {
  assert(ind < MAX_SIZE && ind >=0);
  assert(nGridVec.size() > ind);
  return nGridVec[ind];
}

double Field3StructInput::getGridMin(int ind) {
  assert(ind >=0 && ind < MAX_SIZE);
  assert(MAX_SIZE ==3);
  return (ind>0) ? ((ind>1)? (grid3[0]): grid2[0]) : grid1[0];
}

double Field3StructInput::getGridMax(int ind) {
  assert(ind >=0 && ind < 3);
  assert(MAX_SIZE == 3);
  double max;
  if(ind==0)
    max = grid1[grid1.size()-1];
  else if(ind==1)
    max = grid2[grid2.size()-1];
  else if(ind==2)
    max = grid3[grid3.size()-1];
  return max;
}

int verifyNetcdfFile(std::string& ncFileName, int nc_err) {
  try {
    netCDF::NcFile ncFile(ncFileName, netCDF::NcFile::read);
  } catch (netCDF::exceptions::NcException &e) {
    std::cout << e.what() << std::endl;
    return nc_err;
  }
  return 0;
} 

int readParticleSourceNcFile(std::string ncFileName, 
    o::HostWrite<o::Real>& data, int& maxNPtcls, int& numPtclsRead,
    bool replaceNaN) {
  constexpr int dof = 6;
  try {
    netCDF::NcFile ncf(ncFileName, netCDF::NcFile::read);
    netCDF::NcDim ncf_np(ncf.getDim("nP"));
    int np = ncf_np.getSize();
    if(np < maxNPtcls)
      maxNPtcls = np;
    //if(maxNPtcls >0 && maxNPtcls < np)
    //  np = maxNPtcls;
    numPtclsRead = np;
    std::cout << "nPtcls in source file " << np << "\n";
    data = o::HostWrite<o::Real>(np*dof);
    netCDF::NcVar ncx(ncf.getVar("x"));
    netCDF::NcVar ncy(ncf.getVar("y"));
    netCDF::NcVar ncz(ncf.getVar("z"));
    netCDF::NcVar ncvx(ncf.getVar("vx"));
    netCDF::NcVar ncvy(ncf.getVar("vy"));
    netCDF::NcVar ncvz(ncf.getVar("vz"));
    ncx.getVar(&data[0]);
    ncy.getVar(&data[1*np]);
    ncz.getVar(&data[2*np]);
    ncvx.getVar(&data[3*np]);
    ncvy.getVar(&data[4*np]);
    ncvz.getVar(&data[5*np]);
  } catch (netCDF::exceptions::NcException &e) {
    std::cout << e.what() << std::endl;
    return 1;
  }
  if(replaceNaN) {
    int nans = 0;
    for(int i=0; i<data.size(); ++i)
      if(std::isnan(data[i])) {
        data[i] = 0;
        ++nans;
      }
    if(nans)
      printf("\n*******WARNING replaced %d NaNs in ptclSrc *******\n\n", nans);
  }
  return 0;
}

//Reads from 0 to 3 grids having gridNames; .
int readInputDataNcFileFS3(const std::string& ncFileName,
  Field3StructInput& fs) {
  int maxNPtcls = 0;
  int numPtclsRead = 0;
  return readInputDataNcFileFS3(ncFileName, fs, maxNPtcls, numPtclsRead);
}

// maxNPtcls updated if > that in file. TODO read only maxNPtcls
int readInputDataNcFileFS3(const std::string& ncFileName,
  Field3StructInput& fs, int& maxNPtcls, int& numPtclsRead, std::string nPstr) {
  int ncSizePerComp = 1;
  try {
    netCDF::NcFile ncf(ncFileName, netCDF::NcFile::read);
    for(int i=0; i< fs.nGridNames.size(); ++i) {
      netCDF::NcDim ncGridName(ncf.getDim(fs.nGridNames[i]));
      auto size = ncGridName.getSize();
      fs.nGridVec.push_back(size);
      if(fs.nGridNames[i] == nPstr) {
        if(size < maxNPtcls)
          maxNPtcls = size;
        //else if(size > maxNPtcls)
        //  numPtclsRead = maxNPtcls; //TODO enable it below
        numPtclsRead = size;
      }

      std::cout << ncFileName << " : " << fs.nGridNames[i] << " : " << fs.nGridVec[i] << "\n";
      ncSizePerComp *= fs.nGridVec[i];
    }    
    std::cout << " ncSizePerComp: " << ncSizePerComp << " nComp " << fs.nComp << "\n";

    fs.data = o::HostWrite<o::Real>(ncSizePerComp*fs.nComp);
    for(int i=0; i<fs.nGridRead && i<fs.gridNames.size(); ++i) {
      netCDF::NcVar ncvar(ncf.getVar(fs.gridNames[i].c_str()));
      if(i==0) {
        fs.grid1 = o::HostWrite<o::Real>(fs.nGridVec[0]);
        ncvar.getVar(&(fs.grid1[0]));
      }
      if(i==1) {
        fs.grid2 = o::HostWrite<o::Real>(fs.nGridVec[1]);
        ncvar.getVar(&(fs.grid2[0]));
      }
      if(i==2) {
        fs.grid1 = o::HostWrite<o::Real>(fs.nGridVec[2]);
        ncvar.getVar(&(fs.grid3[0]));
      }
      std::cout << i << " "  << fs.gridNames[i] <<  " " << fs.grid1[1] << " \n";
    }
    // TODO use maxNPtcls and numPtclsRead
    for(int i=0; i<fs.nComp; ++i) {
      netCDF::NcVar ncvar(ncf.getVar(fs.compNames[i].c_str()));
      std::cout << "getVar " << fs.compNames[i] << "\n";
      ncvar.getVar(&(fs.data[i*ncSizePerComp]));
    }
    std::cout << " data[0] " <<  (fs.data)[0] << "\n";
    
    std::cout << " Done reading " << ncFileName << "\n";
  } catch (netCDF::exceptions::NcException &e) {
    std::cout << e.what() << std::endl;
    return 1;
  }

  // replace NaNs
  int nans = 0;
  for(int i=0; i<fs.data.size(); ++i)
    if(std::isnan(fs.data[i])) {
      fs.data[i] = 0;
      ++nans;
    }
  if(nans)
    printf("\n*******WARNING replaced %d NaNs in data ******\n\n", nans);

  return 0;
}

void writeOutputNcFile( o::Write<o::Real>& ptclHistoryData, int numPtcls,
  int dof, OutputNcFileFieldStruct& st, std::string outNcFileName) {
  //if ext not nc, 
  //outNcFileName = outNcFileName + std::to_string(i) + ".nc";
  assert(dof == st.fieldNames.size());
  assert(numPtcls == st.nDims[0]);
  try {
    netCDF::NcFile ncFile(outNcFileName, netCDF::NcFile::replace);
    std::vector<netCDF::NcDim> ncDims;
    for(int i=0; i< st.nDims.size(); ++i) {
      netCDF::NcDim dim = ncFile.addDim(st.numStrings[i], st.nDims[i]);
      ncDims.push_back(dim);
    }
    std::vector<netCDF::NcVar> ncVars;
    for(int i=0; i< st.fieldNames.size(); ++i) {
      netCDF::NcVar var = ncFile.addVar(st.fieldNames[i], 
        netCDF::NcDouble(), ncDims);
      ncVars.push_back(var);
    }
    //stored in timestep order : numPtcls*dof * iHistStep + id*dof
    o::HostWrite<o::Real>ptclsData(ptclHistoryData);
    int histn = (int)ptclsData.size()/(dof*numPtcls);

    for(int i=0; i< dof; ++i) {
      o::HostWrite<o::Real> dat(histn*numPtcls);
      for(int j=0; j<numPtcls; ++j) {
        for(int k=0; k<histn; ++k) {
          dat[j*histn+k] = ptclsData[j*dof + i + k*numPtcls*dof];
        }
      }
      ncVars[i].putVar(&(dat[0]));

      //const std::vector<size_t> start{(size_t)i};
      //const std::vector<size_t> count{(size_t)numPtclxHistn};
      //const std::vector<ptrdiff_t> stridep{dof};
      // http://unidata.github.io/netcdf-cxx4/ncVar_8cpp_source.html#l00788 line-1142
      // https://github.com/Unidata/netcdf-c/blob/master/ncdump/ref_ctest.c
      //ncVars[i].putVar(start, count, stridep, &(ptclsData[0]));
      
    }
  } catch (netCDF::exceptions::NcException& e) {
    std::cout << e.what() << "\n";
  }
}

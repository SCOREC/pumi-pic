#include "GitrmSurfaceModel.hpp"


GitrmSurfaceModel::GitrmSurfaceModel(GitrmMesh& gm, std::string ncFile):
  gm(gm), mesh(gm.mesh),  ncFile(ncFile){
//  initSurfaceModelData(ncFile);
}

void GitrmSurfaceModel::initSurfaceModelData(std::string ncFile, bool debug) {
  getConfigData(ncFile);
  numDetectorSurfaceFaces = gm.numDetectorSurfaceFaces;
  assert(numDetectorSurfaceFaces > 0);

  nDistEsurfaceModel =
     nEnSputtRefDistIn * nAngSputtRefDistIn * nEnSputtRefDistOut;
  nDistEsurfaceModelRef =
     nEnSputtRefDistIn * nAngSputtRefDistIn * nEnSputtRefDistOutRef;
  nDistAsurfaceModel =
     nEnSputtRefDistIn * nAngSputtRefDistIn * nAngSputtRefDistOut;

  prepareSurfaceModelData();

  if(fluxEA > 0) {
    dEdist = (enDist - en0Dist)/nEnDist;
    dAdist = (angDist - ang0Dist)/nAngDist;
  }
  if(debug)
    printf("nEdist %d nAdist %d \n", nEnDist, nAngDist);

  auto nDist = numDetectorSurfaceFaces * nEnDist * nAngDist;
  energyDistribution = o::Write<o::Real>(nDist); //9k/detFace
  sputtDistribution = o::Write<o::Real>(nDist); //9k/detFace
  reflDistribution = o::Write<o::Real>(nDist); //9k/detFace
  mesh.add_tag<o::Real>(o::FACE, "SumParticlesStrike", 1);
  mesh.add_tag<o::Int>(o::FACE, "SumWeightStrike", 1);
  mesh.add_tag<o::Real>(o::FACE, "GrossDeposition", 1);
  mesh.add_tag<o::Real>(o::FACE, "GrossErosion", 1);
  mesh.add_tag<o::Real>(o::FACE, "AveSputtYld", 1); 
  mesh.add_tag<o::Int>(o::FACE, "SputtYldCount", 1);
  //mesh.add_tag<o::Int>(o::FACE, "IsSurface", 1);
  //TODO replace currently used tag "piscesBeadCylinder_inds" 
}


template<typename T>
void GitrmSurfaceModel::make2dCDF(const int nX, const int nY, const int nZ, 
   const o::HostWrite<T>& distribution, o::HostWrite<T>& cdf) {
  assert(distribution.size() == nX*nY*nZ);
  assert(cdf.size() == nX*nY*nZ);
  int index = 0;
  for(int i=0;i<nX;i++) {
    for(int j=0;j<nY;j++) {
      for(int k=0;k<nZ;k++) {
        index = i*nY*nZ + j*nZ + k;
        if(k==0)
          cdf[index] = distribution[index];
        else
          cdf[index] = cdf[index-1] + distribution[index];
      }  
    }  
  }
  for(int i=0;i<nX;i++) {
    for(int j=0;j<nY;j++) {
      if(cdf[i*nY*nZ + (j+1)*nZ - 1] > 0) {
        for(int k=0;k<nZ;k++) {  
          index = i*nY*nZ + j*nZ + k;
          cdf[index] = cdf[index] / cdf[index-k+nZ-1];
        }
      }
    }
  }
}

//TODO make DEVICE
template<typename T>
T GitrmSurfaceModel::interp1dUnstructured(const T samplePoint, const int nx, 
   const T max_x, const T* data, int& lowInd) {
  int done = 0;
  int low_index = 0;
  T value = 0;
  for(int i=0;i<nx;i++) {
    if(done == 0) {
      if(samplePoint < data[i]) {
        done = 1;
        low_index = i-1;
      }   
    }
  }
  value = ((data[low_index+1] - samplePoint)*low_index*max_x/nx
        + (samplePoint - data[low_index])*(low_index+1)*max_x/nx)/
          (data[low_index+1]- data[low_index]);
  lowInd = low_index;
  if(low_index < 0) {
    lowInd = 0;
    if(samplePoint > 0) {
      value = samplePoint;
    } else {
      value = 0;
    }
  }
  if(low_index >= nx) {
    lowInd = nx-1;
    value = max_x;
  }
  return value;
}

//TODO convert to device function
template<typename T>
void GitrmSurfaceModel::regrid2dCDF(const int nX, const int nY, const int nZ, 
   const o::HostWrite<T>& xGrid, const int nNew, const T maxNew, 
   const o::HostWrite<T>& cdf, o::HostWrite<T>& cdf_regrid) {
  int lowInd = 0;
  int index = 0;
  float spline = 0.0;
  for(int i=0;i<nX;i++) {
    for(int j=0;j<nY;j++) {
      for(int k=0;k<nZ;k++) {
        index = i*nY*nZ + j*nZ + k;
        spline = interp1dUnstructured(xGrid[k], nNew, maxNew, &(cdf.data()[index-k]), lowInd);
        if(isnan(spline) || isinf(spline)) 
          spline = 0.0;
        cdf_regrid[index] = spline;  
      }  
    }
  }
}


void GitrmSurfaceModel::prepareSurfaceModelData() {
  o::Write<o::Real> enLogSputtRefCoef_w(nEnSputtRefCoeff);
  o::parallel_for(nEnSputtRefCoeff, OMEGA_H_LAMBDA(int ii) {
    for(int i = 0; i < nEnSputtRefCoeff; i++) {
      enLogSputtRefCoef_w[i] = log10(enSputtRefCoeff[i]);
    }
  });
  o::HostWrite<o::Real>enLogSputtRefCoef(enLogSputtRefCoef_w);

  o::Write<o::Real> enLogSputtRefDistIn_w(nEnSputtRefDistIn);
  o::parallel_for(nEnSputtRefCoeff, OMEGA_H_LAMBDA(int ii) {
    for(int i = 0; i < nEnSputtRefDistIn; i++) {
      enLogSputtRefDistIn_w[i] = log10(enSputtRefDistIn[i]);
    }
  });
  o::HostWrite<o::Real>enLogSputtRefDistIn(enLogSputtRefDistIn_w);
  
  o::Write<o::Real> energyDistGrid01_w(nEnSputtRefDistOut);
  o::parallel_for(nEnSputtRefCoeff, OMEGA_H_LAMBDA(int ii) {
    for(int i = 0; i < nEnSputtRefDistOut; i++) {
      energyDistGrid01_w[i] = i * 1.0 / nEnSputtRefDistOut;
    }
  });
  o::HostWrite<o::Real>energyDistGrid01(energyDistGrid01_w);
  
  o::Write<o::Real> energyDistGrid01Ref_w(nEnSputtRefDistOutRef);
  o::parallel_for(nEnSputtRefCoeff, OMEGA_H_LAMBDA(int ii) {
    for(int i = 0; i < nEnSputtRefDistOutRef; i++) {
      energyDistGrid01Ref_w[i] = i * 1.0 / nEnSputtRefDistOutRef;
    }
  });
  o::HostWrite<o::Real>energyDistGrid01Ref(energyDistGrid01Ref_w);
  
  o::Write<o::Real> angleDistGrid01_w(nAngSputtRefDistOut);
  o::parallel_for(nEnSputtRefCoeff, OMEGA_H_LAMBDA(int ii) {
    for(int i = 0; i < nAngSputtRefDistOut; i++) {
      angleDistGrid01_w[i] = i * 1.0 / nAngSputtRefDistOut;
    }
  });
  o::HostWrite<o::Real>angleDistGrid01(angleDistGrid01_w);
 
  o::HostWrite<o::Real>enDist_CDF_Y(enDist_Y.size());
  o::HostWrite<o::Real>enDist_Y_h(o::deep_copy(enDist_Y));
  make2dCDF<o::Real>(nEnSputtRefDistIn, nAngSputtRefDistIn, nEnSputtRefDistOut,
   enDist_Y_h, enDist_CDF_Y);

  o::HostWrite<o::Real>angPhiDist_Y_h(o::deep_copy(angPhiDist_Y));
  o::HostWrite<o::Real>angPhiDist_CDF_Y(angPhiDist_Y.size());
  make2dCDF<o::Real>(nEnSputtRefDistIn, nAngSputtRefDistIn, nAngSputtRefDistOut,
   angPhiDist_Y_h, angPhiDist_CDF_Y);

  o::HostWrite<o::Real>angThetaDist_Y_h(o::deep_copy(angThetaDist_Y));
  o::HostWrite<o::Real>angThetaDist_CDF_Y(angThetaDist_Y.size());
  make2dCDF<o::Real>(nEnSputtRefDistIn, nAngSputtRefDistIn, nAngSputtRefDistOut,
   angThetaDist_Y_h, angThetaDist_CDF_Y);

  o::HostWrite<o::Real>enDist_R_h(o::deep_copy(enDist_R));
  o::HostWrite<o::Real>enDist_CDF_R(enDist_R.size());
  make2dCDF<o::Real>(nEnSputtRefDistIn, nAngSputtRefDistIn, nEnSputtRefDistOutRef,
   enDist_R_h, enDist_CDF_R);

  o::HostWrite<o::Real>angPhiDist_R_h(o::deep_copy(angPhiDist_R));
  o::HostWrite<o::Real>angPhiDist_CDF_R(angPhiDist_R.size());
  make2dCDF<o::Real>(nEnSputtRefDistIn, nAngSputtRefDistIn, nAngSputtRefDistOut,
   angPhiDist_R_h, angPhiDist_CDF_R);

  o::HostWrite<o::Real>angThetaDist_R_h(o::deep_copy(angThetaDist_R));
  o::HostWrite<o::Real>angThetaDist_CDF_R(angThetaDist_R.size());
  make2dCDF<o::Real>(nEnSputtRefDistIn, nAngSputtRefDistIn, nAngSputtRefDistOut,
   angThetaDist_R_h, angThetaDist_CDF_R);
  
  o::HostRead<o::Real> angPhiSputtRefDistOut_h(o::deep_copy(angPhiSputtRefDistOut));
  o::HostWrite<o::Real>angPhiDist_CDF_Y_regrid_h(angPhiDist_CDF_Y.size());
  regrid2dCDF<o::Real>(nEnSputtRefDistIn, nAngSputtRefDistIn, nAngSputtRefDistOut,
   angleDistGrid01, nAngSputtRefDistOut,
   angPhiSputtRefDistOut_h[nAngSputtRefDistOut - 1],
   angPhiDist_CDF_Y, angPhiDist_CDF_Y_regrid_h);
  angPhiDist_CDF_Y_regrid = o::Reals(angPhiDist_CDF_Y_regrid_h.write());

  o::HostRead<o::Real> angThetaSputtRefDistOut_h(o::deep_copy(angThetaSputtRefDistOut));
  o::HostWrite<o::Real>angThetaDist_CDF_Y_regrid_h(angThetaDist_CDF_Y.size());
  regrid2dCDF<o::Real>(nEnSputtRefDistIn, nAngSputtRefDistIn, nAngSputtRefDistOut,
   angleDistGrid01, nAngSputtRefDistOut, angThetaSputtRefDistOut_h[nAngSputtRefDistOut - 1],
   angThetaDist_CDF_Y, angThetaDist_CDF_Y_regrid_h);
  angThetaDist_CDF_Y_regrid = o::Reals(angThetaDist_CDF_Y_regrid_h.write());

  o::HostRead<o::Real> enSputtRefDistOut_h(o::deep_copy(enSputtRefDistOut));
  o::HostWrite<o::Real>enDist_CDF_Y_regrid_h(enDist_CDF_Y.size());
  regrid2dCDF<o::Real>(nEnSputtRefDistIn, nAngSputtRefDistIn, nEnSputtRefDistOut,
   energyDistGrid01, nEnSputtRefDistOut, enSputtRefDistOut_h[nEnSputtRefDistOut - 1],
   enDist_CDF_Y, enDist_CDF_Y_regrid_h);
  enDist_CDF_Y_regrid = o::Reals(enDist_CDF_Y_regrid_h);

  o::HostWrite<o::Real>angPhiDist_CDF_R_regrid_h(angPhiDist_CDF_R.size());
  regrid2dCDF<o::Real>(nEnSputtRefDistIn, nAngSputtRefDistIn, nAngSputtRefDistOut,
   angleDistGrid01, nAngSputtRefDistOut,
   angPhiSputtRefDistOut_h[nAngSputtRefDistOut - 1],
   angPhiDist_CDF_R, angPhiDist_CDF_R_regrid_h);
  angPhiDist_CDF_R_regrid = o::Reals(angPhiDist_CDF_R_regrid_h);

  o::HostWrite<o::Real>angThetaDist_CDF_R_regrid_h(angThetaDist_CDF_R.size());
  regrid2dCDF<o::Real>(nEnSputtRefDistIn, nAngSputtRefDistIn, nAngSputtRefDistOut,
   angleDistGrid01, nAngSputtRefDistOut,
   angThetaSputtRefDistOut_h[nAngSputtRefDistOut - 1],
   angThetaDist_CDF_R, angThetaDist_CDF_R_regrid_h);
  angThetaDist_CDF_R_regrid = o::Reals(angThetaDist_CDF_R_regrid_h);

  o::HostRead<o::Real>enSputtRefDistOutRef_h(o::deep_copy(enSputtRefDistOutRef));
  o::HostWrite<o::Real>enDist_CDF_R_regrid_h(enDist_CDF_R.size());
  regrid2dCDF<o::Real>(nEnSputtRefDistIn, nAngSputtRefDistIn, nEnSputtRefDistOutRef,
   energyDistGrid01Ref, nEnSputtRefDistOutRef,
   enSputtRefDistOutRef_h[nEnSputtRefDistOutRef - 1],
   enDist_CDF_R, enDist_CDF_R_regrid_h);
  enDist_CDF_R_regrid = o::Reals(enDist_CDF_R_regrid_h);
}

void GitrmSurfaceModel::getConfigData(std::string ncFileName) {
  //TODO get from config file
  //collect data for analysis/plot
  nEnDist = 100;
  en0Dist = 0.0;
  enDist = 1000.0;
  nAngDist = 90; 
  ang0Dist = 0.0;
  angDist = 90.0; 

  //from NC file
  fileString = ncFileName;//"ftridynSelf.nc";
  nEnSputtRefCoeffStr = "nE";
  nAngSputtRefCoeffStr = "nA";
  nEnSputtRefDistInStr = "nE";
  nAngSputtRefDistInStr = "nA";
  nEnSputtRefDistOutStr = "nEdistBins";
  nEnSputtRefDistOutRefStr = "nEdistBinsRef";
  nAngSputtRefDistOutStr = "nAdistBins";
  enSputtRefCoeffStr = "E";
  angSputtRefCoeffStr = "A";
  enSputtRefDistInStr = "E";
  angSputtRefDistInStr = "A";
  enSputtRefDistOutStr = "eDistEgrid";
  enSputtRefDistOutRefStr = "eDistEgridRef";
  angPhiSputtRefDistOutStr = "phiGrid";
  angThetaSputtRefDistOutStr = "thetaGrid";
  sputtYldStr = "spyld";
  reflYldStr = "rfyld";
  enDistYStr = "energyDist";
  angPhiDistYStr = "cosXDist";
  angThetaDistYStr = "cosYDist";
  enDistRStr = "energyDistRef";
  angPhiDistRStr = "cosXDistRef";
  angThetaDistRStr = "cosYDistRef";
/*
ftrydin.nc file header
dimensions:
        nE = 50 ;
        nA = 40 ;
        nEdistBins = 100 ;
        nEdistBinsRef = 500 ;
        nAdistBins = 50 ;
variables:
        double spyld(nE, nA) ;
        double rfyld(nE, nA) ;
        double E(nE) ;
        double A(nA) ;
        double cosXDist(nE, nA, nAdistBins) ;
        double cosYDist(nE, nA, nAdistBins) ;
        double cosZDist(nE, nA, nAdistBins) ;
        double cosXDistRef(nE, nA, nAdistBins) ;
        double cosYDistRef(nE, nA, nAdistBins) ;
        double cosZDistRef(nE, nA, nAdistBins) ;
        double energyDist(nE, nA, nEdistBins) ;
        double energyDistRef(nE, nA, nEdistBinsRef) ;
        double eDistEgrid(nEdistBins) ;
        double eDistEgridRef(nEdistBinsRef) ;
        double phiGrid(nAdistBins) ;
        double thetaGrid(nAdistBins) ;
*/
  //see ftrydin.nc file header
  std::vector<std::string> dataNames{sputtYldStr, reflYldStr, enSputtRefCoeffStr, angSputtRefCoeffStr, 
   enSputtRefDistInStr, angSputtRefDistInStr, angPhiDistYStr, 
   angThetaDistYStr, angPhiDistRStr, angThetaDistRStr, enDistYStr, enDistRStr, 
   enSputtRefDistOutStr, enSputtRefDistOutRefStr, angPhiSputtRefDistOutStr, 
   angThetaSputtRefDistOutStr};

  std::vector<o::Reals> data({sputtYld, reflYld, enSputtRefCoeff, angSputtRefCoeff,
   enSputtRefDistIn, angSputtRefDistIn, angPhiDist_Y, angThetaDist_Y, angPhiDist_R, 
     angThetaDist_R, enDist_Y, enDist_R, enSputtRefDistOut, enSputtRefDistOutRef, angPhiSputtRefDistOut, 
   angThetaSputtRefDistOut});
  
  std::vector<std::string> shapeNames{enSputtRefCoeffStr, nAngSputtRefCoeffStr,
   nEnSputtRefDistInStr, nAngSputtRefDistInStr, nEnSputtRefDistOutStr, 
   nEnSputtRefDistOutRefStr, nAngSputtRefDistOutStr};
  
  //data corresponding to shapeNames
  std::vector<int> shapeData{nEnSputtRefCoeff, nAngSputtRefCoeff, nEnSputtRefDistIn, 
    nAngSputtRefDistIn, nEnSputtRefDistOut, nEnSputtRefDistOutRef, nAngSputtRefDistOut};

  //indices of shapeData, corresponding to entries in data and dataNames
  std::vector<std::vector<int>> shapeVec{{0,1},{0,1},{0},{1},{2},{3},{0,1,6},{0,1,6},
    {0,1,6},{0,1,6},{0,1,4},{0,1,5},{4},{5},{6},{6}};
  getSurfaceModelDataFromFile(fileString, dataNames, shapeNames, shapeVec, shapeData, data);

}

void GitrmSurfaceModel::getSurfaceModelDataFromFile(const std::string fileName,
   const std::vector<std::string>& dataNames, const std::vector<std::string>& shapeNames,
   const std::vector<std::vector<int>>& shapeVec, std::vector<int>& shapeData,
   std::vector<o::Reals>& data){
  for(int i=0; i<dataNames.size(); ++i) {
    auto datName = dataNames[i];
    auto shapeInds = shapeVec[i];
    std::vector<std::string> shapes; 
    for(auto j: shapeInds) {
      shapes.push_back(shapeNames[j]);
    }
    Field3StructInput fs({datName},{},shapes);
    readInputDataNcFileFS3(fileName, fs);
    data[i] = o::Reals(fs.data.write());
    for(auto j: shapeInds) {
      shapeData[i] = fs.getIntValueOf(shapeNames[j]);//fs.getNumGrids(j); 
    }
  }
}


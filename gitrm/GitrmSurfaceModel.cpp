#include "GitrmSurfaceModel.hpp"
#include "GitrmInputOutput.hpp"

GitrmSurfaceModel::GitrmSurfaceModel(GitrmMesh& gm, std::string ncFile):
  gm(gm), mesh(gm.mesh),  ncFile(ncFile) { 
  surfaceAndMaterialModelIds = gm.surfaceAndMaterialModelIds;
  initSurfaceModelData(ncFile, true);
}

//includes only sirfaces to be processed
void GitrmSurfaceModel::setFaceId2SurfaceIdMap() {
  auto nf = mesh.nfaces();
  auto surfModelIds = o::LOs(surfaceAndMaterialModelIds);
  auto numIds = surfModelIds.size();
  const auto side_is_exposed = o::mark_exposed_sides(&mesh);
  auto faceClassIds = mesh.get_array<o::ClassId>(2, "class_id");
  o::Write<o::LO> surfMarked(nf, -1, "surfMarked");
  o::Write<o::LO> total(1,0, "total");
  auto lambda = OMEGA_H_LAMBDA(o::LO fid) {
    if(!side_is_exposed[fid])
      return;
    //if surface to be processed
    for(auto id=0; id < numIds; ++id) {
      if(surfModelIds[id] == faceClassIds[fid]) {
        Kokkos::atomic_fetch_add(&(total[0]), 1);
        surfMarked[fid] = 1;
      }
    }
  };
  o::parallel_for(nf, lambda, "makeSurfaceIndMap");
  auto count_h = o::HostWrite<o::LO>(total);
  numDetectorSurfaceFaces = count_h[0];
  auto surfInds_h = o::HostWrite<o::LO>(nf);
  auto surfMarked_h = o::HostRead<o::LO>(surfMarked);
  int bid = 0;
  for(int fid=0; fid< mesh.nfaces(); ++fid) {
    if(surfMarked_h[fid]) {
      surfInds_h[fid] = bid;
      ++bid;
    }
  }
  mesh.add_tag<o::LO>(o::FACE, "SurfaceIndex", 1, o::LOs(surfInds_h.write()));
}

void GitrmSurfaceModel::initSurfaceModelData(std::string ncFile, bool debug) {
  getConfigData(ncFile);
  setFaceId2SurfaceIdMap();
  if(debug)
    std::cout << "Done reading data \n";
  numDetectorSurfaceFaces = gm.numDetectorSurfaceFaces;
  assert(numDetectorSurfaceFaces > 0);
  nDistEsurfaceModel =
     nEnSputtRefDistIn * nAngSputtRefDistIn * nEnSputtRefDistOut;
  nDistEsurfaceModelRef =
     nEnSputtRefDistIn * nAngSputtRefDistIn * nEnSputtRefDistOutRef;
  nDistAsurfaceModel =
     nEnSputtRefDistIn * nAngSputtRefDistIn * nAngSputtRefDistOut;
  if(debug)
    std::cout << "prepareSurfaceModelData \n";

  prepareSurfaceModelData();

  if(gitrm::SURFACE_FLUX_EA > 0) {
    dEdist = (enDist - en0Dist)/nEnDist;
    dAdist = (angDist - ang0Dist)/nAngDist;
  }
  auto nDist = numDetectorSurfaceFaces * nEnDist * nAngDist;
  if(debug)
    printf(" nEdist %d nAdist %d #DetSurfFaces %d nDist %d\n", 
      nEnDist, nAngDist, numDetectorSurfaceFaces, nDist);
  energyDistribution = o::Write<o::Real>(nDist,0,"surfEnDist"); //9k/detFace
  sputtDistribution = o::Write<o::Real>(nDist,0, "surfSputtDist"); //9k/detFace
  reflDistribution = o::Write<o::Real>(nDist,0, "surfReflDist"); //9k/detFace
  auto nf = mesh.nfaces();
  mesh.add_tag<o::Int>(o::FACE, "SumParticlesStrike", 1, o::Read<o::Int>(nf));
  mesh.add_tag<o::Int>(o::FACE, "SputtYldCount", 1, o::Read<o::Int>(nf));
  mesh.add_tag<o::Real>(o::FACE, "SumWeightStrike", 1, o::Reals(nf));
  mesh.add_tag<o::Real>(o::FACE, "GrossDeposition", 1, o::Reals(nf));
  mesh.add_tag<o::Real>(o::FACE, "GrossErosion", 1, o::Reals(nf));
  mesh.add_tag<o::Real>(o::FACE, "AveSputtYld", 1, o::Reals(nf)); 
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
  auto enSputtRefCft = enSputtRefCoeff;
  o::parallel_for(nEnSputtRefCoeff, OMEGA_H_LAMBDA(int i) {
    enLogSputtRefCoef_w[i] = log10(enSputtRefCft[i]);
  });
  enLogSputtRefCoef = o::Reals(enLogSputtRefCoef_w);
  auto enSputtRefDIn = enSputtRefDistIn;
  o::Write<o::Real> enLogSputtRefDistIn_w(nEnSputtRefDistIn);
  o::parallel_for(nEnSputtRefDistIn, OMEGA_H_LAMBDA(int i) {
    enLogSputtRefDistIn_w[i] = log10(enSputtRefDIn[i]);
  });
  //enLogSputtRefDistIn = new o::Reals(enLogSputtRefDistIn_w);
  o::HostWrite<o::Real>enLogSputtRefDistIn_h(enLogSputtRefDistIn_w);
  
  o::Write<o::Real> energyDistGrid01_w(nEnSputtRefDistOut);
  auto nEnSputtRefDOut = nEnSputtRefDistOut;
  o::parallel_for(nEnSputtRefDOut, OMEGA_H_LAMBDA(int i) {
    energyDistGrid01_w[i] = i * 1.0 / nEnSputtRefDOut;
  });
  energyDistGrid01 = o::Reals(energyDistGrid01_w);
  o::HostWrite<o::Real>energyDistGrid01_h(energyDistGrid01_w);

  auto nEnSputtRefDORef = nEnSputtRefDistOutRef;
  o::Write<o::Real> energyDistGrid01Ref_w(nEnSputtRefDORef);
  o::parallel_for(nEnSputtRefDORef, OMEGA_H_LAMBDA(int i) {
    energyDistGrid01Ref_w[i] = i * 1.0 / nEnSputtRefDORef;
  });
  energyDistGrid01Ref = o::Reals(energyDistGrid01Ref_w);
  o::HostWrite<o::Real>energyDistGrid01Ref_h(energyDistGrid01Ref_w);

  auto nAngSputtRefDOut = nAngSputtRefDistOut;
  o::Write<o::Real> angleDistGrid01_w(nAngSputtRefDOut);
  o::parallel_for(nAngSputtRefDOut, OMEGA_H_LAMBDA(int i) {
    angleDistGrid01_w[i] = i * 1.0 / nAngSputtRefDOut;
  });
  angleDistGrid01 = o::Reals(angleDistGrid01_w);
  o::HostWrite<o::Real>angleDistGrid01_h(angleDistGrid01_w);

  printf("Making CDFs\n"); 
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
  
  printf("Making regrid CDFs\n"); 
  o::HostRead<o::Real> angPhiSputtRefDistOut_h(o::deep_copy(angPhiSputtRefDistOut));
  o::HostWrite<o::Real>angPhiDist_CDF_Y_regrid_h(angPhiDist_CDF_Y.size());
  regrid2dCDF<o::Real>(nEnSputtRefDistIn, nAngSputtRefDistIn, nAngSputtRefDistOut,
   angleDistGrid01_h, nAngSputtRefDistOut,
   angPhiSputtRefDistOut_h[nAngSputtRefDistOut - 1],
   angPhiDist_CDF_Y, angPhiDist_CDF_Y_regrid_h);
  angPhiDist_CDF_Y_regrid = o::Reals(angPhiDist_CDF_Y_regrid_h.write());

  o::HostRead<o::Real> angThetaSputtRefDistOut_h(o::deep_copy(angThetaSputtRefDistOut));
  o::HostWrite<o::Real>angThetaDist_CDF_Y_regrid_h(angThetaDist_CDF_Y.size());
  regrid2dCDF<o::Real>(nEnSputtRefDistIn, nAngSputtRefDistIn, nAngSputtRefDistOut,
   angleDistGrid01_h, nAngSputtRefDistOut, angThetaSputtRefDistOut_h[nAngSputtRefDistOut - 1],
   angThetaDist_CDF_Y, angThetaDist_CDF_Y_regrid_h);
  angThetaDist_CDF_Y_regrid = o::Reals(angThetaDist_CDF_Y_regrid_h.write());

  o::HostRead<o::Real> enSputtRefDistOut_h(o::deep_copy(enSputtRefDistOut));
  o::HostWrite<o::Real>enDist_CDF_Y_regrid_h(enDist_CDF_Y.size());
  regrid2dCDF<o::Real>(nEnSputtRefDistIn, nAngSputtRefDistIn, nEnSputtRefDistOut,
   energyDistGrid01_h, nEnSputtRefDistOut, enSputtRefDistOut_h[nEnSputtRefDistOut - 1],
   enDist_CDF_Y, enDist_CDF_Y_regrid_h);
  enDist_CDF_Y_regrid = o::Reals(enDist_CDF_Y_regrid_h);

  o::HostWrite<o::Real>angPhiDist_CDF_R_regrid_h(angPhiDist_CDF_R.size());
  regrid2dCDF<o::Real>(nEnSputtRefDistIn, nAngSputtRefDistIn, nAngSputtRefDistOut,
   angleDistGrid01_h, nAngSputtRefDistOut,
   angPhiSputtRefDistOut_h[nAngSputtRefDistOut - 1],
   angPhiDist_CDF_R, angPhiDist_CDF_R_regrid_h);
  angPhiDist_CDF_R_regrid = o::Reals(angPhiDist_CDF_R_regrid_h);

  o::HostWrite<o::Real>angThetaDist_CDF_R_regrid_h(angThetaDist_CDF_R.size());
  regrid2dCDF<o::Real>(nEnSputtRefDistIn, nAngSputtRefDistIn, nAngSputtRefDistOut,
   angleDistGrid01_h, nAngSputtRefDistOut,
   angThetaSputtRefDistOut_h[nAngSputtRefDistOut - 1],
   angThetaDist_CDF_R, angThetaDist_CDF_R_regrid_h);
  angThetaDist_CDF_R_regrid = o::Reals(angThetaDist_CDF_R_regrid_h);

  o::HostRead<o::Real>enSputtRefDistOutRef_h(o::deep_copy(enSputtRefDistOutRef));
  o::HostWrite<o::Real>enDist_CDF_R_regrid_h(enDist_CDF_R.size());
  regrid2dCDF<o::Real>(nEnSputtRefDistIn, nAngSputtRefDistIn, nEnSputtRefDistOutRef,
   energyDistGrid01Ref_h, nEnSputtRefDistOutRef,
   enSputtRefDistOutRef_h[nEnSputtRefDistOutRef - 1],
   enDist_CDF_R, enDist_CDF_R_regrid_h);
  enDist_CDF_R_regrid = o::Reals(enDist_CDF_R_regrid_h);
}

void GitrmSurfaceModel::getConfigData(std::string ncFileName) {
  //TODO get from config file
  //collect data for analysis/plot
  nEnDist = 50; //100 //TODO FIXME
  en0Dist = 0.0;
  enDist = 1000.0;
  nAngDist = 45; //90
  ang0Dist = 0.0;
  angDist = 90.0; 
  //from NC file ftridynSelf.nc
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

  std::vector<std::string> ds{nEnSputtRefCoeffStr, nAngSputtRefCoeffStr,
   nEnSputtRefDistInStr, nAngSputtRefDistInStr, nEnSputtRefDistOutStr, 
   nEnSputtRefDistOutRefStr, nAngSputtRefDistOutStr};
  std::vector<int> dd{nEnSputtRefCoeff, nAngSputtRefCoeff, nEnSputtRefDistIn, 
   nAngSputtRefDistIn, nEnSputtRefDistOut, nEnSputtRefDistOutRef, nAngSputtRefDistOut};
  std::cout << " getSurfaceModelData \n"; 
  //grids are read as separate data, since grid association with data is complex.
  auto f = fileString;
  getSurfaceModelData(f, sputtYldStr, ds, {0,1}, sputtYld);
  getSurfaceModelData(f, reflYldStr, ds, {0,1}, reflYld);
  getSurfaceModelData(f, enSputtRefCoeffStr, ds, {0}, enSputtRefCoeff,
    &nEnSputtRefCoeff);
  getSurfaceModelData(f, angSputtRefCoeffStr, ds, {1}, angSputtRefCoeff,
    &nAngSputtRefCoeff);
  getSurfaceModelData(f, enSputtRefDistInStr, ds, {2}, enSputtRefDistIn,
    &nEnSputtRefDistIn);
  getSurfaceModelData(f, angSputtRefDistInStr, ds, {3}, angSputtRefDistIn,
    &nAngSputtRefDistIn);
  //TODO nEnSputtRefDistInStr not used
  getSurfaceModelData(f, angPhiDistYStr, ds, {0,1,6}, angPhiDist_Y);
  getSurfaceModelData(f, angThetaDistYStr, ds, {0,1,6}, angThetaDist_Y);
  getSurfaceModelData(f, angPhiDistRStr, ds, {0,1,6}, angPhiDist_R);
  getSurfaceModelData(f, angThetaDistRStr, ds, {0,1,6}, angThetaDist_R);
  o::Reals enDist_Y_temp;
  //enDist_Y = enDist_Y_temp;
  getSurfaceModelData(f, enDistYStr, ds, {0,1,4}, enDist_Y);//_temp);
  getSurfaceModelData(f, enDistRStr, ds, {0,1,5}, enDist_R);
  getSurfaceModelData(f, enSputtRefDistOutStr, ds, {4}, enSputtRefDistOut,
    &nEnSputtRefDistOut);
  getSurfaceModelData(f, enSputtRefDistOutRefStr, ds, {5}, enSputtRefDistOutRef,
    &nEnSputtRefDistOutRef);
  getSurfaceModelData(f, angPhiSputtRefDistOutStr, ds, {6}, angPhiSputtRefDistOut,
    &nAngSputtRefDistOut);
  getSurfaceModelData(f, angThetaSputtRefDistOutStr, ds, {6}, 
    angThetaSputtRefDistOut, &nAngSputtRefDistOut);
}

template<typename T>
void GitrmSurfaceModel::getSurfaceModelData(const std::string fileName,
   const std::string dataName, const std::vector<std::string>& shapeNames,
   const std::vector<int> shapeInds, o::Read<T>& data, int* size) {
  bool debug = true;
  if(debug)
    std::cout << " reading " << dataName << " \n";
  std::vector<std::string> shapes; 
  for(auto j: shapeInds) {
    shapes.push_back(shapeNames[j]);
  }
  //grid not read along with data
  Field3StructInput fs({dataName},{},shapes);
  //TODO pass T
  readInputDataNcFileFS3(fileName, fs, false);
  data = o::Read<T>(fs.data.write());
  //only first
  if(size) {
    *size = fs.getIntValueOf(shapeNames[shapeInds[0]]);//fs.getNumGrids(j); 
    if(debug)
      std::cout<<" size "<< *size <<" "<<shapeNames[shapeInds[0]]<<"\n";
  }
}

void GitrmSurfaceModel::writeOutSurfaceData(std::string fileName) {
/*
  OutputNcFileFieldStruct outStruct({"nSurfaces", "nEnergies", "nAngles"}, 
    {"grossDeposition", "grossErosion", "aveSpyl", "spylCounts", "surfaceNumber",
     "sumParticlesStrike", "sumWeightStrike"}, {nSurfaces, nEdist, nAdist});
  writeOutputNcFile(ptclsHistoryData, numPtcls, dof, outStruct, outNcFileName);
  
    NcFile ncFile1(fileName, NcFile::replace);
    NcDim nc_nLines = ncFile1.addDim("nSurfaces", nSurfaces);
    vector<NcDim> dims1;
    dims1.push_back(nc_nLines);

    vector<NcDim> dimsSurfE;
    dimsSurfE.push_back(nc_nLines);
    NcDim nc_nEnergies = ncFile1.addDim("nEnergies", nEdist);
    NcDim nc_nAngles = ncFile1.addDim("nAngles", nAdist);
    dimsSurfE.push_back(nc_nAngles);
    dimsSurfE.push_back(nc_nEnergies);
    NcVar nc_grossDep = ncFile1.addVar("grossDeposition", ncFloat, nc_nLines);
    NcVar nc_grossEro = ncFile1.addVar("grossErosion", ncFloat, nc_nLines);
    NcVar nc_aveSpyl = ncFile1.addVar("aveSpyl", ncFloat, nc_nLines);
    NcVar nc_spylCounts = ncFile1.addVar("spylCounts", ncInt, nc_nLines);
    NcVar nc_surfNum = ncFile1.addVar("surfaceNumber", ncInt, nc_nLines);
    NcVar nc_sumParticlesStrike =
        ncFile1.addVar("sumParticlesStrike", ncInt, nc_nLines);
    NcVar nc_sumWeightStrike =
        ncFile1.addVar("sumWeightStrike", ncFloat, nc_nLines);
    nc_grossDep.putVar(&grossDeposition[0]);
    nc_surfNum.putVar(&surfaceNumbers[0]);
    nc_grossEro.putVar(&grossErosion[0]);
    nc_aveSpyl.putVar(&aveSputtYld[0]);
    nc_spylCounts.putVar(&sputtYldCount[0]);
    nc_sumParticlesStrike.putVar(&sumParticlesStrike[0]);
    nc_sumWeightStrike.putVar(&sumWeightStrike[0]);
    NcVar nc_surfEDist = ncFile1.addVar("surfEDist", ncFloat, dimsSurfE);
    NcVar nc_surfReflDist = ncFile1.addVar("surfReflDist", ncFloat, dimsSurfE);
    NcVar nc_surfSputtDist =
        ncFile1.addVar("surfSputtDist", ncFloat, dimsSurfE);
    nc_surfEDist.putVar(&energyDistribution[0]);
    nc_surfReflDist.putVar(&reflDistribution[0]);
    nc_surfSputtDist.putVar(&sputtDistribution[0]);
  */
}

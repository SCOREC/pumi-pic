#include <iostream>
#include "pumipic_utils.hpp"
#include "pumipic_adjacency.hpp"
#include "GitrmInputOutput.hpp"

int main(int argc, char** argv) {
  auto lib = Omega_h::Library(&argc, &argv);
  const auto world = lib.world();
  std::string ncFile = "profiles.nc";
  if(argc > 1)
    ncFile = argv[1];
  bool debug = true;
 
  auto positions = o::Reals(o::HostWrite<o::Real>({-0.00594345,0.0192005,0.00165384}));
  int nPts = positions.size()/3;
  //53.8203 -173.868 0  :ionTemp
  Field3StructInput fs({"gradTiR", "gradTiT", "gradTiZ"}, 
    {"gridx_gradTi", "gridz_gradTi"}, {"nX_gradTi", "nZ_gradTi"});
  readInputDataNcFileFS3(ncFile, fs, debug);
  int nR = fs.getNumGrids(0);
  int nZ = fs.getNumGrids(1);
  auto rMin = fs.getGridMin(0);
  auto zMin = fs.getGridMin(1);
  auto dr = fs.getGridDelta(0);
  auto dz = fs.getGridDelta(1);
  if(debug){
    printf(" %s dr%.5f , dz%.5f , rMin%.5f , zMin%.5f \n",
        ncFile.c_str(), dr, dz, rMin, zMin);
    printf("data size %d \n", fs.data.size());
    for(int i=0; i<10 && i<fs.data.size(); ++i)
      printf(" %g", fs.data[i]);
    printf("\n");
  }
  
  auto readInData_d = o::Reals(fs.data);
  o::Write<o::Real> data3d(nPts*3);
  auto lambda = OMEGA_H_LAMBDA(o::LO i) {
    auto fv = o::zero_vector<3>();
    auto pos= o::zero_vector<3>();
    // coords' size is 3* nverts
    for(o::LO j=0; j<3; ++j)
      pos[j] = positions[3*i+j];
    if(debug && i < 5)
      printf(" index i: %d %.5f %.5f  %.5f \n", i, pos[0], pos[1], pos[2]);
    pumipic::interp2dVector(readInData_d, rMin, zMin, dr, dz, nR, nZ, pos, fv, true);
    for(int j=0; j<3; ++j) //components
      data3d[3*i+j] = fv[j]; 
    if(debug && i<10)
      printf("i %d pos: %g %g %g data3d %g %g %g\n", i, pos[3*i+0], pos[3*i+1], 
        pos[3*i+2], data3d[3*i+0], data3d[3*i+1], data3d[3*i+2]);
  };
  o::parallel_for(nPts, lambda, "interpolate2d_test");

  return 0;
}

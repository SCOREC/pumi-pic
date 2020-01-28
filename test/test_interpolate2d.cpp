#include <iostream>
#include "pumipic_utils.hpp"
#include "pumipic_adjacency.hpp"
#include "GitrmInputOutput.hpp"

namespace o = Omega_h;
namespace p = pumipic;

template<typename T>
void testInterpolate2dVector(o::Read<T>& positions, o::Read<T>& readInData_d,
  T rMin, T zMin, T dr, T dz, int nR, int nZ, bool symm, o::Write<T>& data3d) {
  bool debug = false;
  int nPts = positions.size()/3;
  auto lambda = OMEGA_H_LAMBDA(o::LO i) {
    auto fv = o::zero_vector<3>();
    auto pos= o::zero_vector<3>();
    // coords' size is 3* nverts
    for(o::LO j=0; j<3; ++j)
      pos[j] = positions[3*i+j];
    if(debug)
      printf(" index i: %d %.5f %.5f  %.5f \n", i, pos[0], pos[1], pos[2]);
    pumipic::interp2dVector(readInData_d, rMin, zMin, dr, dz, nR, nZ, pos, fv, symm);
    for(int j=0; j<3; ++j) //components
      data3d[3*i+j] = fv[j]; 
    if(debug && i<10)
      printf("i %d pos: %g %g %g data3d %g %g %g\n", i, pos[3*i+0], pos[3*i+1], 
        pos[3*i+2], data3d[3*i+0], data3d[3*i+1], data3d[3*i+2]);
  };
  o::parallel_for(nPts, lambda, "interpolate2d_test");
}

int main(int argc, char** argv) {
  auto lib = Omega_h::Library(&argc, &argv);
  const auto world = lib.world();
  std::string ncFile = "profiles.nc";
  if(argc > 1)
    ncFile = argv[1];
  bool debug = false;
 
  auto positions = o::Read<o::Real>(o::HostWrite<o::Real>(
    {-0.00594345,0.0192005,0.00165384, 0.0113458, -0.0145734, 0.00617403,
      -0.00109902, -0.0297618, 0.00487984, 0.0189906, 0.00744021, 0.00383919,
      0.00763451, -0.0121241, 0.00832174, 0.0141843, -0.0188253, 0.00716646,
      0.0149736, -0.0147498, 0.0062472, -0.00599064, -0.0201362, 0.00482855,
      0.0172442, 0.00414935, 0.0037038, 0.00729308, -0.0117277, 0.00803973,
      .00700167, -0.0167152, 0.00688208, 0.00115834, -0.0103882, 0.00654887,
      0.0236956, -0.0505513, 0.00409376, 0.0216041, 0.00953564, 0.00436738,
      0.00137706, -0.00957907, 0.00943867, 0.0199658, -0.0202374, 0.00732015
    }));
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
  int nPts = positions.size()/3;
  o::Write<o::Real> data3d(nPts*3);
  bool symm = true;
  testInterpolate2dVector(positions, readInData_d, rMin, zMin, dr, dz, nR, 
    nZ, symm, data3d);
  o::parallel_for(nPts, OMEGA_H_LAMBDA(int i) {
    printf("i %d pos: %g %g %g => data3d %g %g %g\n", i, positions[3*i+0], positions[3*i+1], 
        positions[3*i+2], data3d[3*i+0], data3d[3*i+1], data3d[3*i+2]);
  });
}
/*

#pos ITG ETG
-0.00594345 0.0192005 0.00165384   53.8203 -173.868 0    53.8203 -173.868 0
 0.0113458 -0.0145734 0.00617403  -111.809 143.615 0     -111.809 143.615 0
 -0.00109902 -0.0297618 0.00487984   0.366372 9.92146 0  0.366372 9.92146 0
 0.0189906 0.00744021 0.00383919  -169.466 -66.3939 0   -169.466 -66.3939 0
 0.00763451 -0.0121241 0.00832174  -96.9832 154.016 0    -96.9832 154.016 0
 0.0141843 -0.0188253 0.00716646  -109.527 145.363 0     -109.527 145.363 0
 0.0149736 -0.0147498 0.0062472  -129.664 127.726 0
 -0.00599064 -0.0201362 0.00482855  51.9001 174.45 0
 0.0172442 0.00414935 0.0037038   -176.956 -42.5798 0
 0.00729308 -0.0117277 0.00803973  -96.1153 154.559 0
 0.00700167 -0.0167152 0.00688208  -70.3193 167.874 0
 0.00115834 -0.0103882 0.00654887   -20.1699 180.886 0
 0.0236956 -0.0505513 0.00409376   -2.50284 5.33946 0
 0.0216041 0.00953564 0.00436738   -166.509 -73.4939 0
 0.00137706 -0.00957907 0.00943867  -25.8987 180.155 0
 0.0199658 -0.0202374 0.00732015    -7.16392 7.26139 0
 */

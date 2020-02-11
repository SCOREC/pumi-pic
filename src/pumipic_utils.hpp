#ifndef PUMIPIC_UTILS_HPP
#define PUMIPIC_UTILS_HPP
#include <iostream>
#include <cmath>
#include <utility>
#include <string>
#include "Omega_h_for.hpp"
#include "Omega_h_file.hpp"  //gmsh
#include "Omega_h_tag.hpp"
#include "Omega_h_adj.hpp"
#include "Omega_h_array.hpp"
#include "Omega_h_array_ops.hpp"
#include "Omega_h_element.hpp"
#include "Omega_h_scalar.hpp" //divide
#include "Omega_h_mark.hpp"
#include "Omega_h_fail.hpp" //assert

#include "Omega_h_mesh.hpp"
#include "Omega_h_shape.hpp"
#include "Omega_h_build.hpp"
#include "Omega_h_compare.hpp"

#include "pumipic_constants.hpp"

namespace o = Omega_h;

namespace pumipic {

OMEGA_H_DEVICE o::Vector<3> makeVector3FromArray(const o::Real (&arr)[3]) {
  o::Vector<3> v;
  for(o::LO i=0; i<3; ++i)
    v[i] = arr[i];
  return v;
}

//NOTE: uses absolute magnitude, unlike relative magnitude in o::are_close()
OMEGA_H_DEVICE bool almost_equal(o::Real a, o::Real b, o::Real tol = EPSILON,
 o::Real floor = EPSILON) {
  if(std::abs(a) <= floor && std::abs(b) <= floor)
    return true;
  return std::abs(a - b) < tol;
}

//uses absolute magnitude
template<o::LO N>
OMEGA_H_DEVICE bool almost_equal(const o::Vector<N>& a, const Omega_h::Real b, 
  Omega_h::Real tol=EPSILON, Omega_h::Real floor = EPSILON) {
  for(Omega_h::LO i=0; i<N; ++i) {
    if(!almost_equal(a[i],b, tol, floor)) {
      return false;
    }
  }
  return true;
}

//uses absolute magnitude
template<o::LO N>
OMEGA_H_DEVICE bool almost_equal(const o::Vector<N>& a, const o::Vector<N>& b, 
  Omega_h::Real tol=EPSILON, Omega_h::Real floor = EPSILON) {
  for(Omega_h::LO i=0; i<N; ++i) {
    if(!almost_equal(a[i],b[i], tol, floor)) {
      return false;
    }
  }
  return true;
}

//uses absolute magnitude
OMEGA_H_DEVICE bool almost_equal(const Omega_h::Real *a, const Omega_h::Real b, 
  Omega_h::LO n=3, Omega_h::Real tol=EPSILON, Omega_h::Real floor = EPSILON) {
  for(Omega_h::LO i=0; i<n; ++i) {
    if(!almost_equal(a[i],b, tol, floor)) {
      return false;
    }
  }
  return true;
}

template <class Vec>
OMEGA_H_DEVICE bool all_positive(const Vec a, Omega_h::Real tol=EPSILON) {
  auto isPos = 1;
  for(Omega_h::LO i=0; i<a.size(); ++i) {
    const auto gtez = Omega_h::are_close(a[i],0.0,tol,tol) || a[i] > 0;
    isPos = isPos && gtez;
  }
  return isPos;
}

OMEGA_H_DEVICE Omega_h::LO min3(Omega_h::Vector<3> a) {
  o::LO idx = (a[0] < a[1]) ? 0 : 1;
  idx = (a[idx] < a[2]) ? idx : 2;
  return idx;
}

template <class T> OMEGA_H_DEVICE  
Omega_h::LO min_index(const T a, Omega_h::LO n) {
  Omega_h::LO ind=0;
  Omega_h::Real min = a[0];
  for(Omega_h::LO i=0; i<n-1; ++i) {
    if(min > a[i+1]) {
      min = a[i+1];
      ind = i+1;
    }
  }
  return ind;
}

template <class T> OMEGA_H_DEVICE  
Omega_h::LO max_index(const T a, Omega_h::LO n, o::LO beg=0) {
  Omega_h::LO ind=0;
  Omega_h::Real max = a[0];
  for(Omega_h::LO i=beg; i < (beg+n-1); ++i) {
    if(max < a[i+1]) {
      max = a[i+1];
      ind = i+1;
    }
  }
  return ind;
}

OMEGA_H_DEVICE bool compare_array(const Omega_h::Real *a, const Omega_h::Real *b, 
 const Omega_h::LO n, Omega_h::Real tol=EPSILON) {
  for(Omega_h::LO i=0; i<n; ++i) {
    if(abs(a[i]-b[i]) > tol) {
      return false;
    }
  }
  return true;
}

OMEGA_H_DEVICE bool compare_vector_directions(const Omega_h::Vector<3> &va,
     const Omega_h::Vector<3> &vb) {
  for(Omega_h::LO i=0; i<3; ++i) {
    if((va.data()[i] < 0 && vb.data()[i] > 0) ||
       (va.data()[i] > 0 && vb.data()[i] < 0)) {
      return false;
    }
  }
  return true;
}

OMEGA_H_DEVICE o::Real angle_between(const o::Vector<3> v1, 
  const o::Vector<3> v2) {
  o::Real cos = o::inner_product(v1, v2)/ (o::norm(v1) * o::norm(v2));
  return acos(cos);
}

OMEGA_H_DEVICE void cartesian_to_spherical(const o::Real &x, const o::Real &y, 
  const o::Real &z, o::Real &r, o::Real &theta, o::Real &phi) {
  r = sqrt(x*x + y*y + z*z);
  OMEGA_H_CHECK(!(almost_equal(x,0) || almost_equal(r,0)));
  theta = atan(y/x);
  phi = acos(z/r);
}

template< o::LO N>
OMEGA_H_DEVICE void print_matrix(const Omega_h::Matrix<3, N> &M) {
  for(o::LO i=0; i<N; ++i)
  printf("M%d %.4f, %.4f, %.4f\n", i, M[i][0], M[i][1], M[i][2]);
}

template< o::LO N>
OMEGA_H_DEVICE void print_few_vectors(const o::Few<o::Vector<3>, N> &M) {
  for(o::LO i=0; i<N; ++i)
  printf("%d: %.4f, %.4f, %.4f\n", i, M[i][0], M[i][1], M[i][2]);
}

OMEGA_H_DEVICE void printPtclPathEndPointsAndTet(o::LO id, o::LO elem, 
    o::Vector<3>& orig, o::Vector<3>& dest, o::Matrix<3, 4>& M) {
  printf("PATH ptcl %d e: %d orig: %g %g %g dest: %g %g %g "
        "Tet: %g %g %g  %g %g %g  %g %g %g  %g %g %g \n", 
    id, elem, orig[0], orig[1], orig[2], dest[0], dest[1], dest[2], 
    M[0][0], M[0][1], M[0][2], M[1][0], M[1][1], M[1][2],
    M[2][0], M[2][1], M[2][2], M[3][0], M[3][1], M[3][2]);
}

/** @brief To interpolate field ONE component at atime from 
 *    nComp-component(dof) 2D data
 *  @Note: This function is only for regular structured grid of data.
 *  @param[in]  data, n-component(dof) array (intended for 3 component tag), but 
 *    interpolation is done on 1 comp, at a time,
 *  @param[in] comp, nth component, out of degree of freedom
 *  @return value corresponding to comp
 */
OMEGA_H_DEVICE o::Real interpolate2dField(const o::Reals& data, 
  const o::Real gridx0, const o::Real gridz0, const o::Real dx, 
  const o::Real dz, const o::LO nx, const o::LO nz, 
  const o::Vector<3> &pos, const bool cylSymm = true, 
  const o::LO nComp = 1, const o::LO comp = 0, bool debug= false) {
  
  if(nx*nz == 1) {
    return data[comp];
  }  
  o::Real fxz = 0;
  o::Real fx_z1 = 0;
  o::Real fx_z2 = 0; 
  auto dim1 = pos[0];
  auto z = pos[2]; 
  if(cylSymm) {
      dim1 = sqrt(pos[0]*pos[0] + pos[1]*pos[1]);
  }

  OMEGA_H_CHECK(dx >0 && dz>0);
  o::LO i = floor((dim1 - gridx0)/dx);
  o::LO j = floor((z - gridz0)/dz);
  
  if (i < 0) i=0;
  if (j < 0) j=0;

  auto gridXi = gridx0 + i * dx;
  auto gridXip1 = gridx0 + (i+1) * dx;    
  auto gridZj = gridz0 + j * dz;
  auto gridZjp1 = gridz0 + (j+1) * dz; 

  if (i >=nx-1 && j>=nz-1) {
    fxz = data[(nx-1+(nz-1)*nx)*nComp+comp];
  }
  else if (i >=nx-1) {
    fx_z1 = data[(nx-1+j*nx)*nComp + comp];
    fx_z2 = data[(nx-1+(j+1)*nx)*nComp + comp];
    fxz = ((gridZjp1-z)*fx_z1+(z - gridZj)*fx_z2)/dz;
  }
  else if (j >=nz-1) {
    fx_z1 = data[(i+(nz-1)*nx)*nComp + comp];
    fx_z2 = data[(i+(nz-1)*nx)*nComp+ comp];
    fxz = ((gridXip1-dim1)*fx_z1+(dim1 - gridXi)*fx_z2)/dx;
  }
  else {
    fx_z1 = ((gridXip1-dim1)*data[(i+j*nx)*nComp + comp] + 
            (dim1 - gridXi)*data[(i+1+j*nx)*nComp + comp])/dx;
    fx_z2 = ((gridXip1-dim1)*data[(i+(j+1)*nx)*nComp + comp] + 
            (dim1 - gridXi)*data[(i+1+(j+1)*nx)*nComp + comp])/dx; 
    fxz = ((gridZjp1-z)*fx_z1+(z - gridZj)*fx_z2)/dz;
  }
  if(debug)
    printf("interp2dField pos: %g %g %g : dim1 %g nx %d nz %d gridx0 %g " 
      "gridz0 %g grid1 %g grid2 %g i %d j %d dx %g dz %g fxz %g \n",
      pos[0], pos[1], pos[2], dim1, nx, nz, gridx0, gridz0, gridXi, 
      gridXip1, i, j, dx, dz, fxz);
  return fxz;
}
/*
//TODO make unit test interpolate2dField(
index,data:  418  1.82241e+18 419 1.55216e+18  518 1.82241e+18 519 1.55216e+18  
  pos: 0.013715 -0.0183798 7.45029e-06 : dim1 0.0229329 
  nx 100 nz 50 gridx0 0 gridz0 -0.05 i 18 j 4 dx 0.00121212 dz 0.0102041 
  fxz 1.57387e+18 
index,data:  418  2.29957 419 1.97687  518 2.29957 519 1.97687  
 pos: 0.013715 -0.0183798 7.45029e-06 : dim1 0.0229329 
 nx 100 nz 50 gridx0 0 gridz0 -0.05 i 18 j 4 dx 0.00121212 dz 0.0102041 
 fxz 2.0028 
*/

OMEGA_H_DEVICE o::Real interpolate2d_base(const o::Real d1, const o::Real d2,
  const o::Real grid1, const o::Real grid2, const o::Real v, const o::Real dv) {
  return (d1 * (grid2 - v) + d2 * (v - grid1))/dv;
}

OMEGA_H_DEVICE o::Real interpolate2d_baseg(const o::Reals& data, const int di,
   const o::Reals& grid, const int gi, const o::Real v, const o::Real dv) {
  return interpolate2d_base(data[di], data[di+1], grid[gi], grid[gi+1], v, dv);
}

OMEGA_H_DEVICE o::Real interpolate2d_based(const o::Reals& data, const int di1,
  const int di2, const o::Real grid1, const o::Real grid2, const o::Real v, const o::Real dv) {
  return interpolate2d_base(data[di1], data[di2], grid1, grid2, v, dv);
}

OMEGA_H_DEVICE o::Real interpolate2d(const o::Reals& data, const o::Real gridXi,
  const o::Real gridXip1, const o::Real gridZj, const o::Real gridZjp1, 
  const o::Real x0, const o::Real z, const o::LO nx, const o::LO nz, const int i,
  const int j, const o::Real y=0, const bool cylSymm = true, const o::LO nComp = 1, 
  const o::LO comp = 0, bool debug=false) {
  if(nx <= 1 && nz <= 1)
    return data[comp];
  auto x = x0; 
  auto dx = gridXip1 - gridXi;
  auto dz = gridZjp1 - gridZj;
  if(cylSymm)
    x = sqrt(x*x + y*y);
  o::Real fxz = 0;
  if (i >=nx-1 && j>=nz-1) {
    fxz = data[(nx-1+(nz-1)*nx)*nComp+comp];
  }
  else if (i >=nx-1) {
    fxz = interpolate2d_based(data, (nx-1+j*nx)*nComp + comp, 
        (nx-1+(j+1)*nx)*nComp + comp, z - gridZj, gridZjp1-z, z, dz);
  }
  else if (j >=nz-1) {
    fxz = interpolate2d_based(data, (i+(nz-1)*nx)*nComp + comp, 
        (i+(nz-1)*nx)*nComp+ comp, x - gridXi, gridXip1-x, x, dx);
  }
  else {
    auto fx_z1 = interpolate2d_based(data, (i+j*nx)*nComp + comp,
      (i+1+j*nx)*nComp + comp, gridXi, gridXip1, x, dx); 
    auto fx_z2 = interpolate2d_based(data, (i+(j+1)*nx)*nComp + comp,
      (i+1+(j+1)*nx)*nComp + comp, gridXi, gridXip1, x, dx);
    fxz = interpolate2d_base(fx_z1, fx_z2, gridZj, gridZjp1, z, dz); 
    if(debug) {
      auto test1 = ((gridXip1-x)*data[(i+j*nx)*nComp + comp] + 
              (x - gridXi)*data[(i+1+j*nx)*nComp + comp])/dx;
      auto test2 = ((gridXip1-x)*data[(i+(j+1)*nx)*nComp + comp] + 
        (x - gridXi)*data[(i+1+(j+1)*nx)*nComp + comp])/dx; 
      auto val = ((gridZjp1-z)*test1+(z - gridZj)*test2)/dz;
      printf("int2d fxz %g fx_z1 %g fx_z2 %g test1 %g test2 %g val %g \n", 
        fxz, fx_z1, fx_z2, test1, test2, val);
    }
  }
  if(debug)
    printf("int2d: pos %g %g %g dx,dz %g %g i,j %d %d nx,nz %d %d "
      " grids %g %g %g %g fxz %g\n", x,y,z, dx, dz, i, j, nx, nz, 
      gridXi, gridXip1, gridZj, gridZjp1, fxz);
  return fxz;
}

OMEGA_H_DEVICE o::Real interpolate2d_field(const o::Reals& data, 
  const o::Real gridx0, const o::Real gridz0, const o::Real dx, 
  const o::Real dz, const o::LO nx, const o::LO nz, 
  const o::Vector<3> &pos, const bool cylSymm = true, 
  const o::LO nComp = 1, const o::LO comp = 0, bool debug= false) {
  if(nx <=1 && nz <= 1)
    return data[comp];
  auto x = pos[0];
  auto z = pos[2]; 
  if(cylSymm)
    x = sqrt(x*x+pos[1]*pos[1]);
  OMEGA_H_CHECK(dx >0 && dz>0);
  o::LO i = floor((x - gridx0)/dx);
  o::LO j = floor((z - gridz0)/dz);
  if (i < 0) i=0;
  if (j < 0) j=0;
  auto gridXi = gridx0 + i * dx;
  auto gridXip1 = gridx0 + (i+1) * dx;    
  auto gridZj = gridz0 + j * dz;
  auto gridZjp1 = gridz0 + (j+1) * dz; 
  if(debug)
    printf("2d_field: pos %g %g %g dx,z %g %g i,j %d %d grids %g %g %g %g\n", 
      x,pos[1],z, dx, dz, i, j, gridXi, gridXip1, gridZj, gridZjp1);
  return interpolate2d(data, gridXi, gridXip1, gridZj, gridZjp1, x, z, nx, 
    nz, i, j, 0, false, nComp, comp, debug);  
}

OMEGA_H_DEVICE o::Real interpolate2d_wgrid(const o::Reals& data, 
  const o::Reals& gridx, const o::Reals& gridz, const o::LO nx, const o::LO nz, 
  const o::Vector<3>& pos, const bool cylSymm = true, 
  const o::LO nComp = 1, const o::LO comp = 0, bool debug= false) {
  auto x = pos[0];
  auto z = pos[2]; 
  auto dx = gridx[1] - gridx[0];
  auto dz = gridz[1] - gridz[0];
  OMEGA_H_CHECK(dx >0 && dz>0);
  o::LO i = floor((x - gridx[0])/dx);
  o::LO j = floor((z - gridz[0])/dz);
  if (i < 0) i=0;
  if (j < 0) j=0;
  auto gridXi = gridx[i];
  auto gridXip1 = gridx[i+1];    
  auto gridZj = gridz[j];
  auto gridZjp1 = gridz[j+1];
  if(debug)
    printf("pos %g %g %g dx,z %g %g i,j %d %d grids %g %g %g %g\n", 
      x,pos[1],z, dx, dz, i, j, gridXi, gridXip1, gridZj, gridZjp1);
  return interpolate2d(data, gridXi, gridXip1, gridZj, gridZjp1, x, z, nx, 
    nz, i, j, pos[1], cylSymm, nComp, comp, debug);
}

OMEGA_H_DEVICE o::Real interpolate2d_wgrid(const o::Reals& data, 
  const o::Reals& gridx, const o::Reals& gridz, const o::LO nx, const o::LO nz, 
  const o::Real x, const o::Real z, const bool cylSymm = true, 
  const o::LO nComp = 1, const o::LO comp = 0, bool debug= false) {
  auto pos = o::zero_vector<3>();
  pos[0] = x;
  pos[1]= 0;
  pos[2] = z;
  return interpolate2d_wgrid(data, gridx, gridz, nx, nz, pos, cylSymm, nComp,
   comp, debug);  
}

OMEGA_H_DEVICE o::Real interpolate3d_field(const o::Real x, const o::Real y, 
    const o::Real z, int nx, int ny, int nz, const o::Reals& gridx, 
    const o::Reals& gridy, const o::Reals& gridz, const o::Reals& data) {
    /*auto nx = gridx.size(); 
    auto ny = gridy.size();
    auto nz = gridz.size(); */
    bool debug = false;
    if(debug)
      for(int i=0; i<10; ++i) printf(" %d %g \n", i, gridz[i]);
    o::Real fxyz = 0;
    o::Real dx = gridx[1] - gridx[0];
    o::Real dy = gridy[1] - gridy[0];
    o::Real dz = gridz[1] - gridz[0];
    OMEGA_H_CHECK(!(o::are_close(dx, 0) || o::are_close(dy, 0) || o::are_close(dz, 0)));
    int i = floor((x - gridx[0])/dx);
    int j = floor((y - gridy[0])/dy);
    int k = floor((z - gridz[0])/dz);
    i = (i < 0) ? 0 : ((i >= nx-1) ? (nx-2) : i);
    j = (j < 0 || ny <= 1) ? 0 : ((j >= ny-1) ? (ny-2) : j);
    k = (k < 0 || nz <= 1) ? 0 : ((k >= nz-1) ? (nz-2) : k);

    auto fx_z0 = interpolate2d_baseg(data, i + j*nx + k*nx*ny, gridx, i, x, dx);
    auto fx_z1 = interpolate2d_baseg(data, i + j*nx + (k+1)*nx*ny, 
      gridx, i, x, dx);
    auto fxy_z0 = interpolate2d_baseg(data, i + (j+1)*nx + k*nx*ny,
      gridx, i, x, dx);
    auto fxy_z1 = interpolate2d_baseg(data,i + (j+1)*nx + (k+1)*nx*ny,
      gridx, i, x, dx);
    auto fxz0 = interpolate2d_base(fx_z0, fx_z1, gridz[k], gridz[k+1], z, dz);
    auto fxz1 = interpolate2d_base(fxy_z0, fxy_z1, gridz[k], gridz[k+1], z, dz);
    fxyz = interpolate2d_base(fxz0, fxz1, gridy[j], gridy[j+1], y, dy);
    if(debug) {
      printf("fx_z0 %g fx_z1 %g \n", fx_z0, fx_z1);
      printf("fxy_z0 %g fxy_z1 %g fxz0 %g fxz1 %g fxyz %g\n",
        fxy_z0, fxy_z1, fxz0, fxz1, fxyz);

      printf("x %g y %g z %g i %d j %d k %d dx %g dy %g dz %g \n", 
        x, y, z, i, j, k, dx, dy, dz);
      auto fx_z0_test = (data[i + j*nx + k*nx*ny]*(gridx[i+1]-x) + 
        data[i +1 + j*nx + k*nx*ny]*(x-gridx[i]))/dx;
      auto fx_z1_test = (data[i + j*nx + (k+1)*nx*ny]*(gridx[i+1]-x) + 
        data[i +1 + j*nx + (k+1)*nx*ny]*(x-gridx[i]))/dx;
      printf("fx_z0_test %g fx_z1_test %g \n", fx_z0_test, fx_z1_test);
      auto fxy_z0_test = (data[i + (j+1)*nx + k*nx*ny]*(gridx[i+1]-x) + 
        data[i +1 + (j+1)*nx + k*nx*ny]*(x-gridx[i]))/dx;
      auto fxy_z1_test = (data[i + (j+1)*nx + (k+1)*nx*ny]*(gridx[i+1]-x) + 
        data[i +1 + (j+1)*nx + (k+1)*nx*ny]*(x-gridx[i]))/dx;
      auto fxz0_test = (fx_z0*(gridz[k+1] - z) + fx_z1*(z-gridz[k]))/dz;
      auto fxz1_test = (fxy_z0*(gridz[k+1] - z) + fxy_z1*(z-gridz[k]))/dz;
      auto fxyz_test = (fxz0*(gridy[j+1] - y) + fxz1*(y-gridy[j]))/dy;
      printf("fxy_z0_test %g fxy_z1_test %g fxz0_test %g fxz1_test %g fxyz_test %g\n",
        fxy_z0_test, fxy_z1_test, fxz0_test, fxz1_test, fxyz_test);
      fxyz_test = (ny <= 1) ? fxz0: fxyz_test;
      fxyz_test = (nz <= 1) ? fx_z0: fxyz_test;
      printf(" fxy_test %g\n", fxyz_test);

    }
    fxyz = (ny <= 1) ? fxz0: fxyz;
    fxyz = (nz <= 1) ? fx_z0: fxyz;
    if(debug)
      printf("fxy %g\n", fxyz);
    return fxyz;
}

OMEGA_H_DEVICE void interp2dVector (const o::Reals& data3, const o::Real gridx0, 
  const o::Real gridz0, const o::Real dx, const o::Real dz, const o::LO nx, 
  const o::LO nz, const o::Vector<3> &pos, o::Vector<3> &field, 
  const bool cylSymm = false, int* ptcl=nullptr) {
  bool debug = false;
  if(ptcl) debug = true;
  field[0] = interpolate2d_field(data3, gridx0, gridz0, dx, dz, nx,
    nz, pos, cylSymm, 3, 0, debug);
  field[1] = interpolate2d_field(data3, gridx0, gridz0, dx, dz, nx, 
    nz, pos, cylSymm, 3, 1, debug);
  field[2] = interpolate2d_field(data3, gridx0, gridz0, dx, dz, nx, 
    nz, pos, cylSymm, 3, 2, debug);
  auto f1 = interpolate2dField(data3, gridx0, gridz0, dx, dz, nx, nz, 
    pos, cylSymm, 3, 0, debug);
  auto f2 = interpolate2dField(data3, gridx0, gridz0, dx, dz, nx, nz, 
    pos, cylSymm, 3, 1, debug);
  auto f3 = interpolate2dField(data3, gridx0, gridz0, dx, dz, nx, nz, 
    pos, cylSymm, 3, 2, debug);
  if(cylSymm) {
    auto theta = atan2(static_cast<double>(pos[1]), static_cast<double>(pos[0]));  
    auto field0 = field[0];
    auto field1 = field[1]; 
    field[0] = cos(theta)*field0 - sin(theta)*field1;
    field[1] = sin(theta)*field0 + cos(theta)*field1;
  }
}

OMEGA_H_DEVICE o::Vector<3> centroid_of_triangle(const o::Vector<3>& va,
  const o::Vector<3>& vb, const o::Vector<3>& vc) {
  o::Vector<3> pos;
  pos[0] = va[0] + 0.666666667*(vb[0] + 0.5*(vc[0] - vb[0]) - va[0]); //2.0/3.0
  pos[1] = va[1] + 0.666666667*(vb[1] + 0.5*(vc[1] - vb[1]) - va[1]);
  pos[2] = va[2] + 0.666666667*(vb[2] + 0.5*(vc[2] - vb[2]) - va[2]);
  return pos;  
}

OMEGA_H_DEVICE o::Vector<3> centroid_of_triangle(const o::Matrix<3, 3>& abc) {
  return centroid_of_triangle(abc[0], abc[1], abc[2]);
}

OMEGA_H_DEVICE o::Vector<3> face_centroid_of_tet(const o::LO fid, 
  const o::Reals &coords, const o::LOs &face_verts) {
  const auto facev = o::gather_verts<3>(face_verts, fid);
  const auto abc = Omega_h::gather_vectors<3, 3>(coords, facev);
  //TODO check if y and z are in required order
  return centroid_of_triangle(abc);
}

OMEGA_H_DEVICE o::Vector<3> centroid_of_tet(const o::LO elem, 
  const o::LOs &mesh2verts,  const o::Reals &coords) {
  o::Vector<3> pos;
  auto tetv2v = o::gather_verts<4>(mesh2verts, elem);
  auto M = o::gather_vectors<4, 3>(coords, tetv2v);
  pos[0]= (M[0][0]+M[1][0]+M[2][0]+M[3][0])/4;
  pos[1]= (M[0][1]+M[1][1]+M[2][1]+M[3][1])/4;
  pos[2]= (M[0][2]+M[1][2]+M[2][2]+M[3][2])/4;
  return pos;
}

//2,3 nodes of faces. 0,2,1; 0,1,3; 1,2,3; 2,0,3
OMEGA_H_DEVICE o::LO getFaceMap(const o::LO i) {
  assert(i>=0 && i<8);
  const o::LO fmap[8] = {2,1,1,3,2,3,0,3};
  return fmap[i];
}

OMEGA_H_DEVICE bool isFaceFlipped(const o::LO fi, const o::Few<o::LO, 3>& fv2v, 
  const o::Few<o::LO, 4>& tetv2v) {
  auto matInd1 = getFaceMap(fi*2);
  auto matInd2 = getFaceMap(fi*2+1);
  bool flip = true;
  if(fv2v[1] == tetv2v[matInd1] && fv2v[2] == tetv2v[matInd2])
    flip = false;
  return flip;       
}

//TODO: this is not tested ?
//face normal can point either way
OMEGA_H_DEVICE o::Vector<3> face_normal_of_tet(const o::LO fid, const o::LO elmId,
  const o::Reals &coords, const o::LOs& mesh2verts,  const o::LOs &face_verts, 
  const o::LOs &elem2faces) {
  const auto fv2v = o::gather_verts<3>(face_verts, fid);
  const auto abc = Omega_h::gather_vectors<3, 3>(coords, fv2v);
  const auto tetv2v = o::gather_verts<4>(mesh2verts, elmId);
  o::LO findex = -1;
  o::LO find = 0;
  for(auto iface = elmId *4; iface < elmId *4 +4; ++iface) {
    const auto face_id = elem2faces[iface];
    if(fid == face_id) {
      findex = find;
      break;
    }
    ++find;
  }
  if(findex <0 || findex >3) {
    printf("face_normal_of_tet:getFaceMap:: faceid not found fid %d elmId %d \n",
        fid, elmId);
    OMEGA_H_CHECK(false);
  }
  auto a = abc[0];
  auto b = abc[1];
  auto c = abc[2];
  auto fnorm = o::cross(b - a, c - a);
  if(isFaceFlipped(findex, fv2v, tetv2v))
    fnorm = -1*fnorm;
  return o::normalize(fnorm);
}

// TODO boundary face normal points always outwards
OMEGA_H_DEVICE o::Vector<3> bdry_face_normal_of_tet(const o::LO fid, 
  const o::Reals &coords, const o::LOs &face_verts) {

  const auto fv2v = o::gather_verts<3>(face_verts, fid);
  const auto abc = Omega_h::gather_vectors<3, 3>(coords, fv2v);
  
  o::Vector<3> a = abc[0];
  o::Vector<3> b = abc[1];
  o::Vector<3> c = abc[2];
  o::Vector<3> fnorm = o::cross(b - a, c - a);
  return o::normalize(fnorm);
}


OMEGA_H_DEVICE o::LO elem_id_of_bdry_face_of_tet(const o::LO fid, 
  const o::LOs &f2r_ptr, const o::LOs &f2r_elem) { 
  //bdry
  OMEGA_H_CHECK(f2r_ptr[fid+1] - f2r_ptr[fid] == 1);
  auto ind = f2r_ptr[fid];
  return f2r_elem[ind];
}

//retrieve face coords in the Omega_h order
OMEGA_H_DEVICE void get_face_from_face_index_of_tet(const Omega_h::Matrix<3, 4> &M,
  const Omega_h::LO iface, Omega_h::Few<Omega_h::Vector<3>, 3> &abc) {
  //face_vert:0,2,1; 0,1,3; 1,2,3; 2,0,3
  OMEGA_H_CHECK((iface<4) && (iface>=0));
  abc[0] = M[Omega_h::simplex_down_template(3, 2, iface, 0)];
  abc[1] = M[Omega_h::simplex_down_template(3, 2, iface, 1)];
  abc[2] = M[Omega_h::simplex_down_template(3, 2, iface, 2)];
}

// WARNING: this doesn't give vertex ordering right, so surface normal may be wrong
OMEGA_H_DEVICE o::Matrix<3, 3> get_face_coords_of_tet(const o::LOs& mesh2verts, 
  const o::Reals& coords, const o::LO elem, const o::LO findex) {
  o::Matrix<3, 3> face;
  auto tetv2v = o::gather_verts<4>(mesh2verts, elem);
  auto tet = o::gather_vectors<4, 3>(coords, tetv2v);
  get_face_from_face_index_of_tet(tet, findex, face);
  return face;
}

//TODO replace use of this by the version using Matrix instead of flat array
OMEGA_H_DEVICE void get_face_coords_of_tet(const o::LOs &face_verts, 
  const o::Reals &coords, const o::LO face_id, o::Real (&fdat)[9]) {
  auto fv2v = o::gather_verts<3>(face_verts, face_id);
  const auto face = o::gather_vectors<3, 3>(coords, fv2v);
  for(auto i=0; i<3; ++i)
    for(auto j=0; j<3; ++j)
      fdat[i*3+j] = face[i][j];
}

OMEGA_H_DEVICE o::Matrix<3,3> get_face_coords_of_tet(const o::LOs &face_verts, 
  const o::Reals &coords, const o::LO face_id) {
  const auto fv2v = o::gather_verts<3>(face_verts, face_id);
  return o::gather_vectors<3, 3>(coords, fv2v);
}

OMEGA_H_DEVICE o::LO is_face_within_limit_from_tet(const o::Matrix<3, 4>& tet, 
  const Omega_h::LOs& face_verts, const Omega_h::Reals& coords, const o::LO face_id,
  const o::Real depth) {
  auto fv2v = Omega_h::gather_verts<3>(face_verts, face_id); //Few<LO, 3>
  const auto face = Omega_h::gather_vectors<3, 3>(coords, fv2v);
  o::LO res = 0;
  for(o::LO i=0; i<3; ++i) { //3 vtx of face
    for(o::LO j=0; j<4; ++j) { //4 vtx of tet
      auto dv = face[i] - tet[j];
      auto d2 = o::norm(dv);
      if(d2 <= depth) {
        res = 1;
        break;
      }
      if(res)
        break;
    }
  }
  return res;
}

OMEGA_H_DEVICE o::LO is_tet_within_limit_from_tet(const o::Matrix<3, 4>& tet1, 
   const o::Matrix<3, 4>& tet2, const o::Real depth) {
  o::LO res = 0;
  for(o::LO i=0; i<4; ++i) {
    for(o::LO j=0; j<4; ++j) {
      auto dv = tet1[i] - tet2[j];
      auto d2 = o::norm(dv);
      if(d2 <= depth) {
        res = 1;
        break;
      }
      if(res)
        break;
    }
  }
  return res;
}

OMEGA_H_DEVICE void get_edge_coords_of_tet_face(
   const Omega_h::Few<Omega_h::Vector<3>, 3> &abc, const Omega_h::LO iedge, 
   Omega_h::Few<Omega_h::Vector<3>, 2> &ab) {
  //edge_vert:0,1; 1,2; 2,0
  ab[0] = abc[Omega_h::simplex_down_template(2, 1, iedge, 0)];
  ab[1] = abc[Omega_h::simplex_down_template(2, 1, iedge, 1)];
}

// WARNING: check vertex ordering right, so surface normal may be wrong
OMEGA_H_DEVICE void check_face_of_tet(const Omega_h::Matrix<3, 4> &M,
  const Omega_h::Few<Omega_h::Vector<3>, 3>& face, const Omega_h::LO faceid) {
  Omega_h::Few<Omega_h::Vector<3>, 3> abc;
  get_face_from_face_index_of_tet( M, faceid, abc);
  OMEGA_H_CHECK(true == compare_array(abc[0].data(), face[0].data(), 3)); //a
  OMEGA_H_CHECK(true == compare_array(abc[1].data(), face[1].data(), 3)); //b
  OMEGA_H_CHECK(true == compare_array(abc[2].data(), face[2].data(), 3)); //c
}

// TODO delete this
// type =1 for  interior, 2=bdry
OMEGA_H_DEVICE o::LO get_face_types_of_tet(const o::LO elem, 
  const o::LOs &down_r2f, const o::Read<o::I8> &side_is_exposed, 
  o::LO (&fids)[4], const o::LO type) {
  o::LO nf = 0;
  for(o::LO fi = elem *4; fi < (elem+1)*4; ++fi){
    const auto fid = down_r2f[fi];
    if( (type==1 && !side_is_exposed[fid]) ||
        (type==2 &&  side_is_exposed[fid]) ) {
      fids[nf] = fid;
      ++nf;
    }
  }
  return nf;
}

OMEGA_H_DEVICE o::LO get_exposed_face_ids_of_tet(const o::LO elem, 
  const o::LOs &down_r2f, const o::Read<o::I8> &side_is_exposed, 
  o::LO (&fids)[4]) {
  o::LO nf = 0;
  fids[0] = fids[1] = fids[2] = fids[3] = -1;
  for(o::LO fi = elem *4; fi < (elem+1)*4; ++fi){
    const auto fid = down_r2f[fi];
    if(side_is_exposed[fid]) {
      fids[nf] = fid;
      ++nf;
    }
  }
  return nf;
}

OMEGA_H_DEVICE o::LO get_interior_face_ids_of_tet(const o::LO elem, 
  const o::LOs &down_r2f, const o::Read<o::I8> &side_is_exposed, 
  o::LO (&fids)[4]) {
  o::LO nf = 0;
  fids[0] = fids[1] = fids[2] = fids[3] = -1;
  for(o::LO fi = elem *4; fi < (elem+1)*4; ++fi){
    const auto fid = down_r2f[fi];
    if(!side_is_exposed[fid]) {
      fids[nf] = fid;
      ++nf;
    }
  }
  return nf;
}

} //namespace
#endif

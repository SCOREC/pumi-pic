#ifndef PUMIPIC_UTILS_HPP
#define PUMIPIC_UTILS_HPP
#include <iostream>
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
  if(Kokkos::abs(a) <= floor && Kokkos::abs(b) <= floor)
    return true;
  return Kokkos::abs(a - b) < tol;
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
Omega_h::LO find_min_no_exp(const T a, const T exp_num, Omega_h::LO n){
  
  Omega_h::LO ind=0;
  Omega_h::Real min = a[0];
  for (Omega_h::LO i=0; i<n; ++i){
    if(exp_num[i]==0){
      ind = i;
      min = a[i];
      break;
    }
  }
 
  
  Omega_h::LO ind_track=0;
  Omega_h::LO ind_ret=0;
  
  for(Omega_h::LO i=ind; i<n-1; ++i) {
    if(exp_num[i+1]==0) {
      ind_track++;
      if(a[i+1]<min){
        min = a[i+1];
        ind = i+1;
	ind_ret=ind_track;
      }
      
    }
  }
  return ind_ret;
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
  return Kokkos::acos(cos);
}

OMEGA_H_DEVICE void cartesian_to_spherical(const o::Real &x, const o::Real &y, 
  const o::Real &z, o::Real &r, o::Real &theta, o::Real &phi) {
  r = Kokkos::sqrt(x*x + y*y + z*z);
  OMEGA_H_CHECK(!(almost_equal(x,0) || almost_equal(r,0)));
  theta = Kokkos::atan(y/x);
  phi = Kokkos::acos(z/r);
}

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
      dim1 = Kokkos::sqrt(pos[0]*pos[0] + pos[1]*pos[1]);
  }

  OMEGA_H_CHECK(dx >0 && dz>0);
  o::LO i = Kokkos::floor((dim1 - gridx0)/dx);
  o::LO j = Kokkos::floor((z - gridz0)/dz);
  
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
    printInfo("interp2dField pos: %g %g %g : dim1 %g nx %d nz %d gridx0 %g " 
      "gridz0 %g grid1 %g grid2 %g i %d j %d dx %g dz %g fxz %g \n",
      pos[0], pos[1], pos[2], dim1, nx, nz, gridx0, gridz0, gridXi, 
      gridXip1, i, j, dx, dz, fxz);
  return fxz;
} 

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
  const int j, const o::Real dx, const o::Real dz, const o::Real y=0,
  const bool cylSymm = true, const o::LO nComp = 1, const o::LO comp = 0, bool debug=false) {
  if(nx <= 1 && nz <= 1)
    return data[comp];
  auto x = x0; 
  if(cylSymm)
    x = Kokkos::sqrt(x*x + y*y);
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
    if(debug)
      printInfo("fx_z1 %.15f  fx_z2 %.15f \n", fx_z1, fx_z2);
  }
  if(debug)
    printInfo("int2d: pos %g %g %g dx,dz %g %g i,j %d %d nx,nz %d %d \n"
      " grids %.15f %.15f %.15f %.15f fxz %.15f\n\n", x,y,z, dx, dz, i, j, nx, nz, 
      gridXi, gridXip1, gridZj, gridZjp1, fxz);
  return fxz;
}

OMEGA_H_DEVICE o::Real interpolate2d_field(const o::Reals& data, const o::Real gridx0, 
  const o::Real gridz0, const o::Real dx, const o::Real dz, const o::LO nx,
  const o::LO nz, const o::Vector<3> &pos, const bool cylSymm = true,
  const o::LO nComp = 1, const o::LO comp = 0, bool debug= false) {
  if(nx <=1 && nz <= 1)
    return data[comp];
  auto x = pos[0];
  auto z = pos[2]; 
  if(cylSymm)
    x = Kokkos::sqrt(x*x+pos[1]*pos[1]);
  OMEGA_H_CHECK(dx >0 && dz>0);
  o::LO i = Kokkos::floor((x - gridx0)/dx);
  o::LO j = Kokkos::floor((z - gridz0)/dz);
  if (i < 0) i=0;
  if (j < 0) j=0;
  auto gridXi = gridx0 + i * dx;
  auto gridXip1 = gridx0 + (i+1) * dx;    
  auto gridZj = gridz0 + j * dz;
  auto gridZjp1 = gridz0 + (j+1) * dz; 
  if(debug)
    printInfo("2d_field: pos %g %g %g dx,z %g %g i,j %d %d grids %g %g %g %g\n", 
      x,pos[1],z, dx, dz, i, j, gridXi, gridXip1, gridZj, gridZjp1);
  return interpolate2d(data, gridXi, gridXip1, gridZj, gridZjp1, x, z, nx, 
    nz, i, j, dx, dz, 0, false, nComp, comp, debug);  
}

/** @brief To interpolate field ONE component at atime from 
 *    nComp-component(dof) 2D data
 *  @Note: This function is only for regular structured grid of data.
 *  @param[in]  data, n-component(dof) array (intended for 3 component tag), but 
 *    interpolation is done on 1 comp, at a time,
 *  @param[in] comp, nth component, out of degree of freedom
 *  @return value corresponding to comp
 */

OMEGA_H_DEVICE o::Real interpolate2d_wgrid(const o::Reals& data, 
   const o::Reals& gridx, const o::Reals& gridz, const o::Vector<3>& pos,
   const bool cylSymm = true, const o::LO nComp = 1, const o::LO comp = 0,
   bool debug = false) {
  int nx = gridx.size();
  int nz = gridz.size();
  if(debug)
    printInfo("nx %d nz %d comp %d nComp %d \n", nx, nz, comp, nComp);
  if(nx <=1 || nz <= 1)
    return data[comp];
  auto x = pos[0];
  auto z = pos[2]; 
  x = (cylSymm) ? Kokkos::sqrt(x*x+pos[1]*pos[1]) : x;
  auto dx = gridx[1] - gridx[0]; /*(gridx[nx-1] - gridx[0])/nx*/ 
  auto dz = gridz[1] - gridz[0]; /*(gridz[nz-1] - gridz[0])/nz*/ 
  OMEGA_H_CHECK(dx >0 && dz>0);
  o::LO i = Kokkos::floor((x - gridx[0])/dx);
  o::LO j = Kokkos::floor((z - gridz[0])/dz);
  i = (i < 0) ? 0 : i;
  j = (j < 0) ? 0 : j;
  //off limit values not used in calculation
  auto gridXi = (i >= nx) ? gridx[nx-1] : gridx[i]; /*gridx0 + i * dx*/
  auto gridXip1 = (i>= nx-1) ? gridx[nx-1] : gridx[i+1]; /*gridx0 + (i+1) * dx*/
  auto gridZj = (j>=nz) ? gridz[nz-1] : gridz[j]; /*gridz0 + j * dz */
  auto gridZjp1 = (j>=nz-1) ? gridz[nz-1] : gridz[j+1]; /*gridz0 + (j+1) * dz*/
  if(debug)
    printInfo("pos %g %g %g dx,z %g %g i,j %d %d grids %g %g %g %g\n", 
      x,pos[1],z, dx, dz, i, j, gridXi, gridXip1, gridZj, gridZjp1);
  return interpolate2d(data, gridXi, gridXip1, gridZj, gridZjp1, x, z, nx, 
    nz, i, j, dx, dz, 0, false, nComp, comp, debug);
}

OMEGA_H_DEVICE o::Real interpolate2d_wgrid(const o::Reals& data, 
  const o::Reals& gridx, const o::Reals& gridz, const o::Real x,
  const o::Real z, const bool cylSymm = true, const o::LO nComp = 1,
  const o::LO comp = 0, bool debug= false) {
  auto pos = o::zero_vector<3>();
  pos[0] = x;
  pos[1]= 0;
  pos[2] = z;
  return interpolate2d_wgrid(data, gridx, gridz, pos, cylSymm, nComp,
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
      for(int i=0; i<5; ++i) printInfo(" %d %.15e \n", i, gridz[i]);
    o::Real fxyz = 0;
    o::Real dx = gridx[1] - gridx[0];
    o::Real dy = gridy[1] - gridy[0];
    o::Real dz = gridz[1] - gridz[0];
    OMEGA_H_CHECK(!(o::are_close(dx, 0) || o::are_close(dy, 0) || o::are_close(dz, 0)));
    int i = Kokkos::floor((x - gridx[0])/dx);
    int j = Kokkos::floor((y - gridy[0])/dy);
    int k = Kokkos::floor((z - gridz[0])/dz);
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
      printInfo("fx_z0 %.15e fx_z1 %.15e \n", fx_z0, fx_z1);
      printInfo("fxy_z0 %.15e fxy_z1 %.15e fxz0 %.15e fxz1 %.15e fxyz %.15e\n",
        fxy_z0, fxy_z1, fxz0, fxz1, fxyz);
      printInfo("x %.15e y %.15e z %.15e i %d j %d k %d dx %.15e dy %.15e dz %.15e \n", 
        x, y, z, i, j, k, dx, dy, dz);
    }
    fxyz = (ny <= 1) ? fxz0: fxyz;
    fxyz = (nz <= 1) ? fx_z0: fxyz;
    if(debug)
      printInfo("fxy %.15e\n", fxyz);
    return fxyz;
}

OMEGA_H_DEVICE
void interp2dVector_wgrid (const o::Reals& data3, const o::Reals& gridx,
  const o::Reals& gridz, const o::Vector<3> &pos, o::Vector<3> &field,
  const bool cylSymm = false, const bool debug=false) {
  for(int i=0; i<3; ++i)
    field[i] = interpolate2d_wgrid(data3, gridx, gridz, pos, cylSymm, 3, i, debug);
  if(debug)
    printInfo("Field123 are %.15f %.15f %.15f \n", field[0],field[1],field[2]);
  if(gridx.size() > 1 && gridz.size() > 1 && cylSymm) {
    o::Real theta = Kokkos::atan2(pos[1], pos[0]);
    auto field0 = field[0];
    auto field1 = field[1]; 
    field[0] = Kokkos::cos(theta)*field0 - Kokkos::sin(theta)*field1;
    field[1] = Kokkos::sin(theta)*field0 + Kokkos::cos(theta)*field1;
  }
}

OMEGA_H_DEVICE void interp2dVector (const o::Reals& data3, const o::Real gridx0, 
  const o::Real gridz0, const o::Real dx, const o::Real dz, const o::LO nx, 
  const o::LO nz, const o::Vector<3> &pos, o::Vector<3> &field, 
  const bool cylSymm = false, const bool debug=false) {
  for(int i=0; i<3; ++i)
    field[i] = interpolate2d_field(data3, gridx0, gridz0, dx, dz, nx, nz, 
     pos, cylSymm, 3, i, debug);
  if(debug)
    printInfo("Field123 are %.15f %.15f %.15f \n", field[0],field[1],field[2]);

  if(cylSymm) {
    o::Real theta = Kokkos::atan2(pos[1], pos[0]);
    auto field0 = field[0];
    auto field1 = field[1]; 
    field[0] = Kokkos::cos(theta)*field0 - Kokkos::sin(theta)*field1;
    field[1] = Kokkos::sin(theta)*field0 + Kokkos::cos(theta)*field1;
  }
}

OMEGA_H_DEVICE o::Vector<3> centroid_of_triangle(const o::Vector<3>& va,
  const o::Vector<3>& vb, const o::Vector<3>& vc) {
  o::Vector<3> pos;
  for(int i=0; i<3; ++i)
    pos[i] = va[i] + 0.666666667*(vb[i]+ 0.5*(vc[i] - vb[i]) - va[i]);//TODO 2.0/3.0
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
  const o::LOs& mesh2verts,  const o::Reals& coords) {
  o::Vector<3> pos;
  auto tetv2v = o::gather_verts<4>(mesh2verts, elem);
  auto M = o::gather_vectors<4, 3>(coords, tetv2v);
  for(int i=0; i<3; ++i)
    pos[i] = (M[0][i]+M[1][i]+M[2][i]+M[3][i])/4;
  return pos;
}

//2,3 nodes of faces. 0,2,1; 0,1,3; 1,2,3; 2,0,3
OMEGA_H_DEVICE o::LO getFaceMap(const o::LO i) {
  assert(i>=0 && i<8);
  const o::LO fmap[8] = {2,1,1,3,2,3,0,3};
  return fmap[i];
}

OMEGA_H_DEVICE bool isFaceFlipped(const o::LO ei, const o::Few<o::LO, 2>& ev2v, 
                                  const o::Few<o::LO, 3>& facev2v) {
  const o::LO index = (ev2v[0] == facev2v[0]) ? 1 : (ev2v[0] == facev2v[1]) ? 2 : 0;
  return ev2v[1] != facev2v[index];
}

OMEGA_H_DEVICE bool isFaceFlipped(const o::LO fi, const o::Few<o::LO, 3>& fv2v, 
  const o::Few<o::LO, 4>& tetv2v) {
  const o::LO matInd1 = getFaceMap(fi*2);
  const o::LO matInd2 = getFaceMap(fi*2+1);
  const o::LO index = (fv2v[0] == tetv2v[matInd1]) ? 1 : (fv2v[1] == tetv2v[matInd1]) ? 2 : 0;
  return tetv2v[matInd2] != fv2v[index];
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
    printInfo("face_normal_of_tet:getFaceMap:: faceid not found fid %d elmId %d \n",
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

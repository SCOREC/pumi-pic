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
//#include "Omega_h_array.hpp"
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

namespace pumipic{

/*
NOTE: Don't use [] in host for o::Write and o::Read, since [] will be a 
device operator because, in Omega_h host functions are not defined 
when Kokkos/CUDA is enabled.
The above is true for all functions using device-only stuff.
It is safe to define OMEGA_H_DEVICE for functions, unless these are not
used for host only (openmp) compilation.

*/
OMEGA_H_INLINE bool almost_equal(const Omega_h::Real a, const Omega_h::Real b,
    Omega_h::Real tol=EPSILON)
{
  return std::abs(a-b) <= tol;
}

OMEGA_H_INLINE bool almost_equal(const Omega_h::Real *a, const Omega_h::Real *b, 
  Omega_h::LO n=3, Omega_h::Real tol=EPSILON)
{
  for(Omega_h::LO i=0; i<n; ++i)
  {
    if(!almost_equal(a[i],b[i], tol))
    {
      return false;
    }
  }
  return true;
}

OMEGA_H_DEVICE bool all_positive(const Omega_h::Vector<4> a, Omega_h::Real tol=EPSILON)
{
  for(Omega_h::LO i=0; i<a.size(); ++i)
  {
    if(a[i] < -tol) // TODO set default the right tolerance
     return false;
  }
  return true;
}

template <class T> OMEGA_H_DEVICE Omega_h::LO 
min_index(const T a, Omega_h::LO n, Omega_h::Real tol=EPSILON)
{
  Omega_h::LO ind=0;
  Omega_h::Real min = a[0];
  for(Omega_h::LO i=0; i<n-1; ++i)
  {
    if(min > a[i+1])
    {
      min = a[i+1];
      ind = i+1;
    }
  }
  return ind;
}

OMEGA_H_INLINE Omega_h::Real osh_dot(const Omega_h::Vector<3> &a,
   const Omega_h::Vector<3> &b)
{
  return (a[0]*b[0] + a[1]*b[1] + a[2]*b[2]);
}

OMEGA_H_INLINE Omega_h::Real osh_mag(const Omega_h::Vector<3> &v)
{
  return std::sqrt(v[0]*v[0] + v[1]*v[1] + v[2]*v[2]);
}


OMEGA_H_INLINE bool compare_array(const Omega_h::Real *a, const Omega_h::Real *b, 
 const Omega_h::LO n, Omega_h::Real tol=EPSILON)
{
  for(Omega_h::LO i=0; i<n; ++i)
  {
    if(std::abs(a[i]-b[i]) > tol)
    {
      return false;
    }
  }
  return true;
}

OMEGA_H_INLINE bool compare_vector_directions(const Omega_h::Vector<DIM> &va,
     const Omega_h::Vector<DIM> &vb)
{
  for(Omega_h::LO i=0; i<DIM; ++i)
  {
    if((va.data()[i] < 0 && vb.data()[i] > 0) ||
       (va.data()[i] > 0 && vb.data()[i] < 0))
    {
      return false;
    }
  }
  return true;
}

OMEGA_H_INLINE void print_matrix(const Omega_h::Matrix<3, 4> &M)
{
  printf("M0 %.4f, %.4f, %.4f\n", M[0][0], M[0][1], M[0][2]);
  printf("M1 %.4f, %.4f, %.4f\n", M[1][0], M[1][1], M[1][2]);
  printf("M2 %.4f, %.4f, %.4f\n", M[2][0], M[2][1], M[2][2]);
  printf("M3 %.4f, %.4f, %.4f\n", M[3][0], M[3][1], M[3][2]);
}


inline void print_array(const double* a, int n=3, std::string name=" ")
{
  if(name!=" ")
    std::cout << name << ": ";
  for(int i=0; i<n; ++i)
    std::cout << a[i] << ", ";
  std::cout <<"\n";
}

inline void print_osh_vector(const Omega_h::Vector<3> &v, std::string name=" ", 
  bool line_break=true)
{
  std::string str = line_break ? ")\n" : "); ";
  std::cout << name << ": (" << v.data()[0]  << " " << v.data()[1] << " " 
    << v.data()[2] << str;
}

inline void print_data(const Omega_h::Matrix<3, 4> &M, const Omega_h::Vector<3> &dest,
     Omega_h::Write<Omega_h::Real> &bcc)
{
    print_matrix(M);  //include file problem ?
    print_osh_vector(dest, "point");
    print_array(bcc.data(), 4, "BCoords");
}


/** @brief To interpolate field ONE component at atime from 
 *    nComp-component(dof) 2D data
 *  @warning This function is only for regular structured grid of data.
 *  @param[in]  data, n-component(dof) array (intended for 3 component tag), but 
 *    interpolation is done on 1 comp, at a time,
 *  @param[in] comp, nth component, out of degree of freedom
 *  @return value corresponding to comp
 */
OMEGA_H_DEVICE o::Real interpolate2dField(const o::Reals &data, const o::Real gridx0, 
  const o::Real gridz0, const o::Real dx, const o::Real dz, const o::LO nx, 
  const o::LO nz, const o::Vector<3> &pos, const bool cylSymm = false, 
  const o::LO nComp = 1, const o::LO comp = 0) {
  
  o::LO verbose = 0;
  if(nx*nz == 1)
  {
    return data[comp];
  }  

  o::Real fxz = 0;
  o::Real fx_z1 = 0;
  o::Real fx_z2 = 0; 
  o::Real dim1 = pos[0];
  o::Real z = pos[2]; 

  if(cylSymm){
      dim1 = sqrt(pos[0]*pos[0] + pos[1]*pos[1]);
  }

  OMEGA_H_CHECK(dx >0 && dz>0);

  o::LO i = floor((dim1 - gridx0)/dx);
  o::LO j = floor((z - gridz0)/dz);
  
  if (i < 0) i=0;
  if (j < 0) j=0;

  o::Real gridXi = gridx0 + i * dx;
  o::Real gridXip1 = gridx0 + (i+1) * dx;    
  o::Real gridZj = gridz0 + j * dz;
  o::Real gridZjp1 = gridz0 + (j+1) * dz; 

  if (i >=nx-1 && j>=nz-1) {
    fxz = data[(nx-1+(nz-1)*nx)*nComp+comp];
    if(verbose > 3){
      std::cout << "if : \n";
    }
  }
  else if (i >=nx-1) {
    fx_z1 = data[(nx-1+j*nx)*nComp + comp];
    fx_z2 = data[(nx-1+(j+1)*nx)*nComp + comp];
    fxz = ((gridZjp1-z)*fx_z1+(z - gridZj)*fx_z2)/dz;
    if(verbose > 3){
      std::cout << "i >=nx-1 : \n";
    }
  }
  else if (j >=nz-1) {
    fx_z1 = data[(i+(nz-1)*nx)*nComp + comp];
    fx_z2 = data[(i+(nz-1)*nx)*nComp+ comp];
    fxz = ((gridXip1-dim1)*fx_z1+(dim1 - gridXi)*fx_z2)/dx;
    if(verbose > 3){
      std::cout << "j >=nz-1 :\n";
    }
  }
  else {
    fx_z1 = ((gridXip1-dim1)*data[(i+j*nx)*nComp + comp] + 
            (dim1 - gridXi)*data[(i+1+j*nx)*nComp + comp])/dx;
    fx_z2 = ((gridXip1-dim1)*data[(i+(j+1)*nx)*nComp + comp] + 
            (dim1 - gridXi)*data[(i+1+(j+1)*nx)*nComp + comp])/dx; 
    fxz = ((gridZjp1-z)*fx_z1+(z - gridZj)*fx_z2)/dz;
    if(verbose > 3) {
      std::cout << "Else: \n";
    }
  }
  if(verbose > 3) {
    std::cout << " dim1(x):" << dim1 << " pos1:" << pos[1] << " pos2 " << pos[2] 
    << " i: " << i << " j:" << j << " dx: " <<  dx << " gridx0:" << gridx0 
    << " nx " << nx << " nz:" << nz << " fxz " << fxz << "\n";
  }
  return fxz;
}


OMEGA_H_DEVICE void interp2dVector (const o::Reals &data3, o::Real gridx0, 
  o::Real gridz0, o::Real dx, o::Real dz, int nx, int nz,
  const o::Vector<3> &pos, o::Vector<3> &field, const bool cylSymm = false) {

  field[0] = interpolate2dField(data3, gridx0, gridz0, dx, dz, nx, 
    nz, pos, cylSymm, 3, 0);
  field[1] = interpolate2dField(data3, gridx0, gridz0, dx, dz, nx, 
    nz, pos, cylSymm, 3, 1);
  field[2] = interpolate2dField(data3, gridx0, gridz0, dx, dz, nx, 
    nz, pos, cylSymm, 3, 2);
  if(cylSymm) {
    o::Real theta = atan2(pos[1], pos[0]);   
    field[0] = cos(theta)*field[0] - sin(theta)*field[1];
    field[1] = sin(theta)*field[0] + cos(theta)*field[1];
  }
}


OMEGA_H_DEVICE o::Vector<3> find_face_centroid(const o::LO fid, const o::Reals &coords, 
   const o::LOs &face_verts){

  const auto facev = o::gather_verts<3>(face_verts, fid);
  const auto abc = Omega_h::gather_vectors<3, 3>(coords, facev);
  //TODO check if y and z are in required order

  // Mid point of face, as in GITR. 
  o::Vector<3> pos;
  pos[0] = abc[0][0] + 2.0/3.0*(abc[1][0] + 0.5*(abc[2][0] - abc[1][0]) - abc[0][0]);
  pos[1] = abc[0][1] + 2.0/3.0*(abc[1][1] + 0.5*(abc[2][1] - abc[1][1]) - abc[0][1]);
  pos[2] = abc[0][2] + 2.0/3.0*(abc[1][2] + 0.5*(abc[2][2] - abc[1][2]) - abc[0][2]);
  return pos;
}

//2,3 nodes of faces. 0,2,1; 0,1,3; 1,2,3; 2,0,3
OMEGA_H_DEVICE o::LO getfmap(int i) {
  assert(i>=0 && i<8);
  const o::LO fmap[8] = {2,1,1,3,2,3,0,3};
  return fmap[i];
}

OMEGA_H_DEVICE o::Vector<3> get_face_normal(const o::LO fid, const o::LO elmId,
  const o::Reals &coords, const o::LOs& mesh2verts,  const o::LOs &face_verts, 
  const o::LOs &down_r2fs) {

  const auto fv2v = o::gather_verts<3>(face_verts, fid);
  const auto abc = Omega_h::gather_vectors<3, 3>(coords, fv2v);
  const auto tetv2v = o::gather_verts<4>(mesh2verts, elmId);
  //TODO check if face is flipped

  const auto beg_face = elmId *4;
  const auto end_face = beg_face +4;
  o::LO findex = -1;
  o::LO find = 0;
  for(auto iface = beg_face; iface < end_face; ++iface) {
    const auto face_id = down_r2fs[iface];
    if(fid == face_id) {
      findex = find;
      break;
    }
    ++find;
  }

  bool inverse = true;
  o::LO matInd1 = getfmap(findex*2);
  o::LO matInd2 = getfmap(findex*2+1);
  if(fv2v[1] == tetv2v[matInd1] && fv2v[2] == tetv2v[matInd2])
    inverse = false;

  o::Vector<3> a = abc[0];
  o::Vector<3> b = abc[1];
  o::Vector<3> c = abc[2];
  o::Vector<3> fnorm = o::cross(b - a, c - a);
  if(inverse)
    fnorm = -1*fnorm;
  return o::normalize(fnorm);
}


OMEGA_H_DEVICE o::LO elem_of_bdry_face(const o::LO fid, const o::LOs &f2r_ptr,
  const o::LOs &f2r_elem) { 
  //bdry
  OMEGA_H_CHECK(f2r_ptr[fid+1] - f2r_ptr[fid] == 1);
  auto ind = f2r_ptr[fid];
  return f2r_elem[ind];
}


OMEGA_H_DEVICE o::LO angle_between(o::Vector<3> v1, o::Vector<3> v2) {
  auto cos = osh_dot(v1, v2)/ (o::norm(v1) * o::norm(v2));
  return std::acos(cos);
}


OMEGA_H_DEVICE o::Vector<3> centroid_of_tet(o::LO elem, const o::LOs &mesh2verts, 
  const o::Reals &coords) {
  o::Vector<3> pos;
  auto tetv2v = o::gather_verts<4>(mesh2verts, elem);
  auto M = o::gather_vectors<4, 3>(coords, tetv2v);
  pos[0]= (M[0][0]+M[1][0]+M[2][0]+M[3][0])/4;
  pos[1]= (M[0][1]+M[1][1]+M[2][1]+M[3][1])/4;
  pos[2]= (M[0][2]+M[1][2]+M[2][2]+M[3][2])/4;
  return pos;
}

} //namespace
#endif


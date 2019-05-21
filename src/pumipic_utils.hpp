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

namespace o=Omega_h;


namespace pumipic{



/** @brief To interpolate field ONE component at atime from 3D data
 *  @warning This function is only for regular structured grid of data
 *  @param[in]  data, 3component array (intended for 3 component tag), but 
 *  interpolation is done on 1 comp, at a time,
 *  @param[in] comp, component, from degree of freedom
 *  @return value corresponding to comp
 */
OMEGA_H_INLINE o::Real interpolate2dField(const o::Reals &data, const o::LO comp, 
  const o::Real gridx0, const o::Real gridz0, const o::Real dx, const o::Real dz, 
  const o::LO nx, const o::LO nz, const o::Vector<3> &pos, const bool cylSymm = false) {
  
  if(nx*nz == 1)
  {
    return data[comp];
  }

  o::Real x = pos[0];
  o::Real y = pos[1];
  o::Real z = pos[2];   

  o::Real fxz = 0;
  o::Real fx_z1 = 0;
  o::Real fx_z2 = 0; 
  o::Real dim1 = x;

  if(cylSymm)
    dim1 = sqrt(x*x + y*y);

  
  o::LO i = floor((dim1 - gridx0)/dx);
  o::LO j = floor((z - gridz0)/dz);
  
  if (i < 0) i=0;
  if (j < 0) j=0;

  o::Real gridXi = gridx0 + i * dx;
  o::Real gridXip1 = gridx0 + (i+1) * dx;    
  o::Real gridZj = gridz0 + j * dz;
  o::Real gridZjp1 = gridz0 + (j+1) * dz; 

  if (i >=nx-1 && j>=nz-1) {
      fxz = data[(nx-1+(nz-1)*nx)*3+comp];
  }
  else if (i >=nx-1) {
      fx_z1 = data[(nx-1+j*nx)*3+comp];
      fx_z2 = data[(nx-1+(j+1)*nx)*3+comp];
      fxz = ((gridZjp1-z)*fx_z1+(z - gridZj)*fx_z2)/dz;
  }
  else if (j >=nz-1) {
      fx_z1 = data[(i+(nz-1)*nx)*3+comp];
      fx_z2 = data[(i+(nz-1)*nx)*3+comp];
      fxz = ((gridXip1-dim1)*fx_z1+(dim1 - gridXi)*fx_z2)/dx;
      
  }
  else {
    fx_z1 = ((gridXip1-dim1)*data[(i+j*nx)*3+comp] + 
            (dim1 - gridXi)*data[(i+1+j*nx)*3+comp])/dx;
    fx_z2 = ((gridXip1-dim1)*data[(i+(j+1)*nx)*3+comp] + 
            (dim1 - gridXi)*data[(i+1+(j+1)*nx)*3+comp])/dx; 
    fxz = ((gridZjp1-z)*fx_z1+(z - gridZj)*fx_z2)/dz;
  }
  
  return fxz;
}

// Duplicate for host only
// This is using 2D grid and data
inline o::Real interpolate2dFieldHost(const o::HostWrite<o::Real> &data, const o::LO comp, 
  const o::Real gridx0, const o::Real gridz0, const o::Real dx, const o::Real dz, 
  const o::LO nx, const o::LO nz, const o::Vector<3> &pos, const bool cylSymm = false) {
  
  bool verbose = 0;
  if(nx*nz == 1)
  {
    return data[comp];
  }

  o::Real x = pos[0];
  o::Real y = pos[1];
  o::Real z = pos[2];   

  o::Real fxz = 0;
  o::Real fx_z1 = 0;
  o::Real fx_z2 = 0; 
  o::Real dim1 = x;

  if(cylSymm){
      dim1 = sqrt(x*x + y*y);
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
    fxz = data[(nx-1+(nz-1)*nx)*3+comp];
    if(verbose > 3){
      std::cout << "if : " << fxz << "\n";
    }
  }
  else if (i >=nx-1) {
    fx_z1 = data[(nx-1+j*nx)*3+comp];
    fx_z2 = data[(nx-1+(j+1)*nx)*3+comp];
    fxz = ((gridZjp1-z)*fx_z1+(z - gridZj)*fx_z2)/dz;
    if(verbose > 3){
      std::cout << "i >=nx-1 : " << fxz << "\n";
    }
  }
  else if (j >=nz-1) {
    fx_z1 = data[(i+(nz-1)*nx)*3+comp];
    fx_z2 = data[(i+(nz-1)*nx)*3+comp];
    fxz = ((gridXip1-dim1)*fx_z1+(dim1 - gridXi)*fx_z2)/dx;
    if(verbose > 3){
      std::cout << "j >=nz-1 : " << fxz << "\n";
    }
  }
  else {
    fx_z1 = ((gridXip1-dim1)*data[(i+j*nx)*3+comp] + 
            (dim1 - gridXi)*data[(i+1+j*nx)*3+comp])/dx;
    fx_z2 = ((gridXip1-dim1)*data[(i+(j+1)*nx)*3+comp] + 
            (dim1 - gridXi)*data[(i+1+(j+1)*nx)*3+comp])/dx; 
    fxz = ((gridZjp1-z)*fx_z1+(z - gridZj)*fx_z2)/dz;
    if(verbose > 3){
      std::cout << "else: " << fxz << fx_z1 << fx_z2 << 
        gridXip1-dim1 << " " << dim1 - gridXi << 
        " " << (i+j*nx)*3+comp << " " << data[(i+j*nx)*3+comp]
        << " " << data[(i+1+j*nx)*3+comp] << "\n";
    }
  }
  if(verbose > 3){
    std::cout << "fxz " << fxz << " gridXi:" << gridXi <<" gridXip1:"
    <<gridXip1 <<" gridZj:"<< gridZj<<" gridZjp1:"<<gridZjp1
    << " i: " << i << " j:" << j  << " dim1:"<< dim1 << " x:" 
    << x << " y:" << y << " dx:" << dx << " gridx0:" << gridx0 <<"\n";
  }
  return fxz;
}



OMEGA_H_INLINE void interp2dVector (const o::Reals &data3, o::Real gridx0, 
  o::Real gridz0, o::Real dx, o::Real dz, int nx, int nz,
  const o::Vector<3> &pos, o::Vector<3> &field, const bool cylSymm = false) {

  field[0] = interpolate2dField(data3, 0, gridx0, gridz0, dx, dz, nx, nz, pos, cylSymm);
  field[1] = interpolate2dField(data3, 1, gridx0, gridz0, dx, dz, nx, nz, pos, cylSymm);
  field[2] = interpolate2dField(data3, 2, gridx0, gridz0, dx, dz, nx, nz, pos, cylSymm);
  if(cylSymm) {
    o::Real theta = atan2(pos[1], pos[0]);   
    field[0] = cos(theta)*field[0] - sin(theta)*field[1];
    field[1] = sin(theta)*field[0] + cos(theta)*field[1];
  }
}


// Duplicate for host only
inline void interp2dVectorHost (const o::HostWrite<o::Real> &data3, o::Real gridx0, 
  o::Real gridz0, o::Real dx, o::Real dz, int nx, int nz,
  const o::Vector<3> &pos, o::Vector<3> &field, const bool cylSymm = false) {

  field[0] = interpolate2dFieldHost(data3, 0, gridx0, gridz0, dx, dz, nx, nz, pos, cylSymm);
  field[1] = interpolate2dFieldHost(data3, 1, gridx0, gridz0, dx, dz, nx, nz, pos, cylSymm);
  field[2] = interpolate2dFieldHost(data3, 2, gridx0, gridz0, dx, dz, nx, nz, pos, cylSymm);
  if(cylSymm) {
    o::Real theta = atan2(pos[1], pos[0]);   
    field[0] = cos(theta)*field[0] - sin(theta)*field[1];
    field[1] = sin(theta)*field[0] + cos(theta)*field[1];
  }
}


OMEGA_H_INLINE bool almost_equal(const Omega_h::Real a, const Omega_h::Real b,
    Omega_h::Real tol=EPSILON)
{
  return std::abs(a-b) <= tol;
}

OMEGA_H_INLINE bool almost_equal(const Omega_h::Real *a, const Omega_h::Real *b, Omega_h::LO n=3,
    Omega_h::Real tol=EPSILON)
{
  for(Omega_h::LO i=0; i<n; ++i)
  {
    if(!almost_equal(a[i],b[i]))
    {
      return false;
    }
  }
  return true;
}

OMEGA_H_INLINE bool all_positive(const Omega_h::Real *a, Omega_h::LO n=1, Omega_h::Real tol=EPSILON)
{
  for(Omega_h::LO i=0; i<n; ++i)
  {
    if(a[i] < -tol) // TODO set default the right tolerance
     return false;
  }
  return true;
}

OMEGA_H_INLINE Omega_h::LO min_index(Omega_h::Real *a, Omega_h::LO n, Omega_h::Real tol=EPSILON)
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

OMEGA_H_INLINE void osh_cross(const Omega_h::Vector<3> &a,
   const Omega_h::Vector<3> &b, Omega_h::Vector<3> &c)
{
  c[0] = a[1]*b[2] - a[2]*b[1];
  c[1] = - a[0]*b[2] + a[2]*b[0];
  c[2] = a[0]*b[1] - a[1]*b[0];
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
  std::cout << "M0  " << M[0].data()[0] << ", " << M[0].data()[1] << ", " << M[0].data()[2] <<"\n";
  std::cout << "M1  " << M[1].data()[0] << ", " << M[1].data()[1] << ", " << M[1].data()[2] <<"\n";
  std::cout << "M2  " << M[2].data()[0] << ", " << M[2].data()[1] << ", " << M[2].data()[2] <<"\n";
  std::cout << "M3  " << M[3].data()[0] << ", " << M[3].data()[1] << ", " << M[3].data()[2] <<"\n";
}


OMEGA_H_INLINE void print_array(const double* a, int n=3, std::string name=" ")
{
  if(name!=" ")
    std::cout << name << ": ";
  for(int i=0; i<n; ++i)
    std::cout << a[i] << ", ";
  std::cout <<"\n";
}

OMEGA_H_INLINE void print_osh_vector(const Omega_h::Vector<3> &v, std::string name=" ", bool line_break=true)
{
  std::string str = line_break ? ")\n" : "); ";
  std::cout << name << ": (" << v.data()[0]  << " " << v.data()[1] << " " << v.data()[2] << str;
}

OMEGA_H_INLINE void print_data(const Omega_h::Matrix<3, 4> &M, const Omega_h::Vector<3> &dest,
     Omega_h::Write<Omega_h::Real> &bcc)
{
    print_matrix(M);  //include file problem ?
    print_osh_vector(dest, "point");
    print_array(bcc.data(), 4, "BCoords");
}

} //namespace
#endif


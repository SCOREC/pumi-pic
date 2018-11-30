#ifndef GITRM_ADJACENCY_HPP
#define GITRM_ADJACENCY_HPP

#include <iostream>
#include <cmath>
#include <utility>

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


#include "gitrm_utils.hpp"


namespace GITRm{

//OMEGA_H_INLINE needed ?
OMEGA_H_INLINE Omega_h::Real dot(const Omega_h::Vector<3> &a, 
   const Omega_h::Vector<3> &b) OMEGA_H_NOEXCEPT 
{
  return (a[0]*b[0] + a[1]*b[1] + a[2]*b[2]);
}

template <typename T>
OMEGA_H_INLINE bool almost_equal(const T &a, const T &b, 
    Omega_h::Real tol=1e-10) OMEGA_H_NOEXCEPT
{ 
  return std::abs(a-b) <= tol;
}


template <typename T>
OMEGA_H_INLINE bool all_positive(const T* a, Omega_h::LO n=1, 
  Omega_h::Real tol=1e-10) OMEGA_H_NOEXCEPT
{ 
  for(Omega_h::LO i=0; i<n; ++i)
  {
    if(! (almost_equal(a[i], 0.0) || a[i] >tol) ) 
     return false;
  }
  return true;
}


template <typename T>
OMEGA_H_INLINE Omega_h::LO most_negative_index(const T* a, Omega_h::LO n, 
  Omega_h::Real tol=1e-10) OMEGA_H_NOEXCEPT
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


OMEGA_H_INLINE bool find_barycentric( const Omega_h::Vector<3> &a, 
     const Omega_h::Vector<3> &b, const Omega_h::Vector<3> &c, 
     const Omega_h::Vector<3> &d,  const Omega_h::Vector<3> &p, 
     Omega_h::Write<Omega_h::Real> &bc) OMEGA_H_NOEXCEPT 
{
  Omega_h::Real u = dot(p-b, Omega_h::cross(d-b, c-b));
  Omega_h::Real v = dot(p-a, Omega_h::cross(c-a, d-a));
  Omega_h::Real w = dot(p-a, Omega_h::cross(d-a, b-a));
  Omega_h::Real vol = dot(d-a, Omega_h::cross(b-a, c-a)); 
  Omega_h::Real inv_vol = 0.0;
  if(std::abs(vol) > 0)
    inv_vol = 1.0/vol;    
  else
    return 0;
  bc[0] = inv_vol * u;
  bc[1] = inv_vol * v;
  bc[2] = inv_vol * w;         
  bc[3] = 1.0 - bc[0] - bc[1] - bc[2];

  return 1;
}



} //namespace
#endif //define


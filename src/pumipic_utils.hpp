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

namespace pumipic{


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

OMEGA_H_INLINE bool all_positive(const Omega_h::Write<Omega_h::Real> a, Omega_h::LO n=1, Omega_h::Real tol=EPSILON)
{
  for(Omega_h::LO i=0; i<n; ++i)
  {
    if(a[i] < -tol) // TODO set default the right tolerance
     return false;
  }
  return true;
}

OMEGA_H_INLINE Omega_h::LO min_index(const Omega_h::Write<Omega_h::Real> a, Omega_h::LO n, Omega_h::Real tol=EPSILON)
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

void print_matrix(const Omega_h::Matrix<3, 4> &M)
{
  std::cout << "M0  " << M[0].data()[0] << ", " << M[0].data()[1] << ", " << M[0].data()[2] <<"\n";
  std::cout << "M1  " << M[1].data()[0] << ", " << M[1].data()[1] << ", " << M[1].data()[2] <<"\n";
  std::cout << "M2  " << M[2].data()[0] << ", " << M[2].data()[1] << ", " << M[2].data()[2] <<"\n";
  std::cout << "M3  " << M[3].data()[0] << ", " << M[3].data()[1] << ", " << M[3].data()[2] <<"\n";
}


void print_array(const double* a, int n=3, std::string name=" ")
{
  if(name!=" ")
    std::cout << name << ": ";
  for(int i=0; i<n; ++i)
    std::cout << a[i] << ", ";
  std::cout <<"\n";
}

void print_osh_vector(const Omega_h::Vector<3> &v, std::string name=" ", bool line_break=true)
{
  std::string str = line_break ? ")\n" : "); ";
  std::cout << name << ": (" << v.data()[0]  << " " << v.data()[1] << " " << v.data()[2] << str;
}

void print_data(const Omega_h::Matrix<3, 4> &M, const Omega_h::Vector<3> &dest,
     Omega_h::Write<Omega_h::Real> &bcc)
{
    print_matrix(M);  //include file problem ?
    print_osh_vector(dest, "point");
    print_array(bcc.data(), 4, "BCoords");
}
} //namespace
#endif


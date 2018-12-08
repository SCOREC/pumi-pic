#ifndef GITRM_UTILS_HPP
#define GITRM_UTILS_HPP

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

#include "gitrm_adjacency.hpp"



namespace GITRm{

/* //not delcared other files?
OMEGA_H_INLINE Omega_h::Real dot(const Omega_h::Vector<3> &a,
   const Omega_h::Vector<3> &b) OMEGA_H_NOEXCEPT
{
  return (a[0]*b[0] + a[1]*b[1] + a[2]*b[2]);
}
*/

template <typename T>
OMEGA_H_INLINE bool almost_equal(const T &a, const T &b,
    Omega_h::Real tol=1e-10) OMEGA_H_NOEXCEPT
{
  return std::abs(a-b) <= tol;
}

extern template bool almost_equal(const Omega_h::Real &a, const Omega_h::Real &b, Omega_h::Real tol=1e-10);

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

extern template bool all_positive(const double* a, Omega_h::LO n=1,
  Omega_h::Real tol=1e-10);

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

extern template Omega_h::LO most_negative_index(const  Omega_h::LO * a, Omega_h::LO n,
  Omega_h::Real tol);

template <typename T>
OMEGA_H_INLINE bool compare_array(const T &a, const T &b, Omega_h::LO n,
  Omega_h::Real tol=1e-10) OMEGA_H_NOEXCEPT
{
  for(Omega_h::LO i=0; i<n-1; ++i)
  {
    if(! almost_equal(a[i], b[i]))
    {
      return false;
    }
  }
  return true;
}

extern template bool compare_array(const Omega_h::Real &a, const Omega_h::Real &b,
   Omega_h::LO n, Omega_h::Real tol=1e-10);




template <typename T>
void print_array(const T& a, int n, std::string name=" ")
{
  std::cout << name << ": ";
  for(int i=0; i<n; ++i)
    std::cout << a[i] << ", ";
  std::cout <<"\n";
}

extern template void print_array(const double &a, int n, std::string name=" ");

void print_osh_vector(const Omega_h::Vector<3> &v, std::string name=" ", bool line_break=true)
{
  std::string str = line_break ? ")\n" : "); ";
  std::cout << name << ": (" << v.data()[0]  << " " << v.data()[1] << " " << v.data()[2] << str;
}

void print_matrix(const Omega_h::Matrix<3, 4> &M)
{
  std::cout << "M0  " << M[0].data()[0] << ", " << M[0].data()[1] << ", " << M[0].data()[2] <<"\n";
  std::cout << "M1  " << M[1].data()[0] << ", " << M[1].data()[1] << ", " << M[1].data()[2] <<"\n";
  std::cout << "M2  " << M[2].data()[0] << ", " << M[2].data()[1] << ", " << M[2].data()[2] <<"\n";
  std::cout << "M3  " << M[3].data()[0] << ", " << M[3].data()[1] << ", " << M[3].data()[2] <<"\n";
}

} //namespace
#endif


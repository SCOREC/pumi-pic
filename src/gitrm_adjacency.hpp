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
#include "Omega_h_int_scan.hpp" //offset

#include "gitrm_utils.hpp"

//  #define DEBUG 1

namespace GITRm{


OMEGA_H_INLINE Omega_h::Real oh_dot(const Omega_h::Vector<3> &a,
   const Omega_h::Vector<3> &b) OMEGA_H_NOEXCEPT
{
  return (a[0]*b[0] + a[1]*b[1] + a[2]*b[2]);
}


//to make sure the same way face is retrieved for bc and adj check
OMEGA_H_INLINE void get_face_coords(const Omega_h::Matrix<3, 4> &M,
          const Omega_h::LO iface, Omega_h::Few<Omega_h::Vector<3>, 3>& abc)
{

#ifdef DEBUG
    std::cout << Omega_h::simplex_down_template(3, 2, iface, 0) << ","
              << Omega_h::simplex_down_template(3, 2, iface, 1) << ","
              << Omega_h::simplex_down_template(3, 2, iface, 2) << ":Mindex "
              << iface << " :iface\n";
#endif // DEBUG
   //face_vert:0,2,1=a,c,b; 0,1,3; 1,2,3; 2,0,3
    abc[0] = M[Omega_h::simplex_down_template(3, 2, iface, 0)];
    abc[1] = M[Omega_h::simplex_down_template(3, 2, iface, 1)];
    abc[2] = M[Omega_h::simplex_down_template(3, 2, iface, 2)];
    return;
}
//Merge with that in gitrm_utils
void print_arrayp(const double* a, int n=3, std::string name=" ")
{
  if(name!=" ")
    std::cout << name << ": ";
  for(int i=0; i<n; ++i)
    std::cout << a[i] << ", ";
  std::cout <<"\n";
}


//TODO merge with or move to gitrm_utils::compare_array
template <typename T>
OMEGA_H_INLINE bool compare_array(const T *a, const T *b, const Omega_h::LO n,
  Omega_h::Real tol=1e-10) OMEGA_H_NOEXCEPT
{
  for(Omega_h::LO i=0; i<n-1; ++i)
  {
    if(std::abs(a[i]-b[i]) > tol)
    {
      return false;
    }
  }
  return true;
}


OMEGA_H_INLINE void check_face(const Omega_h::Matrix<3, 4> &M,
    const Omega_h::Few<Omega_h::Vector<3>, 3>& face, const Omega_h::LO faceid )
{
    Omega_h::Few<Omega_h::Vector<3>, 3> abc;
    get_face_coords( M, faceid, abc);

#ifdef DEBUG
    print_arrayp(abc[0].data(),3, "a");
    print_arrayp(face[0].data(),3, "face1");
    print_arrayp(abc[1].data(), 3, "b");
    print_arrayp(face[1].data(), 3, "face2");
    print_arrayp(abc[2].data(), 3, "c");
    print_arrayp(face[2].data(), 3, "face3");
#endif
    compare_array(abc[0].data(), face[0].data(), 3); //a
    compare_array(abc[1].data(), face[1].data(), 3); //b
    compare_array(abc[2].data(), face[2].data(), 3); //c
}


OMEGA_H_INLINE bool find_barycentric( const Omega_h::Matrix<3, 4> &M,
     const Omega_h::Vector<3> &p, Omega_h::Write<Omega_h::Real> &bc) OMEGA_H_NOEXCEPT
{
  Omega_h::Real vals[4];
  for(Omega_h::LO iface=0; iface<4; ++iface)
  {
    Omega_h::Vector<3> a, b, c; //TODO pass it in instead of Matrix
    Omega_h::Few<Omega_h::Vector<3>, 3> abc;
    get_face_coords( M, iface, abc);
    a = abc[0];
    b = abc[1];
    c = abc[2];

    auto ab = b - a;
    auto ac = c - a;
    auto ap = p - a;
    vals[iface] = oh_dot(ap, Omega_h::cross(ac, ab)); //ab,ac

#ifdef DEBUG
    std::cout << "vol: " << vals[iface] << " for points_of_this_TET:\n" ;
    //print_matrix({a,b,c});  //FIX this
    //print_osh_vector(p, "p"); //FIX this
    print_arrayp(a.data(),3);print_arrayp(b.data(),3);print_arrayp(c.data(),3);print_arrayp(p.data(),3);
    std::cout << "\n";
#endif // DEBUG
  }

  Omega_h::Real vol6 = oh_dot(M[3]-M[0], Omega_h::cross(M[1]-M[0], M[2]-M[0]));
  Omega_h::Real inv_vol = 0.0;
  if(vol6 > 0)
    inv_vol = 1.0/vol6;
  else
    return 0;
  bc[0] = inv_vol * vals[0];
  bc[1] = inv_vol * vals[1];
  bc[2] = inv_vol * vals[2];
  bc[3] = inv_vol * vals[3];
  //almost_equal(bc[3], 1.0 - bc[0] - bc[1] - bc[2]);
  return 1;
}



OMEGA_H_INLINE bool line_triangle_intx(const Omega_h::Matrix<3, 4> &M,
    const Omega_h::Few<Omega_h::Vector<3>, 3>& face, const Omega_h::LO faceid,
    const Omega_h::Vector<3> origin, const Omega_h::Vector<3> dest, Omega_h::Vector<3> &xpoint )
{
  const Omega_h::Vector<3> dir = dest - origin; //normalize ??

  const Omega_h::Real tol = 1e-10; //macro ?
  Omega_h::Few<Omega_h::Vector<3>, 3> abc;
  get_face_coords( M, faceid, abc);

  const Omega_h::Vector<3> edge1 = abc[1] - abc[0];
  const Omega_h::Vector<3> edge2 = abc[2] - abc[1];
  const Omega_h::Vector<3> dir_x_edge2 = cross(dir, edge2);
  Omega_h::Real det = oh_dot(edge1, dir_x_edge2);
  if(det < tol)  //back facing
    return false;

  if(det > -tol && det < tol) // line parallel to triangle.
    return false;

  const Omega_h::Vector<3> a2orig = origin - abc[0];
  const Omega_h::Real param_u = oh_dot(a2orig, dir_x_edge2);
  if(param_u < 0 || param_u > det)
    return false;

  Omega_h::Vector<3> orig_x_edge1 = cross(a2orig, edge1);
  const Omega_h::Real param_v = oh_dot(dir , orig_x_edge1);
  if(param_v < 0 || param_u + param_v >det)
    return false;

  const Omega_h::Real param = 1/det * oh_dot(edge2, orig_x_edge1); //parameter t of x point
  const Omega_h::Real pvec_len = oh_dot(dest-origin, dest-origin);
  const Omega_h::Real orig2xpt = oh_dot(param*dir, param*dir);

  if(param > tol && orig2xpt <= pvec_len)
  {
     xpoint = origin + param*dir;
     return true;
  }
  else
    return false; //skip this ?

  return false;
}


/*
   TO be deleted
   simplex_down_template(dim=3, face_dim=2, which_face=fn, face_vertn=vn)
   Tet- face:vertexes => 0:0,2,1; 1:0,1,3; 2:1,2,3; 3:2,0,3
*/
OMEGA_H_INLINE bool find_barycentric2( const Omega_h::Vector<3> &a,
     const Omega_h::Vector<3> &b, const Omega_h::Vector<3> &c,
     const Omega_h::Vector<3> &d,  const Omega_h::Vector<3> &p,
     Omega_h::Write<Omega_h::Real> &bc) OMEGA_H_NOEXCEPT
{

  /*
  Few<Vector<3>, 3> sp;
  sp[0] = p[simplex_down_template(3, 2, iface, 0)];
  sp[1] = p[simplex_down_template(3, 2, iface, 1)];
  sp[2] = p[simplex_down_template(3, 2, iface, 2)];
  */
//face_vert:0,2,1=a,c,b; 0,1,3; 1,2,3; 2,0,3
  Omega_h::Real u = oh_dot(p-a, Omega_h::cross(b-a, c-a)); //face_vert:0,2,1=a,c,b
  Omega_h::Real v = oh_dot(p-b, Omega_h::cross(a-b, d-b)); //0,1,3
  Omega_h::Real w = oh_dot(p-c, Omega_h::cross(b-c, d-c)); //1,2,3
  Omega_h::Real x = oh_dot(p-a, Omega_h::cross(d-a, b-a)); //2,0,3

  Omega_h::Real vol = oh_dot(d-a, Omega_h::cross(b-a, c-a));
  Omega_h::Real inv_vol = 0.0;
  if(vol > 0)
    inv_vol = 1.0/vol;
  else
    return 0;
  bc[0] = inv_vol * u;
  bc[1] = inv_vol * v;
  bc[2] = inv_vol * w;
  bc[3] = inv_vol * x;

  return 1;
}



} //namespace
#ifdef DEBUG
#undef DEBUG
#endif // DEBUG
#endif //define


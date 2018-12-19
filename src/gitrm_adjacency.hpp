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

//#define DEBUG 1

namespace GITRm
{

const Omega_h::LO DIM = 3; // mesh DIM. Other DIMs will cause error
const Omega_h::LO FDIM = 2; //mesh face DIM


Omega_h::Real osh_dot(const Omega_h::Vector<3> &a,
   const Omega_h::Vector<3> &b) OMEGA_H_NOEXCEPT
{
  return (a[0]*b[0] + a[1]*b[1] + a[2]*b[2]);
}

/*
   see description: Omega_h_simplex.hpp, Omega_h_refine_topology.hpp line 26
   face_vert:0,2,1; 0,1,3; 1,2,3; 2,0,3.
   corresp. opp. vertexes: 3,2,0,1, by simplex_opposite_template(DIM, FDIM, iface)
   side note: r3d.cpp line 528: 3,2,1; 0,2,3; 0,3,1; 0,1,2 .Vertexes opp.:0,1,2,3
              3
            / | \
          /   |   \
         0----|----2
          \   |   /
            \ | /
              1
*/
//retrieve face for bcc and adj the same way
OMEGA_H_INLINE void get_face_coords(const Omega_h::Matrix<DIM, 4> &M,
          const Omega_h::LO iface, Omega_h::Few<Omega_h::Vector<DIM>, 3> &abc)
{
   //face_vert:0,2,1; 0,1,3; 1,2,3; 2,0,3
    abc[0] = M[Omega_h::simplex_down_template(DIM, FDIM, iface, 0)];
    abc[1] = M[Omega_h::simplex_down_template(DIM, FDIM, iface, 1)];
    abc[2] = M[Omega_h::simplex_down_template(DIM, FDIM, iface, 2)];

#ifdef DEBUG
    std::cout << "Mat_index " << abc[0] << ", " << abc[1] << ", "
              << abc[2] << " iface:" << iface << "\n";
#endif // DEBUG
}


OMEGA_H_INLINE void get_edge_coords(const Omega_h::Few<Omega_h::Vector<DIM>, 3> &abc,
          const Omega_h::LO iedge, Omega_h::Few<Omega_h::Vector<DIM>, 2> &ab)
{
   //edge_vert:0,1; 1,2; 2,0
    ab[0] = abc[Omega_h::simplex_down_template(FDIM, 1, iedge, 0)];
    ab[1] = abc[Omega_h::simplex_down_template(FDIM, 1, iedge, 1)];
#ifdef DEBUG
    std::cout << "abc_index " << ab[0] << ", " << ab[1]
              << " iedge:" << iedge << "\n";
#endif // DEBUG
}

//Merge with that in gitrm_utils
void print_array(const double* a, int n=3, std::string name=" ")
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

OMEGA_H_INLINE bool compare_vector_directions(const Omega_h::Vector<DIM> &va,
     const Omega_h::Vector<DIM> &vb) OMEGA_H_NOEXCEPT
{
  for(Omega_h::LO i=0; i<DIM; ++i)
  {
    //std::cout << "MATCH: "<< va.data()[i]  << " " << vb.data()[i] << "\n";

    if((va.data()[i] < 0 && vb.data()[i] > 0) ||
       (va.data()[i] > 0 && vb.data()[i] < 0))
    {
      return false;
    }
  }
  return true;
}

OMEGA_H_INLINE void check_face(const Omega_h::Matrix<DIM, 4> &M,
    const Omega_h::Few<Omega_h::Vector<DIM>, 3>& face, const Omega_h::LO faceid )
{
    Omega_h::Few<Omega_h::Vector<DIM>, 3> abc;
    get_face_coords( M, faceid, abc);

#ifdef DEBUG
    print_array(abc[0].data(),3, "a");
    print_array(face[0].data(),3, "face1");
    print_array(abc[1].data(), 3, "b");
    print_array(face[1].data(), 3, "face2");
    print_array(abc[2].data(), 3, "c");
    print_array(face[2].data(), 3, "face3");
#endif
    compare_array(abc[0].data(), face[0].data(), DIM); //a
    compare_array(abc[1].data(), face[1].data(), DIM); //b
    compare_array(abc[2].data(), face[2].data(), DIM); //c
}

// BC coords are not in order of its corresp. opp. vertexes. Bccoord of tet(iface, xpoint)
//  corresp. to vertex obtained from simplex_opposite_template(DIM, 2, iface)
OMEGA_H_INLINE bool find_barycentric_tet( const Omega_h::Matrix<DIM, 4> &Mat,
     const Omega_h::Vector<DIM> &ptp, Omega_h::Write<Omega_h::Real> &bcc) OMEGA_H_NOEXCEPT
{
  Omega_h::Real vals[4];
  Omega_h::Few<Omega_h::Vector<DIM>, 3> abc;
  for(Omega_h::LO iface=0; iface<4; ++iface)
  {
    get_face_coords(Mat, iface, abc);

    auto vab = abc[1] - abc[0]; //b - a;
    auto vac = abc[2] - abc[0]; //c - a;
    auto vap = ptp - abc[0]; // p - a;
    //associate opposite vertex
    vals[iface] = osh_dot(vap, Omega_h::cross(vac, vab)); //ac, ab NOTE

#ifdef DEBUG
    std::cout << "vol: " << vals[iface] << " for points_of_this_TET:\n" ;
    //print_matrix({abc[0],abc[1],abc[2]});  //FIX this
    //print_osh_vector(ptp, "ptp"); //FIX this
    print_array(abc[0].data(),3);
    print_array(abc[1].data(),3);
    print_array(abc[2].data(),3);
    print_array(ptp.data(),3);
    std::cout << "\n";
#endif // DEBUG
  }
  get_face_coords(Mat, 0, abc); // bottom face, iface=0
  OMEGA_H_CHECK(3 == Omega_h::simplex_opposite_template(DIM, FDIM, 0)); //iface=0
  Omega_h::Vector<DIM> cross_ac_ab = Omega_h::cross(abc[2]-abc[0], abc[1]-abc[0]); //NOTE
  Omega_h::Real vol6 = osh_dot(Mat[3]-Mat[0], cross_ac_ab);
  Omega_h::Real inv_vol = 0.0;
  if(vol6 > 0) // TODO include delta
    inv_vol = 1.0/vol6;
  else
    return 0;

  bcc[0] = inv_vol * vals[0]; //cooresp. to vtx != 0, but opp. to face 0.
  bcc[1] = inv_vol * vals[1];
  bcc[2] = inv_vol * vals[2];
  bcc[3] = inv_vol * vals[3]; // 1-others

  return 1; //success
}

//TODO for debugging
OMEGA_H_INLINE Omega_h::Real find_dist_to_surface( const Omega_h::Matrix<DIM, 4> &Mat,
     const Omega_h::Vector<DIM> &ptp, const Omega_h::Write<Omega_h::Real> &bcc) OMEGA_H_NOEXCEPT
{
  Omega_h::Real dist = 0;


  return dist;
}




OMEGA_H_INLINE bool line_triangle_intx_moller(const Omega_h::Matrix<3, 4> &M,
    const Omega_h::Few<Omega_h::Vector<3>, 3> &face, const Omega_h::LO face_id,
    const Omega_h::Vector<3> &origin, const Omega_h::Vector<3> &dest,
    Omega_h::Vector<3> &xpoint) OMEGA_H_NOEXCEPT
{
  const Omega_h::Vector<3> dir = dest - origin; //Omega_h::normalize(dest - origin) ??

  const Omega_h::Real tol = 1e-10; //macro ?
  Omega_h::Few<Omega_h::Vector<3>, 3> abc;
  get_face_coords( M, face_id, abc);

  print_array(origin.data(), 3, "orig");
  print_array(dest.data(), 3, "dest");
  print_array(dir.data(), 3, "dir");
  print_array(abc[0].data(),3,"facea");print_array(abc[1].data(),3,"faceb");print_array(abc[2].data(),3,"facec");


  const Omega_h::Vector<3> edge1 = abc[1] - abc[0];
  const Omega_h::Vector<3> edge2 = abc[2] - abc[1];
  const Omega_h::Vector<3> dir_x_edge2 = cross(dir, edge2);
  const Omega_h::Real det = osh_dot(edge1, dir_x_edge2);
  if(det < tol)  //back facing
    return false;

  std::cout << "Front facing..\n";

  if(det > -tol && det < tol) // line parallel to triangle.
    return false;

  std::cout << "NOT ||l to triangle..\n";

  const Omega_h::Vector<3> a2orig = origin - abc[0];
  const Omega_h::Real param_u = osh_dot(a2orig, dir_x_edge2);
  if(param_u < 0 || param_u > det)
    return false;

  std::cout << "u range is valid..\n";

  Omega_h::Vector<3> orig_x_edge1 = cross(a2orig, edge1);
  const Omega_h::Real param_v = osh_dot(dir , orig_x_edge1);
  if(param_v < 0 || param_u + param_v >det)
    return false;

   std::cout << "v range is valid..\n";

  const Omega_h::Real param = 1/det * osh_dot(edge2, orig_x_edge1); //parameter t of x point
  const Omega_h::Real pvec_len = osh_dot(dest-origin, dest-origin);
  const Omega_h::Real orig2xpt = osh_dot(param*dir, param*dir);

  if(param > tol && orig2xpt <= pvec_len)
  {
     xpoint = origin + param*dir;
     return true;
  }
  std::cout << "FALSE: orig2xpt " <<  orig2xpt << " pvec_len " << pvec_len << " param  " << param << "\n";

  return false;
}


// BC coords are not in order of its corresp. vertexes. Bccoord of triangle (iedge, xpoint)
//  corresp. to vertex obtained from simplex_opposite_template(FDIM, 1, iedge)
OMEGA_H_INLINE bool find_barycentric_tri_simple(const Omega_h::Few<Omega_h::Vector<DIM>, 3> &abc,
     const Omega_h::Vector<3> &xpoint, Omega_h::Write<Omega_h::Real> &bc) OMEGA_H_NOEXCEPT
{
  Omega_h::Vector<DIM> a = abc[0];
  Omega_h::Vector<DIM> b = abc[1];
  Omega_h::Vector<DIM> c = abc[2];
  Omega_h::Vector<DIM> cross = 1/2.0 * Omega_h::cross(b-a, c-a); //NOTE order
  Omega_h::Vector<DIM> norm = Omega_h::normalize(cross);
  Omega_h::Real area = osh_dot(norm, cross);
  if(std::abs(area) < 1e-10)
    return 0;
  Omega_h::Few<Omega_h::Vector<DIM>, 2> ab;

  for(Omega_h::LO iedge = 0; iedge <3; ++iedge)
  {
    get_edge_coords(abc, iedge, ab);
    bc[iedge] = 1/2.0 * osh_dot(norm, Omega_h::cross(ab[1]-ab[0], xpoint-ab[0]));
    bc[iedge] = bc[iedge]/area;
  }
  OMEGA_H_CHECK(std::abs(1.0 - bc[0] - bc[1] - bc[2]) <= 1e-10);

  return 1;
}
//#define DEBUG 1
//en.wikipedia.org/wiki/Line%E2%80%93plane_intersection
OMEGA_H_INLINE bool line_triangle_intx(const Omega_h::Few<Omega_h::Vector<DIM>, 3> &abc,
  const Omega_h::Vector<DIM> &origin, const Omega_h::Vector<DIM> &dest,
  Omega_h::Vector<DIM> &xpoint )OMEGA_H_NOEXCEPT
{
  const Omega_h::Vector<DIM> line = dest - origin; //Iab

  const Omega_h::Real tol = 1e-10; // TODO: global

  const Omega_h::Vector<DIM> edge1 = abc[1] - abc[0]; //P01
  const Omega_h::Vector<DIM> edge2 = abc[2] - abc[0]; //P02
  Omega_h::Vector<DIM> norm = Omega_h::cross(edge1, edge2);
  Omega_h::Vector<DIM> a2origin = origin - abc[0];
  Omega_h::Real det = -1.0 * osh_dot(line, norm);

#ifdef DEBUG
  //For testing. Only find intersection point. Wrong for orig {1,0.2,0.2} to dest{12,-0.2,0.2}
  Omega_h::Real planed = osh_dot(norm, abc[0]);
  Omega_h::Real par_t = (planed - osh_dot(norm, origin)) / osh_dot(norm, line);
  if (par_t >= 0.0 && par_t <= 1.0) {
    Omega_h::Vector<DIM> q = origin + par_t * line;
    print_array(q.data(), 3, ">>Tcalc: xpt ");
  }
  std::cout << ">> det:" << det<< "\n";
#endif // DEBUG

  if(std::abs(det) < tol)
    return 0;

  Omega_h::Real tnumer = osh_dot(norm, a2origin);
  Omega_h::Real paramt = tnumer/det;

  Omega_h::Vector<DIM> cross_e2_mline = -1.0 * Omega_h::cross(edge2, line);

  Omega_h::Real unumer = osh_dot(cross_e2_mline, a2origin);
  Omega_h::Real bcu = unumer/det;

  Omega_h::Vector<DIM> cross_mline_e1 = -1.0 * Omega_h::cross(line, edge1);

  Omega_h::Real vnumer = osh_dot(cross_mline_e1, a2origin);
  Omega_h::Real bcv = vnumer/det;

#ifdef DEBUG
  std::cout << ">>tuv u+v  " << paramt << " " << bcu << " " << bcv << " " << bcu+bcv << "\n";
#endif // DEBUG
  if(paramt < tol || paramt >1.0) // 0= self intersection with originating face
    return false;

  if(bcu < 0 || bcv < 0 || bcu+bcv > 1.0)
    return false;

  if(! compare_vector_directions(norm, line))
  {
    std::cout << "*** Line OPOSITE to element face normal \n";
  }

  xpoint = origin + paramt*line;

//#ifdef DEBUG
  Omega_h::Write<Omega_h::Real> bc{0,0,0};
  find_barycentric_tri_simple(abc, xpoint, bc);

  print_array(origin.data(), 3, "orig");
  print_array(dest.data(), 3, "dest");
  print_array(abc[0].data(), 3, " a");
  print_array(abc[1].data(), 3, " b");
  print_array(abc[2].data(), 3, " c");

  print_array(norm.data(),3,"norm");
  print_array(line.data(),3,"line");
  print_array(xpoint.data(), 3, "xpoint");
  print_array(bc.data(), 3, "bc");
//#endif // DEBUG
  return true;
}





} //namespace
#ifdef DEBUG
#undef DEBUG
#endif // DEBUG
#endif //define


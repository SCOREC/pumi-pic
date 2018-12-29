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

const static Omega_h::Real EPSILON = 1e-6;

//#define DEBUG 1

namespace GITRm
{

const Omega_h::LO DIM = 3; // mesh DIM. Other DIMs will cause error
const Omega_h::LO FDIM = 2; //mesh face DIM

void print_matrix(const Omega_h::Matrix<3, 4> &M)
{
  std::cout << "M0  " << M[0].data()[0] << ", " << M[0].data()[1] << ", " << M[0].data()[2] <<"\n";
  std::cout << "M1  " << M[1].data()[0] << ", " << M[1].data()[1] << ", " << M[1].data()[2] <<"\n";
  std::cout << "M2  " << M[2].data()[0] << ", " << M[2].data()[1] << ", " << M[2].data()[2] <<"\n";
  std::cout << "M3  " << M[3].data()[0] << ", " << M[3].data()[1] << ", " << M[3].data()[2] <<"\n";
}

void print_osh_vector(const Omega_h::Vector<3> &v, std::string name=" ", bool line_break=true)
{
  std::string str = line_break ? ")\n" : "); ";
  std::cout << name << ": (" << v.data()[0]  << " " << v.data()[1] << " " << v.data()[2] << str;
}

bool almost_equal(const Omega_h::Real a, const Omega_h::Real b,
    Omega_h::Real tol=1e-6)
{
  return std::abs(a-b) <= tol;
}

bool almost_equal(const Omega_h::Real *a, const Omega_h::Real *b, Omega_h::LO n=3,
    Omega_h::Real tol=1e-6)
{
  for(Omega_h::LO i=0; i<n; ++i)
  {
    if(!almost_equal(a[i],b[i]))
    {
      std::cout <<i << " " << a[i] << " " << b[i] << " "<< std::abs(a[i]-b[i]) << " False \n";
      return false;
    }
  }
  return true;
}

bool all_positive(const Omega_h::Real *a, Omega_h::LO n=1, Omega_h::Real tol=1e-6)
{
  for(Omega_h::LO i=0; i<n; ++i)
  {
    if(a[i] < 0.0) //tol) // TODO optimize. if val=0, then <=0 detects on both elements; <0 neither
     return false;
  }
  return true;
}

Omega_h::LO min_index(Omega_h::Real *a, Omega_h::LO n, Omega_h::Real tol=1e-6)
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


Omega_h::Real osh_dot(const Omega_h::Vector<3> &a,
   const Omega_h::Vector<3> &b)// OMEGA_H_NOEXCEPT
{
  return (a[0]*b[0] + a[1]*b[1] + a[2]*b[2]);
}

/*
   see description: Omega_h_simplex.hpp, Omega_h_refine_topology.hpp line 26
   face_vert:0,2,1; 0,1,3; 1,2,3; 2,0,3.
   corresp. opp. vertexes: 3,2,0,1, by simplex_opposite_template(DIM, FDIM, iface, i) ?
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
    print_osh_vector(*abc.data(), "Mat_index ");
#endif // DEBUG
}


OMEGA_H_INLINE void get_edge_coords(const Omega_h::Few<Omega_h::Vector<DIM>, 3> &abc,
          const Omega_h::LO iedge, Omega_h::Few<Omega_h::Vector<DIM>, 2> &ab)
{
   //edge_vert:0,1; 1,2; 2,0
    ab[0] = abc[Omega_h::simplex_down_template(FDIM, 1, iedge, 0)];
    ab[1] = abc[Omega_h::simplex_down_template(FDIM, 1, iedge, 1)];
#ifdef DEBUG
    std::cout << "abc_index " << ab[0].data() << ", " << ab[1].data()
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
  Omega_h::Real tol=1e-6)
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
    OMEGA_H_CHECK(true == compare_array(abc[0].data(), face[0].data(), DIM)); //a
    OMEGA_H_CHECK(true == compare_array(abc[1].data(), face[1].data(), DIM)); //b
    OMEGA_H_CHECK(true == compare_array(abc[2].data(), face[2].data(), DIM)); //c
}

// BC coords are not in order of its corresp. opp. vertexes. Bccoord of tet(iface, xpoint)
//  corresp. to vertex obtained from simplex_opposite_template(DIM, 2, iface) ?
//TODO Warning: Check opposite_template use in this before using
OMEGA_H_INLINE bool find_barycentric_tet( const Omega_h::Matrix<DIM, 4> &Mat,
     const Omega_h::Vector<DIM> &pos, Omega_h::Write<Omega_h::Real> &bcc)
{
  for(Omega_h::LO i=0; i<3; ++i) bcc[i] = -1;

  Omega_h::Real vals[4];
  Omega_h::Few<Omega_h::Vector<DIM>, 3> abc;
  for(Omega_h::LO iface=0; iface<4; ++iface)
  {
    get_face_coords(Mat, iface, abc);

    auto vab = abc[1] - abc[0]; //b - a;
    auto vac = abc[2] - abc[0]; //c - a;
    auto vap = pos - abc[0]; // p - a;
    //associate opposite vertex
    vals[iface] = osh_dot(vap, Omega_h::cross(vac, vab)); //ac, ab NOTE

#ifdef DEBUG
    std::cout << "vol: " << vals[iface] << " for points_of_this_TET:\n" ;
    //print_matrix({abc[0],abc[1],abc[2]});  //FIX this
    //print_osh_vector(pos, "pos"); //FIX this
    print_array(abc[0].data(),3);
    print_array(abc[1].data(),3);
    print_array(abc[2].data(),3);
    print_array(pos.data(),3);
    std::cout << "\n";
#endif // DEBUG
  }
  get_face_coords(Mat, 0, abc); // bottom face, iface=0
  OMEGA_H_CHECK(3 == Omega_h::simplex_opposite_template(DIM, FDIM, 0)); //iface=0 ?
  Omega_h::Vector<DIM> cross_ac_ab = Omega_h::cross(abc[2]-abc[0], abc[1]-abc[0]); //NOTE
  Omega_h::Real vol6 = osh_dot(Mat[3]-Mat[0], cross_ac_ab);
  Omega_h::Real inv_vol = 0.0;
  if(vol6 > 1e-10) // TODO include delta
    inv_vol = 1.0/vol6;
  else
    return 0;

  bcc[0] = inv_vol * vals[0]; //cooresp. to vtx != 0, but opp. to face 0.
  bcc[1] = inv_vol * vals[1];
  bcc[2] = inv_vol * vals[2];
  bcc[3] = inv_vol * vals[3]; // 1-others

  return 1; //success
}


OMEGA_H_INLINE bool line_triangle_intx_moller(const Omega_h::Matrix<3, 4> &M,
    const Omega_h::Few<Omega_h::Vector<3>, 3> &face, const Omega_h::LO face_id,
    const Omega_h::Vector<3> &origin, const Omega_h::Vector<3> &dest,
    Omega_h::Vector<3> &xpoint)
{
  const Omega_h::Vector<3> dir = dest - origin; //Omega_h::normalize(dest - origin) ??

  const Omega_h::Real tol = 1e-10; //macro ?
  Omega_h::Few<Omega_h::Vector<3>, 3> abc;
  get_face_coords( M, face_id, abc);

  print_array(origin.data(), 3, "orig");
  print_array(dest.data(), 3, "dest");
  print_array(dir.data(), 3, "dir");
  print_array(abc[0].data(),3,"facea");print_array(abc[1].data(),3,"faceb");print_array(abc[2].data(),3,"facec");


  const Omega_h::Vector<3> edge0 = abc[1] - abc[0];
  const Omega_h::Vector<3> edge1 = abc[2] - abc[1];
  const Omega_h::Vector<3> dir_x_edge1 = cross(dir, edge1);
  const Omega_h::Real det = osh_dot(edge0, dir_x_edge1);
  if(det < tol)  //back facing
    return false;

  std::cout << "Front facing..\n";

  if(det > -tol && det < tol) // line parallel to triangle.
    return false;

  std::cout << "NOT ||l to triangle..\n";

  const Omega_h::Vector<3> a2orig = origin - abc[0];
  const Omega_h::Real param_u = osh_dot(a2orig, dir_x_edge1);
  if(param_u < 0 || param_u > det)
    return false;

  std::cout << "u range is valid..\n";

  Omega_h::Vector<3> orig_x_edge0 = cross(a2orig, edge0);
  const Omega_h::Real param_v = osh_dot(dir , orig_x_edge0);
  if(param_v < 0 || param_u + param_v >det)
    return false;

   std::cout << "v range is valid..\n";

  const Omega_h::Real param = 1/det * osh_dot(edge1, orig_x_edge0); //parameter t of x point
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

//TODO Fix, this defined in utils not found here
Omega_h::LO min_index_(const Omega_h::Reals &a, Omega_h::LO n, Omega_h::Real tol=1e-6)
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

// BC coords are not in order of its corresp. vertexes. Bccoord of triangle (iedge, xpoint)
// corresp. to vertex obtained from simplex_opposite_template(FDIM, 1, iedge) ?
OMEGA_H_INLINE bool find_barycentric_tri_simple(const Omega_h::Few<Omega_h::Vector<DIM>, 3> &abc,
     const Omega_h::Vector<3> &xpoint, Omega_h::Write<Omega_h::Real> &bc)
{
  Omega_h::Vector<DIM> a = abc[0];
  Omega_h::Vector<DIM> b = abc[1];
  Omega_h::Vector<DIM> c = abc[2];
  Omega_h::Vector<DIM> cross = 1/2.0 * Omega_h::cross(b-a, c-a); //NOTE order
  Omega_h::Vector<DIM> norm = Omega_h::normalize(cross);
  Omega_h::Real area = osh_dot(norm, cross);
  if(std::abs(area) < 1e-6)
    return 0;

  bc[0] = 1/area * 1/2.0 * osh_dot(norm, Omega_h::cross(b-a, xpoint-a));
  bc[1] = 1/area * 1/2.0 * osh_dot(norm, Omega_h::cross(c-b, xpoint-b));
  bc[2] = 1/area * 1/2.0 * osh_dot(norm, Omega_h::cross(xpoint-a, c-a));

  OMEGA_H_CHECK(std::abs(1.0 - bc[0] - bc[1] - bc[2]) <= 1e-6);

  return 1;
}


/*
 If (p0 - l0).n =0, then the line is ||l to plane, either touching or away in ||l.
 If the particle touches(overlaps) with boundary then it should be a collision(exit),
 since EField at boundary shouldn't be used for driving particles.
 The intolerence should be <10^-5, and ideally <= 10^-6 (resolution near SOL ~microns).
       v2
    e2/ \ e1
     /___\
    v0 e0 v1
  Edges: 0,1;1,2;2,0  Omega_h_simplex

*/
//#define DEBUG 1
//TODO use tolerence
OMEGA_H_INLINE bool line_triangle_intx_simple(const Omega_h::Few<Omega_h::Vector<DIM>, 3> &abc,
    const Omega_h::Vector<DIM> &origin, const Omega_h::Vector<DIM> &dest,
    Omega_h::Vector<DIM> &xpoint, Omega_h::LO *edge=nullptr)
{
  *edge = -1;
  xpoint = {0, 0, 0};

  //Boundary exclusion. Don't set it globally and change randomnly.
  const Omega_h::Real bound_intol = 1e-10; //TODO optimum value ?

  bool found = false;
  const Omega_h::Vector<DIM> line = dest - origin;
  const Omega_h::Vector<DIM> edge0 = abc[1] - abc[0];
  const Omega_h::Vector<DIM> edge1 = abc[2] - abc[0];
  const Omega_h::Vector<DIM> normv = Omega_h::cross(edge0, edge1);
  const Omega_h::Vector<DIM> snorm_unit = Omega_h::normalize(normv);
  const Omega_h::Real dist2plane = osh_dot(abc[0] - origin, snorm_unit);
  const Omega_h::Real proj_lined =  osh_dot(line, snorm_unit);

  if(std::abs(proj_lined >0))
  {
    const Omega_h::Real par_t = dist2plane/proj_lined;

    if (par_t > bound_intol && par_t <= 1.0) //TODO test tol value
    {
      xpoint = origin + par_t * line;
      Omega_h::Write<Omega_h::Real> bcc{0, 0, 0};
      if(find_barycentric_tri_simple(abc, xpoint, bcc))
      {
        if(bcc[0] < 0 || bcc[2] < 0 || bcc[0]+bcc[2] > 1.0)
          *edge = min_index_(bcc, 3, 1e-6);
        else if(compare_vector_directions(normv, line))
          found = true;
      }
#ifdef DEBUG
      print_array(bcc.data(), 3, "BCC");
#endif // DEBUG
    }
#ifdef DEBUG
    else if(par_t < bound_intol) // dist2plane ~0. Line contained in plane, no intersection?
    {
      std::cout << "Self-intersection of ptcl origin with plane at origin. t= " << par_t << "\n";
      print_osh_vector(line, "line", false);
      print_osh_vector(normv, "normv", false);
      print_osh_vector(origin, "origin", false);
      print_osh_vector(dest, "dest", false);

      std::cout << " dist2plane " <<  dist2plane << " proj_lined " << proj_lined << "\n";
    }
#endif // DEBUG
  }
  else
  {
#ifdef DEBUG
    std::cout << "Line and plane are parallel \n";
#endif // DEBUG
  }
  return found;
}


//TODO pass base Adj and use its members a2ab ab2b, instead of passing the member ref.
OMEGA_H_INLINE bool wall_collision_search(const Omega_h::Few<Omega_h::Vector<DIM>, 3> &face, const Omega_h::LO fid,
  const Omega_h::Vector<DIM> &origin, const Omega_h::Vector<DIM> &dest, const Omega_h::LOs &down_f2es,
  const Omega_h::LOs &up_e2f_edges, const Omega_h::LOs &up_e2f_faces, Omega_h::Read<Omega_h::I8> &side_is_exposed,
  const Omega_h::LOs &up_f2r_faces, const Omega_h::LOs &up_f2r_reg, Omega_h::LO &adj_face_id, Omega_h::LO &adj_elem_id,
  Omega_h::Vector<DIM> &xpoint)
{
  Omega_h::LO edge = -1;
  bool detected = line_triangle_intx_simple(face, origin, dest, xpoint, &edge);
  if(detected) return true;
  //one check on a face, then follow min_entry edge to next exposed face
  auto edge_id = down_f2es[fid*3 + edge];

  auto fstart = up_e2f_edges[edge_id];
  auto fend = up_e2f_edges[edge_id+1];
  for(Omega_h::LO fp=fstart; fp<fend; ++fp)
  {
    auto afid = up_e2f_faces[fp];
    if(side_is_exposed[afid] && afid != fid)
    {
      adj_face_id = afid;
      break;
    }
  }
  //find element id of adj_face_id. only 1 region for exposed face
  if(up_f2r_faces[adj_face_id+1] == 1+up_f2r_faces[adj_face_id])
  {
    adj_elem_id = up_f2r_reg[up_f2r_faces[adj_face_id]];
    //elem_ids[iptcl] = adj_elem;
    //part_flags.data()[iptcl] = 2; //collision
  }
  else
  {
    return false; //Error, not an exposed face
  }
}



// en.wikipedia.org/wiki/Line%E2%80%93plane_intersection
OMEGA_H_INLINE bool line_triangle_intx_combined(const Omega_h::Few<Omega_h::Vector<DIM>, 3> &abc,
  const Omega_h::Vector<DIM> &origin, const Omega_h::Vector<DIM> &dest,
  Omega_h::Vector<DIM> &xpoint, Omega_h::LO &edge)
{
  bool status = false;
  edge = -1;
  xpoint = {0, 0, 0};
  const Omega_h::Vector<DIM> line = dest - origin;

  //Boundary exclusion. Don't set it globally and change randomnly.
  const Omega_h::Real bound_intol = 1e-10; //TODO optimum value ?

  Omega_h::LO edgev[6];
  //Needed ?
  for(Omega_h::LO ie=0; ie<3; ++ie)
  {
    //down template has last edge 2_0
    edgev[2*ie] = Omega_h::simplex_down_template(FDIM, 1, ie, 0); //1=edge
    edgev[2*ie+1] = Omega_h::simplex_down_template(FDIM, 1, ie, 1);
#ifdef DEBUG
    std::cout << "...\n LINE_TRI_edgev "  << edgev[2*ie] << " " <<  edgev[2*ie+1] << "\n";
#endif // DEBUG
  }

  const Omega_h::Vector<DIM> edge0 = abc[1] - abc[0];//abc[edgev[1]] - abc[edgev[0]]; //v0=a;v1=b
  const Omega_h::Vector<DIM> edge2 = abc[2] - abc[0];//abc[edgev[4]] - abc[edgev[5]]; //v5=a;v4=c
  Omega_h::Vector<DIM> normv = Omega_h::cross(edge0, edge2);
  Omega_h::Vector<DIM> a2origin = origin - abc[0];// abc[edgev[0]];
  // line.normv=0 => line is ||l to plane, either touching or away in ||l or perp dir.
  Omega_h::Real det = -1.0 * osh_dot(line, normv);

  if(std::abs(det) < bound_intol)
  {
    return false;
  }

  //If (p0 - l0).n =0, line is ||l to plane, either touching or line is away, but not away perp.
  Omega_h::Real tnumer = osh_dot(normv, a2origin);
  Omega_h::Real paramt = tnumer/det;

  Omega_h::Vector<DIM> cross_e2_mline = -1.0 * Omega_h::cross(edge2, line);
  Omega_h::Real unumer = osh_dot(cross_e2_mline, a2origin);
  //attach to edge0 or vertc
  Omega_h::Real bcu = unumer/det;

  Omega_h::Vector<DIM> cross_mline_e0 = -1.0 * Omega_h::cross(line, edge0);
  Omega_h::Real vnumer = osh_dot(cross_mline_e0, a2origin);
  //attach to edge2 or vertb
  Omega_h::Real bcv = vnumer/det;

  // 0= self intersection with originating face
  bool validt = (paramt < bound_intol || paramt >1.0)? false : true;
  bool valid_uv = (bcu < 0 || bcv < 0 || bcu+bcv > 1.0)? false : true;
  bool match_dir = compare_vector_directions(normv, line);

  //found
  if(validt && valid_uv && match_dir)
  {
      xpoint = origin + paramt*line;
      status = true;
  }
  else //not found.
  {
    //order  face-to-edge clockwise ac,bc,ab ?
    //order: ab(edge0), bc(edge1), ac(edge2) /ERROR in ordering

    // Edges ordered as e0, e1, e2 as counter-clockwise on a face; e0=v0_to_v1
    const Omega_h::Reals bcc{bcu, 1.0-(bcu+bcv), bcv}; //works .TODO check how ???
    edge = min_index_(bcc, 3, 1e-6);
#ifdef DEBUG
    std::cout<<"\n minEDGE "<<edge<<" bcoords:"<<bcc.data()[0]<<" "<<bcc.data()[1]<<" "<<bcc.data()[2]<<"\n";
#endif // DEBUG
  }

  //TODO handle this case
  if(! match_dir)
  {
#ifdef DEBUG
      std::cout << "*** Line dir OPOSITE to element face normal \n";
#endif // DEBUG
  }

  return status;
}

/*
Process  all elements in parallel, from a while loop. Particles are processed in groups.
Adjacent element IDs are stored for further searching if the ptcl not done. 
If the most negative barycentric coord is for an exposed face, the wall collision routine is called,
If collision point is not found, its adjacent face (having most -ve bccords) is found and stored
for searching in the next step, in which case it skips adjacency search in next step.
Wait for the completion of threads in while loop. 
At this stage, particles are to be associated with the adjacent elements, but the particle data
are still with the source elements. Copy them into the adj. elements.
Next, check if all particles are found, or collision is detected. Make a list of remaining elements.
Continue while loop until all particles are done.
Omega_h::Write data set are to be updated during the run. These will be replaced by 'particle_structures'.
Kokkos functions to be used when Omega_h doesn't provide it.
*/
//#define DEBUG 1
OMEGA_H_INLINE bool search_mesh(Omega_h::LO nptcl, Omega_h::LO nelems, const Omega_h::Vector<3> *orig, const Omega_h::Vector<3> *dest,
   const Omega_h::Adj &dual, const Omega_h::Adj &down_r2f, const Omega_h::Adj &down_f2e, const Omega_h::Adj &up_e2f, const Omega_h::Adj &up_f2r,
   const Omega_h::Read<Omega_h::I8> &side_is_exposed, const Omega_h::LOs &mesh2verts, const Omega_h::Reals &coords, const Omega_h::LOs &face_verts,
   Omega_h::Write<Omega_h::LO> &part_flags, Omega_h::Write<Omega_h::LO> &elem_ids, Omega_h::Write<Omega_h::LO> &coll_adj_face_ids,
   Omega_h::Write<Omega_h::Real> &bccs, Omega_h::Write<Omega_h::Real> &xpoints, Omega_h::LO &loops)
{
  const auto up_e2f_edges = &up_e2f.a2ab;
  const auto up_e2f_faces = &up_e2f.ab2b;
  const auto down_f2es = &down_f2e.ab2b;
  const auto down_r2fs = &down_r2f.ab2b;
  const auto dual_faces = &dual.ab2b;
  const auto dual_elems = &dual.a2ab;
  const auto up_f2r_faces = &up_f2r.a2ab;
  const auto up_f2r_reg = &up_f2r.ab2b;

  //particle search: adjacency + boundary crossing
  auto search_ptcl = OMEGA_H_LAMBDA( Omega_h::LO ielem)
  {
    //temporary
    if(ielem != elem_ids[0]) return; //Assume all other elements are empty

    // NOTE ielem is taken as sequential from 0 ... is it elementID ? TODO verify it
    const auto tetv2v = Omega_h::gather_verts<4>(mesh2verts, ielem);
    const auto M = Omega_h::gather_vectors<4, 3>(coords, tetv2v);

    // parallel_for loop for groups of remaining particles in this element
    //......

    // Each group of particles inside the parallel_for.
    // TODO Change ntpcl, iptcl start and limit. Update global(?) indices inside.
    for(Omega_h::LO iptcl = 0; iptcl < nptcl; ++iptcl)
    {
#ifdef DEBUG
      std::cout << "Elem " << ielem << " part:" << iptcl << "\n";
#endif //DEBUG
      //temporary
      if(elem_ids[iptcl] != ielem) continue;
      bool continue_coll = (coll_adj_face_ids[iptcl] !=-1) ? true:false;
      Omega_h::LO coll_face_id = -1;


      bool do_collision = false;
      if(!continue_coll)
      {
        Omega_h::Write<Omega_h::Real> bcc(4, -1.0);

        //TESTING. Check particle origin containment in current element
        bool test_res = find_barycentric_tet(M, orig[iptcl], bcc);
#ifdef DEBUG
        if(!(all_positive(bcc.data(), 4)))
          std::cout << "ORIGIN ********NOT in elemet_id " << ielem << " \n";
#endif //DEBUG

        const bool res = find_barycentric_tet(M, dest[iptcl], bcc);

        if(all_positive(bcc.data(), 4))
        {
          part_flags.data()[iptcl] = 0;
          elem_ids[iptcl] = ielem;
          for(Omega_h::LO i=0; i<4; ++i) bccs[4*iptcl+i] = bcc[i];
          OMEGA_H_CHECK(almost_equal(bcc[0] + bcc[1] + bcc[2] +bcc[3], 1.0)); //?
#ifdef DEBUG
          std::cout << "********found in " << ielem << " \n";
          print_matrix(M);
          //print_data(M, dest[iptcl], bcc);
#endif //DEBUG
        }
        else
        {
          const Omega_h::LO min_entry = min_index(bcc.data(), 4);

          //get element ID
          auto dface_ind = (*dual_elems)[ielem];
          const auto beg_face = ielem *4;
          const auto end_face = beg_face +4;
          Omega_h::LO f_index = 0;

          for(auto iface = beg_face; iface < end_face; ++iface) //not 0..3
          {

            const auto face_id = (*down_r2fs)[iface];

#ifdef DEBUG
            auto fv2v = Omega_h::gather_verts<3>(face_verts, face_id); //Few<LO, 3>
            const auto face = Omega_h::gather_vectors<3, 3>(coords, fv2v);
            Omega_h::Few<Omega_h::Vector<3>, 3> abc;
            get_face_coords( M, f_index, abc);
            //check_face(M, face, f_index); //TODO fix this
#endif //DEBUG

            // Collision search
            if(!side_is_exposed[face_id])
            {
               //OMEGA_H_CHECK(side2side_elems[side + 1] - side2side_elems[side] == 2);
               auto adj_elem  = (*dual_faces)[dface_ind];
#ifdef DEBUG
               std::cout << "el " << ielem << " adj " << adj_elem << " face " << f_index << "\n";
#endif //DEBUG
               // DON't merge it with the above if().
               if(f_index == min_entry)
               {
#ifdef DEBUG
                 std::cout << "=====> For el|face_id=" << ielem << "," << (*down_r2fs)[iface]  << " :ADJ elem= " << adj_elem << "\n";
#endif //DEBUG
                 elem_ids[iptcl] = adj_elem;
                 break;
               }
               ++dface_ind;
            }
            else if(f_index == min_entry && bcc[f_index] < 0) // if exposed face
            {
              coll_face_id = face_id;
              do_collision = true;
#ifdef DEBUG
              std::cout << "To_search_coll for face " << coll_face_id << " faceind:" << f_index << "\n";
#endif //DEBUG
              break;
            }
            ++f_index;
          } //faces
        } //not found
      }// !coll_continue

      //TODO split into another fuction ?
      // This is the else part of the above if, and will cause loop divergence.
      // To avoid it, the collision can be a separate function, with its call another step in while loop,
      // But then the collision search will be a second run with the same element,
      // once adjacency search finds an exposed face as having smallest volume coordinate.
      if(do_collision || continue_coll)
      {
        Omega_h::LO adj_elem_id=-1, adj_face_id=-1, cross_edge = -1;
        bool next_elem = false;
        Omega_h::Vector<3> xpoint{0,0,0};
        Omega_h::LO face_id = continue_coll? coll_adj_face_ids[iptcl]:coll_face_id;
        auto fv2v = Omega_h::gather_verts<3>(face_verts, face_id); //Few<LO, 3>
        const auto face = Omega_h::gather_vectors<3, 3>(coords, fv2v);

#ifdef DEBUG
        std::cout << "********* \n Call Wall collision for el,face_id " << ielem << "," << face_id << "\n";
#endif //DEBUG
        bool detected = line_triangle_intx_simple(face, orig[iptcl], dest[iptcl], xpoint, &cross_edge);

        if(detected)
        {
          //elem_ids[iptcl] = -1;
          //coll_adj_face_ids[iptcl] = -1;
          part_flags.data()[iptcl] = 0;
          for(Omega_h::LO i=0; i<3; ++i)xpoints[iptcl*3+i] = xpoint.data()[i];
          //store current face_id and element_ids
#ifdef DEBUG
          print_osh_vector(xpoint, "COLLISION POINT");
#endif //DEBUG
          break;
        }
#ifdef DEBUG
        std::cout << "edge " << cross_edge << "\n";
#endif //DEBUG
        if(cross_edge >= 0)
        {
          //one check on a face, then follow min_entry edge to next exposed face
          auto edge_id = (*down_f2es)[face_id*3 + cross_edge];

          auto fstart = (*up_e2f_edges)[edge_id];
          auto fend = (*up_e2f_edges)[edge_id+1];
          for(Omega_h::LO fp=fstart; fp<fend; ++fp)
          {
            auto afid = (*up_e2f_faces)[fp];
            if(side_is_exposed[afid] && afid != face_id)
            {
              adj_face_id = afid;
              break;
            }
          }
          //find element id of adj_face_id. only 1 region for exposed face
          if((*up_f2r_faces)[adj_face_id+1] == 1+(*up_f2r_faces)[adj_face_id])
          {
            adj_elem_id = (*up_f2r_reg)[(*up_f2r_faces)[adj_face_id]];
            elem_ids[iptcl] = adj_elem_id;
            coll_adj_face_ids[iptcl] = adj_face_id;
            //part_flags.data()[iptcl] = 2; //collision
            next_elem = true;
          }
#ifdef DEBUG
        //std::cout << "adj_face_id " << adj_face_id << " el " << adj_elem_id << " "<< (*up_f2r_faces)[adj_face_id ]
        //          << " " << (*up_f2r_faces)[adj_face_id+1] << "\n";
#endif //DEBUG
        }

        if(cross_edge <0 || !next_elem)
        {
          elem_ids[iptcl] = ielem; //current element
#ifdef DEBUG
          std::cout << "Collision tracking lost.\n";
#endif //DEBUG
          //Error stop
          part_flags.data()[iptcl] = -1;
        }

      } //do_collision
    }//iptcl
  };

  bool found = false;

  while(!found)
  {
    //TODO check if particle is on boundary and remove from list if so.

    // Searching all elements. TODO exclude those done ?
    Omega_h::parallel_for(nelems,  search_ptcl, "search_ptcl");
    found = true;


    // TODO synchronize

    //TODO this could be a sequential bottle-neck
    for(int i=0; i<nptcl; ++i){ if(part_flags[i] > 0) {found = false; break;} }
    //Copy particle data from previous to next (adjacent) element
    ++loops;
  }
#ifdef DEBUG
  std::cout << "While loop nums " << loops << "\n";
#endif //DEBUG
} //search_mesh




} //namespace
#ifdef DEBUG
#undef DEBUG
#endif // DEBUG
#endif //define


#ifndef PUMIPIC_ADJACENCY_HPP
#define PUMIPIC_ADJACENCY_HPP

#include <iostream>
#include "Omega_h_for.hpp"
#include "Omega_h_adj.hpp"
#include "Omega_h_element.hpp"
#include "Omega_h_shape.hpp"

#include <particle_structs.hpp>

#include "pumipic_utils.hpp"
#include "pumipic_constants.hpp"
#include "pumipic_kktypes.hpp"
#include "pumipic_profiling.hpp"

namespace o = Omega_h;
namespace ps = particle_structs;

namespace pumipic {

  /*
    Top Level adjacency search routine

    @param[in] mesh The omega_h mesh
    @param[in] ptcls The particle structure
    @param[in] x_ps_orig The prepush coordinates of particles
    @param[in] x_ps_tgt The pushed coordinates of particles
    @param[in] pids The particle ids
    @param[out] elem_ids The new parent elements found by search. The passed in values will be used as the current parent element if provided
    @param[in] looplimit The max number of search loops to provide. Defaults to 0 - no limit on loops
    @param[in] requireIntersection If true, intersection with model boundaries will be calculated
    @param[out] inter_points Will be filled with the intersection points of particle paths with model boundaries
    @param[out] inter_faces Will be filled with the mesh faces on a model boundary the particle path intersected with
    @param[in] debug Turn on to output debug information
  */
  template <class ParticleType, typename Origin, typename SegmentInt, typename Target = Origin>
  bool search_mesh(o::Mesh& mesh, ParticleStructure<ParticleType>* ptcls,
                   Origin x_ps_orig, Target x_ps_tgt, SegmentInt pids,
                   o::Write<o::LO>& elem_ids,
                   bool requireIntersection,
                   o::Write<o::LO>& inter_faces,
                   o::Write<o::Real>& inter_points,
                   int looplimit = 0, 
                   int debug = 0);


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
   triangle:
              1
             / \
            1   0
           /     \
          2---2---0
*/

#define TriVerts 3
#define TriDim 2
//compute the area coordinates formed by each edge of searchElm
//the coordinates are returned in the order of the edges bounding
//searchElm
// vertex_major is to specify whether the returned Bcc will be in edge based
// notation or vertex based notation; default is edge based notation
OMEGA_H_DEVICE void barycentric_tri(
  const o::Reals triArea,
  const o::Matrix<TriDim, TriVerts> &faceCoords,
  const o::Vector<TriDim> &pos,
  o::Vector<TriVerts> &bcc,
  const int searchElm, const bool vertex_major=false) {
  const auto parent_area = triArea[searchElm];
  const auto vshift = vertex_major ? 1 : 0;
  for(int i=0; i<3; i++) {
    const auto kIdx = simplex_down_template(o::FACE, o::EDGE, (i+vshift)%3, 0);
    const auto lIdx = simplex_down_template(o::FACE, o::EDGE, (i+vshift)%3, 1);
    const auto kxy = faceCoords[kIdx];
    const auto lxy = faceCoords[lIdx];
    o::Few<o::Vector<2>, 2> tri;
    tri[0] = lxy - kxy;
    tri[1] = pos - kxy;
    const auto area = o::triangle_area_from_basis(tri);
    bcc[i] = area/parent_area;
  }
}

//TODO use  barycentric_coords_tet()
OMEGA_H_DEVICE bool find_barycentric_tet( const Omega_h::Matrix<DIM, 4> &mat,
     const Omega_h::Vector<DIM> &pos, Omega_h::Vector<4> &bcc, bool debug=false) {
  for(Omega_h::LO i=0; i<4; ++i) 
    bcc[i] = -1;
  o::Real tol = 1.0e-20;

  Omega_h::Real vals[4];
  Omega_h::Few<Omega_h::Vector<DIM>, 3> abc;
  for(Omega_h::LO iface=0; iface<4; ++iface) {
    get_face_from_face_index_of_tet(mat, iface, abc);
    auto vab = abc[1] - abc[0]; //b - a;
    auto vac = abc[2] - abc[0]; //c - a;
    auto vap = pos - abc[0]; // p - a;
    vals[iface] = o::inner_product(vap, Omega_h::cross(vac, vab)); //ac, ab
  }
  //volume using bottom face=0
  get_face_from_face_index_of_tet(mat, 0, abc);
  auto vtx3 = Omega_h::simplex_opposite_template(DIM, FDIM, 0);
  OMEGA_H_CHECK(3 == vtx3);
  // abc in order, for bottom face: M[0], M[2](=abc[1]), M[1](=abc[2])
  auto cross_ac_ab = Omega_h::cross(abc[2]-abc[0], abc[1]-abc[0]);
  auto vol6 = o::inner_product(mat[vtx3]-mat[0], cross_ac_ab);
  if(debug) 
    printInfo(" old:bccvals %g %g %g %g vol %g \n", vals[0]/6.0, vals[1]/6.0, 
      vals[2]/6.0, vals[3]/6.0,vol6/6.0);

  Omega_h::Real inv_vol = 0.0;
  if(vol6 > tol) // TODO tolerance
    inv_vol = 1.0/vol6;
  else {
    return 0;
  }
  //bcc[0] for face0 corresp to its opp vtx, so on.
  for(int i=0; i<4; ++i)
    bcc[i] = inv_vol * vals[i];
  return 1; //success
}

// BC coords are not in order of its corresp. opp. vertexes. Bccoord of tet(iface, xpoint)
OMEGA_H_DEVICE bool barycentric_coords_tet(const o::LOs& mesh2verts, 
   const o::Reals& coords, const o::Vector<DIM>& pos, const o::LO elem, 
   o::Vector<DIM+1>& bcc, o::Real tol=0, bool debug=false) {
  auto verts = Omega_h::gather_verts<DIM + 1>(mesh2verts, elem);
  auto tet = Omega_h::gather_vectors<DIM + 1, DIM>(coords, verts);
  auto vals = o::zero_vector<DIM+1>();
  Omega_h::Few<Omega_h::Vector<DIM>, 3> abc;
  for(Omega_h::LO iface=0; iface<4; ++iface) {
    get_face_from_face_index_of_tet(tet, iface, abc);
    auto vab = abc[1] - abc[0]; //b - a;
    auto vac = abc[2] - abc[0]; //c - a;
    auto vap = pos - abc[0]; // p - a;
    vals[iface] = 1.0/6.0*o::inner_product(vap, Omega_h::cross(vac, vab)); //ac, ab
    bcc[iface] = 0;
  }
  auto basis = Omega_h::simplex_basis<DIM, DIM>(tet);
  auto vol = Omega_h::tet_volume_from_basis(basis);
  if(debug)
    printInfo(" bccvals %g %g %g %g vol %g \n", vals[0], vals[1], vals[2], vals[3],vol);
  if(vol < tol)
    return 0;
  bcc = 1.0/vol * vals;
  return 1; //success
}

// BC coords are not in order of its corresp. vertexes. Bccoord of triangle (iedge, xpoint)
// corresp. to vertex obtained from simplex_opposite_template(FDIM, 1, iedge) ?
OMEGA_H_DEVICE bool find_barycentric_tri_simple(
  const Omega_h::Few<Omega_h::Vector<DIM>, 3> &abc,
  const Omega_h::Vector<3> &xpoint, Omega_h::Vector<3> &bc) {
  auto a = abc[0];
  auto b = abc[1];
  auto c = abc[2];
  auto cross = 1/2.0 * Omega_h::cross(b-a, c-a); //NOTE order
  auto norm = Omega_h::normalize(cross);
  Omega_h::Real area = o::inner_product(norm, cross);

  if(Kokkos::abs(area) < 1e-20) { //TODO
    printInfo("area is too small \n");
    return 0;
  }
  Omega_h::Real fac = 1/(area*2.0);
  bc[0] = fac * o::inner_product(norm, Omega_h::cross(b-a, xpoint-a));
  bc[1] = fac * o::inner_product(norm, Omega_h::cross(c-b, xpoint-b));
  bc[2] = fac * o::inner_product(norm, Omega_h::cross(xpoint-a, c-a));

  return 1;
}

/** \brief returns true if line dest-origin intersects triangle abc.
  \info Algorithm checks values of projection of vectors : 
  from line start to plane (Pstart), from plane to line end (Pend), and
  line onto surface normal (Pseg). 
  Requirements: No holes in domain => containment check using BCC
  should include edge/face (no -ve tolerance).
  If end point of a particle is on edge/face, and BCC fails to detect it,
  L-T search should include edge/face with a tolerance. 
  So, edge/face belongs to both adjacent elements. 
  Any wall boundary handling has to be done separately. 

  If particle start is on edge/face and BCC failed to detect end point, 
  L-T needs a tolerance to make sure start position is valid.
  This would be allowing a small negative projection of start point 
  on the plane, instead of limiting it to be positive for the standard.
  Particle path should have a positive projection, Pseg>0, which means a 
  sliding particle on an edge/face is not an intersection since Pseg=0. 
  If BCC failed to detect such slides, the maximum projection of segment
  on plane is used to find the next adjacent element. If it is wall, 
  then xpoint is same as origin when par_t is zero due to Pseg=0.
  
  If particle end point is on edge/face and L-T detects it, then the 
  adjacent element is searched next. If BCC fails in that element
  since it is on edge/face, then L-T will fail since the particle
  hasn't gone out. So, BCC search needs a tolerance, which means
  domain extends a bit outward than the wall. 
  NOTE: For boundary simulation this may be a problem, since the
  particle may be still in domain for a zero and a little -ve
  distance to boundary, causing mismatch in fields with wall.

  Limits/Tolerance: Pstart >= -tol; Pend > -tol; Pseg>0. 

  Examples of particle handling around boundary:
  If Pstart =0, and Pend=0, then Pseg won't be >0; i.e, if start and 
  end on bdry, then line won't make +ve projn with surface normal.
  If this case failed in BCC search then particle has moved out of the 
  element, then take the adjacent element for the maximum projection of
  segment onto plane, which is calculated only if start and end projection
  are within limits.
  If Pstart>0 (ptcl within), Pend=0, then Pseg will be>0; i.e, if start
  is inside domain and end is on bdry, then line has to make +ve projn 
  with surface normal.
  If Pstart=0,Pseg>0,then Pend will be >0(outside); i.e if start is on 
  bdry, and line  has +ve proj with normal, then end point will be outside.
 */
OMEGA_H_DEVICE bool line_triangle_intx_simple (
  const Omega_h::Few<Omega_h::Vector<3>, 3> &abc, 
  const Omega_h::Vector<3> &origin, const Omega_h::Vector<3> &dest,
  Omega_h::Vector<3> &xpoint, Omega_h::Real& dproj, bool reverse=false, 
  Omega_h::Real tol=0, bool debug=false) {
  for(int i=0; i<3; ++i)
    xpoint[i] = 0;

  bool found = false;
  auto line = dest - origin;
  auto edge0 = abc[1] - abc[0];
  auto edge1 = abc[2] - abc[0];
  auto normv = Omega_h::cross(edge0, edge1);
  if(reverse) {
    normv = -1*normv;
    if(debug)
      printInfo("LTintX Surface normal is flipped\n");
  }
  auto snorm_unit = Omega_h::normalize(normv);
  Omega_h::Real dist2plane = o::inner_product(abc[0] - origin, snorm_unit);
  auto plane2dest = dest - abc[0];
  Omega_h::Real proj_end = o::inner_product(snorm_unit, plane2dest);
  if(debug)
    printInfo("LTintX dist2plane %.10f pro_end %.10f\n", dist2plane, proj_end);  
  // equal required if tol=0 is used
  if(dist2plane >= -tol && proj_end >= -tol) {
    dproj = o::inner_product(line, snorm_unit);
    Omega_h::Real par_t = (dproj>0) ? dist2plane/dproj : 0; 
    xpoint = origin + par_t * line;
    // line has to make a +ve projection
    if(dproj>0) {
      Omega_h::Vector<3> bcc;
      bool res = find_barycentric_tri_simple(abc, xpoint, bcc);
      if(res && bcc[0] >=0 && bcc[0] <=1 && bcc[1] >=0 && 
        bcc[1]<=1 && bcc[2] >=0 && bcc[2] <=1)
          found = true;
      if(debug)
        printInfo("LTintX Found %d bcc+ %d par_t= %.10f dist2plane= %.10f "
           "projline= %.10f proj_out_line %.10f X %.10f %.10f %.10f \n", found, res, 
           par_t, dist2plane, dproj, proj_end, xpoint[0], xpoint[1], xpoint[2]);
    }
  }
  return found;
}

template <typename Segment>
OMEGA_H_DEVICE o::Vector<3> makeVector3(int pid, Segment xyz) {
  o::Vector<3> v;
  for(int i=0; i<3; ++i)
    v[i] = xyz(pid,i);
  return v;
}

template <typename Segment>
OMEGA_H_DEVICE o::Vector<2> makeVector2(int pid, Segment xyz) {
  o::Vector<2> v;
  for(int i=0; i<2; ++i)
    v[i] = xyz(pid,i);
  return v;
}

OMEGA_H_DEVICE o::Matrix<3, 3> gatherVectors3x3(o::Reals const& a, o::Few<o::LO, 3> v) {
  return o::gather_vectors<3, 3>(a, v);
}

OMEGA_H_DEVICE o::Matrix<3, 4> gatherVectors4x3(o::Reals const& a, o::Few<o::LO, 4> v) {
  return o::gather_vectors<4, 3>(a, v);
}


OMEGA_H_DEVICE bool isPointWithinElemTet(const o::LOs& mesh2verts, 
   const o::Reals& coords, const o::Vector<DIM>& pos, const o::LO elem, 
   o::Vector<DIM+1>& bcc, const o::Real tol=1.0e-20, bool debug=false) {
  barycentric_coords_tet(mesh2verts, coords, pos, elem, bcc, tol, debug);
  return all_positive(bcc, tol);
}

OMEGA_H_DEVICE bool isPointWithinElemTet(const o::LOs& mesh2verts, 
   const o::Reals& coords, const o::Vector<3>& pos, const o::LO elem, 
   const o::Real tol=1.0e-20) {
  auto bcc = o::zero_vector<4>();
  return isPointWithinElemTet(mesh2verts, coords, pos, elem, bcc, tol);
}

// Return a face to walk toward the destination, or -1 when it is contained.
// A barycentric walk does not promise the first face crossed by a trajectory.
template <int Dim>
OMEGA_H_DEVICE o::LO find_bcc_side(o::Real measure,
    const o::Matrix<Dim, Dim + 1>& vertices, const o::Vector<Dim>& destination,
    o::Real tol);

// Compatibility entry points retain their argument order, but use the common
// tracer rather than maintaining independent 3D traversal loops.
template <class ParticleStruct, typename Origin, typename Target, typename Ids>
bool search_mesh_3d(o::Mesh& mesh, ParticleStruct* particles,
    Origin origin, Target target, Ids ids, o::Write<o::LO>& elements,
    o::Write<o::Real>& points, o::Write<o::LO>& faces,
    int looplimit = 0, int debug = 0) {
  OMEGA_H_CHECK(mesh.dim() == 3);
  return search_mesh(mesh, particles, origin, target, ids, elements, true,
                     faces, points, looplimit, debug);
}

template <class ParticleType, typename Origin, typename Ids, typename Target = Origin>
bool search_mesh(o::Mesh& mesh, ParticleStructure<ParticleType>* particles,
    Origin origin, Target target, Ids ids, o::Write<o::LO>& elements,
    o::Write<o::Real>& points, o::Write<o::LO>& faces,
    int looplimit = 0, int debug = 0) {
  return search_mesh(mesh, particles, origin, target, ids, elements, true,
                     faces, points, looplimit, debug);
}

// To interpoalte field stored at vertices. Field has dof components, and 
// stored in order 0,1,2,3 at tet's vertices. BCC in order of faces
OMEGA_H_DEVICE Omega_h::Real interpolateTetVtx(const Omega_h::LOs& mesh2verts,
  const Omega_h::Reals& field, o::LO elem, const Omega_h::Vector<4>& bcc, 
  o::LO dof=1, o::LO comp=0, bool debug=false) {
  OMEGA_H_CHECK(all_positive(bcc)==1);
  auto tetv2v = o::gather_verts<4>(mesh2verts, elem);
  auto fv4 = o::gather_vectors<4, 1>(field, tetv2v);
  Omega_h::Real val = 0;
  for(Omega_h::LO fi=0; fi<4; ++fi) {//faces
    auto d = Omega_h::simplex_opposite_template(3,2,fi); //3,2,0,1
    auto fd = d*dof + comp;
    val = val + bcc[fi]*fv4[fd][0];
    if(debug) {// preprocess
      printInfo("bcc %.15f field %.15f val %.15f\n", bcc[fi], fv4[fd][0], val);
      printInfo("interp: %g %d %g %g \n", bcc[fi]*fv4[fd][0], d, 
        bcc[fi], fv4[fd][0]);
    }
  }
  return val;
}


OMEGA_H_DEVICE void interpolate3dFieldTet(const Omega_h::LOs& mesh2verts,
  const Omega_h::Reals &field, o::LO elem, const Omega_h::Vector<4> &bcc, 
  Omega_h::Vector<3>& fv) {
  for(int i=0; i<3; ++i) {
    fv[i] = interpolateTetVtx(mesh2verts, field, elem, bcc, 3, i);
  }
}

OMEGA_H_DEVICE void findBCCoordsInTet(const Omega_h::Reals &coords, 
   const Omega_h::LOs &mesh2verts, const Omega_h::Vector<3> &xyz, 
   const Omega_h::LO elem, Omega_h::Vector<4> &bcc) {
  const auto tetv2v = Omega_h::gather_verts<4>(mesh2verts, elem);
  auto mat = Omega_h::gather_vectors<4, 3>(coords, tetv2v);
  const bool res = find_barycentric_tet(mat, xyz, bcc);
  OMEGA_H_CHECK(res==1);
  OMEGA_H_CHECK(all_positive(bcc)==1);
}

// Voronoi regions of triangle, to find nearest point on triangle
enum TriRegion {
  VTXA, 
  VTXB,
  VTXC,
  EDGEAB,
  EDGEAC,
  EDGEBC,
  TRIFACE,
  NREGIONS 
};

//TODO test this function, is this needed ?
OMEGA_H_DEVICE o::Vector<3> closest_point_on_triangle_wnormal(
  const o::Few< o::Vector<3>, 3> &abc,
  const o::Vector<3> &p, o::LO* reg=nullptr) {
  // Check if P in vertex region outside A
  auto q = o::zero_vector<3>();
  auto a = abc[0];
  auto b = abc[1];
  auto c = abc[2];

  auto ab = b - a;
  auto ac = c - a;
  auto bc = c - b;
  // Compute parametric position s for projection P’ of P on AB,
  // P’ = A + s*AB, s = snom/(snom+sdenom)
  float snom = o::inner_product(p - a, ab);
  float sdenom = o::inner_product(p - b, a - b);
  // Compute parametric position t for projection P’ of P on AC,
  // P’ = A + t*AC, s = tnom/(tnom+tdenom)
  float tnom = o::inner_product(p - a, ac);
  float tdenom = o::inner_product(p - c, a - c);
  if(snom <= 0.0 && tnom <= 0.0){
    if(reg)
      *reg = VTXA;
    return a;
  } // Vertex region early out
  // Compute parametric position u for projection P’ of P on BC,
  // P’ = B + u*BC, u = unom/(unom+udenom)
  float unom = o::inner_product(p - b, bc);
  float udenom = o::inner_product(p - c, b - c);
  if (sdenom <= 0.0 && unom <= 0.0){
    if(reg)
      *reg = VTXB; // Vertex region early out
    return b;
  }
  if(tdenom <= 0.0 && udenom <= 0.0){
    if(reg)
     *reg = VTXC; // Vertex region early out
    return c;
  }
  // P is outside (or on) AB if the triple scalar product [N PA PB] <= 0
  auto n = o::cross(b - a, c - a);
  auto temp = o::cross(a - p, b - p);
  float vc = o::inner_product(n, temp);
  // If P outside AB and within feature region of AB,
  // return projection of P onto AB
  if (vc <= 0.0 && snom >= 0.0 && sdenom >= 0.0){
    q = a + snom / (snom + sdenom) * ab;
    if(reg)
     *reg = EDGEAB;
    return q;
  }
  // P is outside (or on) BC if the triple scalar product [N PB PC] <= 0
  auto temp1 = o::cross(b - p, c - p);
  float va = o::inner_product(n, temp1);
  // If P outside BC and within feature region of BC,
  // return projection of P onto BC
  if (va <= 0.0 && unom >= 0.0 && udenom >= 0.0){
    q = b + unom / (unom + udenom) * bc;
    if(reg)
     *reg = EDGEBC;
    return q;
  }
  // P is outside (or on) CA if the triple scalar product [N PC PA] <= 0
  auto temp2 = o::cross(c - p, a - p);
  float vb = o::inner_product(n, temp2);
  // If P outside CA and within feature region of CA,
  // return projection of P onto CA
  if (vb <= 0.0 && tnom >= 0.0 && tdenom >= 0.0){
    q =  a + tnom / (tnom + tdenom) * ac;
    if(reg)
     *reg = EDGEAC;
    return q;
  }
  // P must project inside face region. Compute Q using barycentric coordinates
  float u = va / (va + vb + vc);
  float v = vb / (va + vb + vc);
  float w = 1.0 - u - v; // = vc / (va + vb + vc)
  q = u * a + v * b + w * c;
  if(reg)
   *reg = TRIFACE;
  return q;
}


//Ref: Real-time Collision Detection by Christer Ericson, 2005.
//ptp = ref point; ptq = nearest point on triangle; abc = triangle
OMEGA_H_DEVICE o::Vector<3> closest_point_on_triangle( 
   const o::Few< o::Vector<3>, 3> &abc, 
   const o::Vector<3> &ptp, o::LO* reg=nullptr) {
  o::LO debug = 0;
  o::LO region = -1;
  auto ptq = o::zero_vector<3>();
  // Check if P in vertex region outside A
  auto pta = abc[0];
  auto ptb = abc[1];
  auto ptc = abc[2];
  auto vab = ptb - pta;
  auto vac = ptc - pta;
  auto vap = ptp - pta;
  auto d1 = o::inner_product(vab, vap);
  auto d2 = o::inner_product(vac, vap);
  if(d1 <= 0 && d2 <= 0) {
    // barycentric coordinates (1,0,0)
    for(int i=0; i<3; ++i)
      ptq[i] = pta[i];
    region = VTXA;
    if(reg)
      *reg = region;
    return ptq; 
  }
  // Check if P in vertex region outside B
  auto vbp = ptp - ptb;
  auto d3 = o::inner_product(vab, vbp);
  auto d4 = o::inner_product(vac, vbp);
  if(region <0 && d3 >= 0 && d4 <= d3){ 
    // barycentric coordinates (0,1,0)
    for(int i=0; i<3; ++i)
      ptq[i] = ptb[i];
    region = VTXB;
    if(reg)
      *reg = region;
    return ptq; 
  }
  // Check if P in edge region of AB, if so return projection of P onto AB
  auto vc = d1*d4 - d3*d2;
  if(region <0 && vc <= 0 && d1 >= 0 && d3 <= 0) {
    auto v = d1 / (d1 - d3);
    // barycentric coordinates (1-v,v,0)
    ptq = v*vab;
    ptq = ptq + pta; 
    region = EDGEAB;
    return ptq;
  }
  // Check if P in vertex region outside C
  auto vcp = ptp - ptc;
  auto d5 = o::inner_product(vab, vcp);
  auto d6 = o::inner_product(vac, vcp);
  if(region <0 && d6 >= 0 && d5 <= d6) { 
    // barycentric coordinates (0,0,1)
    for(int i=0; i<3; ++i)
      ptq[i] = ptc[i]; 
    region = VTXC;
    if(reg)
      *reg = region;
    return ptq;
  }
  // Check if P in edge region of AC, if so return projection of P onto AC
  auto vb = d5*d2 - d1*d6;
  if(region <0 && vb <= 0 && d2 >= 0 && d6 <= 0) {
    auto w = d2 / (d2 - d6);
    // barycentric coordinates (1-w,0,w)
    ptq = w*vac;
    ptq = ptq + pta; 
    region = EDGEAC;
    if(reg)
      *reg = region;
    return ptq;
  }
  // Check if P in edge region of BC, if so return projection of P onto BC
  auto va = d3*d6 - d5*d4;
  if(region <0 && va <= 0 && (d4 - d3) >= 0 && (d5 - d6) >= 0) {
    auto w = (d4 - d3) / ((d4 - d3) + (d5 - d6));
    // barycentric coordinates (0,1-w,w)
    ptq =  ptb + w * (ptc - ptb); 
    region = EDGEBC;
    if(reg)
      *reg = region;
    return ptq;
  }
  // P inside face region. Compute Q through its barycentric coordinates (u,v,w)
  if(region <0) {
    auto inv = 1.0/(va + vb + vc);
    auto v = vb * inv;
    auto w = vc * inv;
    // u*a + v*b + w*c, u = va * inv = 1 - v - w
    ptq =  pta + v * vab+ w * vac;
    region = TRIFACE;
    if(reg)
      *reg = region;
    return ptq;
  }
  if(debug)
    printInfo("d's:: %0.3f %0.3f %0.3f %0.3f %0.3f %0.3f \n", d1, d2, d3, d4, d5, d6);

  return ptq;
}

// The old 2D API uses -1 for "start from the particle parent" and -nelems
// for a particle already outside after a field-following operation.
template <class ParticleStruct, typename Origin, typename Target, typename Ids>
bool search_mesh_2d(o::Mesh& mesh, ParticleStruct* particles,
    Origin origin, Target target, Ids ids, o::Write<o::LO> elements,
    int looplimit = 0, bool debug = false) {
  OMEGA_H_CHECK(mesh.dim() == 2);
  OMEGA_H_CHECK(elements.size() == particles->capacity());
  const auto outside = -mesh.nelems();
  auto initialize = PS_LAMBDA(const int parent, const int pid, const bool mask) {
    if (!mask)
      elements[pid] = -1;
    else if (elements[pid] == -1)
      elements[pid] = parent;
    else if (elements[pid] == outside)
      elements[pid] = -1;
  };
  parallel_for(particles, initialize, "search_2d_initialize");
  o::Write<o::LO> faces;
  o::Write<o::Real> points;
  const bool found = search_mesh(mesh, particles, origin, target, ids, elements,
                                 false, faces, points, looplimit, debug);
  if (!found) {
    const auto vertices = mesh.ask_elem_verts();
    const auto coords = mesh.coords();
    const auto areas = measure_elements_real(&mesh);
    auto discard_unlocated = PS_LAMBDA(const int, const int pid, const bool mask) {
      if (!mask || elements[pid] < 0) return;
      const auto elem = elements[pid];
      const auto verts = o::gather_verts<3>(vertices, elem);
      if (find_bcc_side<2>(areas[elem], o::gather_vectors<3, 2>(coords, verts),
                           makeVector2(pid, target), EPSILON) != -1)
        elements[pid] = -1;
    };
    parallel_for(particles, discard_unlocated, "search_2d_discard_unlocated");
  }
  return found;
}

OMEGA_H_DEVICE o::LO search_mesh_2d_pt(const o::Read<o::I8> side_is_exposed,
                                       const o::LOs faces2verts,
                                       const o::Reals coords,
                                       const o::Adj faces2edges,
                                       const o::Adj edges2faces,
                                       const o::Reals triArea,
                                       // particle origin
                                       const o::Vector<2> ptclOrigin,
                                       // particle destination
                                       const o::Vector<2> ptclDest,
                                       // particle id
                                       const o::LO pid,
                                       // starting element for particle search
                                       const o::LO initial_elem,
                                       o::LO& loops,
                                       // [optional] number of loops before giving up
                                       const o::LO looplimit = 0,
                                       const bool debug = false) {

  // 0: particle is not done yet.
  // 1: particle has hit a boundary or reached its destination
  o::LO ptcl_done = 0;
  // store the last crossed edge
  o::LO lastEdge = -1;
  bool found = false;
  loops = 0;
  const o::LOs faceEdges = faces2edges.ab2b;
  o::LO elem_id = initial_elem;
  while (!found) {
    //active particle that is still moving to its target position
    if (!ptcl_done) {
      const o::LO searchElm = elem_id;
      OMEGA_H_CHECK(searchElm >= 0);
      const auto edges = o::gather_down<3>(faceEdges, searchElm);
      const auto faceVerts = o::gather_verts<3>(faces2verts, searchElm);
      const auto faceCoords = o::gather_vectors<3,2>(coords, faceVerts);
      const auto idx = find_bcc_side<2>(triArea[searchElm], faceCoords,
                                         ptclDest, EPSILON);
      ptcl_done = (idx == -1);
      lastEdge = ptcl_done ? -1 : edges[idx];
    }

    if (!ptcl_done) {
      assert(lastEdge != -1);
      const o::LO bridge = lastEdge;
      const auto exposed = side_is_exposed[bridge];
      ptcl_done = exposed;
      //leaves domain if exposed
      if (exposed) elem_id = -1;
    }

    auto e2f_vals = edges2faces.ab2b; // CSR value array
    auto e2f_offsets = edges2faces.a2ab; // CSR offset array, index by mesh edge ids
    if (!ptcl_done) {
      const o::LO searchElm = elem_id;
      const o::LO bridge = lastEdge;
      const o::LO e2f_first = e2f_offsets[bridge];
      #ifdef _DEBUG
        const o::LO e2f_last = e2f_offsets[bridge+1];
        const o::LO upFaces = e2f_last - e2f_first;
        assert(upFaces == 2);
      #endif
      const o::LO faceA = e2f_vals[e2f_first];
      const o::LO faceB = e2f_vals[e2f_first+1];
      assert(faceA != faceB);
      assert(faceA == searchElm || faceB == searchElm);
      const o::LO nextElm = (faceA == searchElm) ? faceB : faceA;
      elem_id = nextElm;
    }

    // check if we have found the particle final element
    found = ptcl_done ? true: false;
    ++loops;

    if (looplimit && loops >= looplimit && !ptcl_done) {
      const o::LO searchElm = elem_id;
      elem_id = -1;
      //TODO: use preprocessor macro
      if (debug) {
        printInfo("elm %d ptcl %d notFound %g %g to %g %g\n",
               searchElm, pid, ptclOrigin[0], ptclOrigin[1],
               ptclDest[0], ptclDest[1]);
      }
      elem_id = -1;
      printInfo("ERROR: loop limit %d exceeded, particle %d is "
              "not found; deleting them.\n", looplimit, pid);
      break;
    }
  }

  return elem_id;
}

} //namespace
#endif //define

#include "pumipic_adjacency.tpp"

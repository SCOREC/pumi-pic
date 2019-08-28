#ifndef PUMIPIC_ADJACENCY_HPP
#define PUMIPIC_ADJACENCY_HPP

#include <iostream>

#include "Omega_h_for.hpp"
#include "Omega_h_adj.hpp"
#include "Omega_h_element.hpp"
#include "Omega_h_shape.hpp"

#include <SellCSigma.h>
#include <SCS_Macros.h>

#include "pumipic_utils.hpp"
#include "pumipic_constants.hpp"
#include "pumipic_kktypes.hpp"

#include "segSegIsect.hpp"

namespace o = Omega_h;
namespace ps = particle_structs;

namespace pumipic
{

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
//retrieve face coords in the Omega_h order
OMEGA_H_INLINE void get_face_coords(const Omega_h::Matrix<DIM, 4> &M,
          const Omega_h::LO iface, Omega_h::Few<Omega_h::Vector<DIM>, 3> &abc)
{
   //face_vert:0,2,1; 0,1,3; 1,2,3; 2,0,3
    OMEGA_H_CHECK(iface<4 && iface>=0);
    abc[0] = M[Omega_h::simplex_down_template(DIM, FDIM, iface, 0)];
    abc[1] = M[Omega_h::simplex_down_template(DIM, FDIM, iface, 1)];
    abc[2] = M[Omega_h::simplex_down_template(DIM, FDIM, iface, 2)];

}

OMEGA_H_INLINE void get_edge_coords(const Omega_h::Few<Omega_h::Vector<DIM>, 3> &abc,
          const Omega_h::LO iedge, Omega_h::Few<Omega_h::Vector<DIM>, 2> &ab)
{
   //edge_vert:0,1; 1,2; 2,0
    ab[0] = abc[Omega_h::simplex_down_template(FDIM, 1, iedge, 0)];
    ab[1] = abc[Omega_h::simplex_down_template(FDIM, 1, iedge, 1)];
}

OMEGA_H_INLINE void check_face(const Omega_h::Matrix<DIM, 4> &M,
    const Omega_h::Few<Omega_h::Vector<DIM>, 3>& face, const Omega_h::LO faceid )
{
    Omega_h::Few<Omega_h::Vector<DIM>, 3> abc;
    get_face_coords( M, faceid, abc);

    OMEGA_H_CHECK(true == compare_array(abc[0].data(), face[0].data(), DIM)); //a
    OMEGA_H_CHECK(true == compare_array(abc[1].data(), face[1].data(), DIM)); //b
    OMEGA_H_CHECK(true == compare_array(abc[2].data(), face[2].data(), DIM)); //c
}

//https://www.scratchapixel.com/lessons/3d-basic-rendering/ray-tracing-rendering-a-triangle/barycentric-coordinates
#define TriVerts 3
#define TriDim 2
OMEGA_H_DEVICE void barycentric_tri(
    const o::Reals triArea,
    const o::Matrix<TriDim, TriVerts> &faceCoords,
    const o::Vector<TriDim> &pos,
    o::Vector<TriVerts> &bcc,
    const int searchElm) {
  //triangle defined by vertices A,B,C
  // point P is potentially inside of it
  //     C
  //    /  \
  //   /    \
  //  A ---- B
  //omega_h triangles list their vertices following the
  // right hand rule (counter clockwise)
  // i.e.; B=0 C=1 A=2 or C=0 A=1 B=2 or A=0 B=1 C=2
  o::Few<o::Vector<2>, 2> tri;
  tri[0] = faceCoords[1] - faceCoords[0];
  tri[1] = faceCoords[2] - faceCoords[0];
  auto abc_area = o::triangle_area_from_basis(tri);
  OMEGA_H_CHECK(o::are_close(abc_area, triArea[searchElm])); //sanity check

  tri[0] = faceCoords[0] - faceCoords[2];
  tri[1] = pos           - faceCoords[2];
  auto cap_area = o::triangle_area_from_basis(tri);

  tri[0] = faceCoords[1] - faceCoords[0];
  tri[1] = pos           - faceCoords[0];
  auto abp_area = o::triangle_area_from_basis(tri);

  // barycentric coords
  bcc[0] = cap_area/abc_area;   //u
  bcc[1] = abp_area/abc_area;   //v
  bcc[2] = 1 - bcc[0] - bcc[1]; //w
}

// BC coords are not in order of its corresp. opp. vertexes. Bccoord of tet(iface, xpoint)
//TODO Warning: Check opposite_template use in this before using
OMEGA_H_INLINE bool find_barycentric_tet( const Omega_h::Matrix<DIM, 4> &Mat,
     const Omega_h::Vector<DIM> &pos, Omega_h::Vector<4> &bcc)
{
  for(Omega_h::LO i=0; i<4; ++i) bcc[i] = -1;

  Omega_h::Real vals[4];
  Omega_h::Few<Omega_h::Vector<DIM>, 3> abc;
  for(Omega_h::LO iface=0; iface<4; ++iface) // TODO last not needed
  {
    get_face_coords(Mat, iface, abc);
    auto vab = abc[1] - abc[0]; //b - a;
    auto vac = abc[2] - abc[0]; //c - a;
    auto vap = pos - abc[0]; // p - a;
    vals[iface] = osh_dot(vap, Omega_h::cross(vac, vab)); //ac, ab NOTE

  }
  //volume using bottom face=0
  get_face_coords(Mat, 0, abc);
  auto vtx3 = Omega_h::simplex_opposite_template(DIM, FDIM, 0);
  OMEGA_H_CHECK(3 == vtx3);
  // abc in order, for bottom face: M[0], M[2](=abc[1]), M[1](=abc[2])
  Omega_h::Vector<DIM> cross_ac_ab = Omega_h::cross(abc[2]-abc[0], abc[1]-abc[0]); //NOTE
  Omega_h::Real vol6 = osh_dot(Mat[vtx3]-Mat[0], cross_ac_ab);
  Omega_h::Real inv_vol = 0.0;
  if(vol6 > EPSILON) // TODO tolerance
    inv_vol = 1.0/vol6;
  else
  {
    return 0;
  }
  bcc[0] = inv_vol * vals[0]; //for face0, cooresp. to its opp. vtx.
  bcc[1] = inv_vol * vals[1];
  bcc[2] = inv_vol * vals[2];
  bcc[3] = inv_vol * vals[3]; // 1-others

  return 1; //success
}

// BC coords are not in order of its corresp. vertexes. Bccoord of triangle (iedge, xpoint)
// corresp. to vertex obtained from simplex_opposite_template(FDIM, 1, iedge) ?
OMEGA_H_INLINE bool find_barycentric_tri_simple(const Omega_h::Few<Omega_h::Vector<DIM>, 3> &abc,
     const Omega_h::Vector<3> &xpoint, Omega_h::Vector<3> &bc)
{
  Omega_h::Vector<DIM> a = abc[0];
  Omega_h::Vector<DIM> b = abc[1];
  Omega_h::Vector<DIM> c = abc[2];
  Omega_h::Vector<DIM> cross = 1/2.0 * Omega_h::cross(b-a, c-a); //NOTE order
  Omega_h::Vector<DIM> norm = Omega_h::normalize(cross);
  Omega_h::Real area = osh_dot(norm, cross);

  if(std::abs(area) < 1e-6)  //TODO
    return 0;
  Omega_h::Real fac = 1/(area*2.0);
  bc[0] = fac * osh_dot(norm, Omega_h::cross(b-a, xpoint-a));
  bc[1] = fac * osh_dot(norm, Omega_h::cross(c-b, xpoint-b));
  bc[2] = fac * osh_dot(norm, Omega_h::cross(xpoint-a, c-a));

  return 1;
}

/** \brief returns true if line dest-origin intersects the triangle abc
 */
OMEGA_H_INLINE bool line_triangle_intx_simple(const Omega_h::Few<Omega_h::Vector<DIM>, 3> &abc,
    const Omega_h::Vector<DIM> &origin, const Omega_h::Vector<DIM> &dest,
    Omega_h::Vector<DIM> &xpoint, bool reverse=false )
{
  const auto debug = 0;
  for(int i=0; i<DIM; ++i)
    xpoint[i] = 0;

  //Boundary exclusion. Don't set it globally and change randomnly.
  const Omega_h::Real bound_intol = 0;//SURFACE_EXCLUDE; //TODO optimum value ?

  bool found = false;
  const Omega_h::Vector<DIM> line = dest - origin;
  const Omega_h::Vector<DIM> edge0 = abc[1] - abc[0];
  const Omega_h::Vector<DIM> edge1 = abc[2] - abc[0];
  Omega_h::Vector<DIM> normv = Omega_h::cross(edge0, edge1);
  if(reverse)
  {
    normv = -1*normv;
    if(debug)
      printf("Surface normal reversed\n");

  }
  const Omega_h::Vector<DIM> snorm_unit = Omega_h::normalize(normv);
  const Omega_h::Real dist2plane = osh_dot(abc[0] - origin, snorm_unit);
  const Omega_h::Real proj_lined =  osh_dot(line, snorm_unit);
  const Omega_h::Vector<DIM> surf2dest = dest - abc[0];

  if(std::abs(proj_lined) >0)
  {
    const Omega_h::Real par_t = dist2plane/proj_lined;
    if(debug)
      printf(" abs(proj_lined)>0;  par_t= %f dist2plane= %f "
             "; proj_lined= %f \n", par_t, dist2plane, proj_lined);
    if (par_t > bound_intol && par_t <= 1.0) //TODO test tol value
    {
      xpoint = origin + par_t * line;
      Omega_h::Vector<3> bcc;
      bool res = find_barycentric_tri_simple(abc, xpoint, bcc);
      if(res)
      {
        if(! (bcc[0] < 0 || bcc[2] < 0 || bcc[0]+bcc[2] > 1.0) ) //TODO all zeros ?
        {
          const Omega_h::Real proj = osh_dot(snorm_unit, surf2dest);
          if(proj >0) found = true;
          else if(proj<0)
          {
            if(debug)
              printf("Particle Entering domain\n");
          }
          else if(almost_equal(proj,0.0)) //TODO use tol
          { 
            if(debug)
              printf("Particle path on surface\n");
          }
        }
      }
    }
    else if(par_t >1.0)
    {
      if(debug)
        printf("Line origin and destination are on the same side of face \n");
    }
    else if(par_t < bound_intol) // dist2plane ~0. Line contained in plane, no intersection?
    {
      if(debug)
        printf("No/Self-intersection of ptcl origin with plane at origin."
               "t= %f %f %f\n", par_t, dist2plane, proj_lined);
    }
  }
  else
  {
    if(debug)
      printf("WARNING: Line and plane are parallel \n");
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
OMEGA_H_DEVICE o::LO getfmap(int i) {
  assert(i>=0 && i<8);
  const o::LO fmap[8] = {2,1,1,3,2,3,0,3};
  return fmap[i];
}
OMEGA_H_DEVICE o::Matrix<3, 3> gatherVectors3x3(o::Reals const& a, o::Few<o::LO, 3> v) {
  return o::gather_vectors<3, 3>(a, v);
}
OMEGA_H_DEVICE o::Matrix<3, 4> gatherVectors4x3(o::Reals const& a, o::Few<o::LO, 4> v) {
  return o::gather_vectors<4, 3>(a, v);
}

//How to avoid redefining the MemberType? each application will define it
//differently. Templating search_mesh with
//template < typename ParticleType >
//results in an error on get<> as an unresolved function.

template < class ParticleType>
bool search_mesh(o::Mesh& mesh, ps::SellCSigma< ParticleType >* scs,
                 Segment3d x_scs_d, Segment3d xtgt_scs_d, SegmentInt pid_d,
                 o::Write<o::LO> elem_ids, o::Write<o::Real> xpoints_d,
                 o::Write<o::LO> xface_id, int looplimit=0) {
  const int debug = 0;

  const auto dual = mesh.ask_dual();
  const auto down_r2f = mesh.ask_down(3, 2);
  const auto side_is_exposed = mark_exposed_sides(&mesh);
  const auto mesh2verts = mesh.ask_elem_verts();
  const auto coords = mesh.coords();
  const auto face_verts =  mesh.ask_verts_of(2);
  const auto down_r2fs = down_r2f.ab2b;
  const auto dual_faces = dual.ab2b;
  const auto dual_elems = dual.a2ab;

  const auto scsCapacity = scs->capacity();

  // ptcl_done[i] = 1 : particle i has hit a boundary or reached its destination
  o::Write<o::LO> ptcl_done(scsCapacity, 1, "ptcl_done");
  // particle intersection points
  o::Write<o::Real> xpoints(3*scsCapacity, 0);
  // store the next parent for each particle
  o::Write<o::LO> elem_ids_next(scsCapacity,-1);
  // flag to move origin if intersection fails
  auto lamb = SCS_LAMBDA(const int& e, const int& pid, const int& mask) {
    if(mask > 0) {
      elem_ids[pid] = e;
      ptcl_done[pid] = 0;
      if (debug)
        printf("pid %3d mask %1d elem_ids %6d\n", pid, mask, elem_ids[pid]);
    } else {
      elem_ids[pid] = -1;
      ptcl_done[pid] = 1;
    }
  };
  scs->parallel_for(lamb, "init_search");
  bool found = false;
  int loops = 0;
  while(!found) {
    if(debug) {
      fprintf(stderr, "------------ %d ------------\n", loops);
    }
    //pid is same for a particle between iterations in this while loop
    auto lamb = SCS_LAMBDA(const int& e, const int& pid, const int& mask) {
      //inactive particle that is still moving to its target position
      if( mask > 0 && !ptcl_done[pid] ) {
        auto elmId = e;
        auto ptcl = pid_d(pid);
        if(debug)
          printf("Elem %d ptcl: %d\n", elmId, ptcl);
        if(elmId != elem_ids[pid]) {
          elmId = elem_ids[pid];
          if(debug)
            printf("Switching: Elem %d ptcl: %d\n", elmId, ptcl);
        }
        OMEGA_H_CHECK(elmId >= 0);
        auto tetv2v = o::gather_verts<4>(mesh2verts, elmId);
        auto M = gatherVectors4x3(coords, tetv2v);
        auto dest = makeVector3(pid, xtgt_scs_d);
        auto orig = makeVector3(pid, x_scs_d);
        Omega_h::Vector<4> bcc;
        if(loops == 0) {
          //make sure particle origin is in initial element
          find_barycentric_tet(M, orig, bcc);
          if(!all_positive(bcc, 0)) {
            printf("ptcl %d elem %d => %d orig %.3f %.3f %.3f dest %.3f %.3f %.3f\n",
              ptcl, e, elmId, orig[0], orig[1], orig[2], dest[0], dest[1], dest[2]);
            printf("Particle doesn't belong to this element at loops=0");
            OMEGA_H_CHECK(false);
          }
        }
        //check if the destination is this element
        find_barycentric_tet(M, dest, bcc);
        if(all_positive(bcc, 0)) {
          if(debug)
            printf("ptcl %d is in destination elm %d\n", ptcl, elmId);
          elem_ids_next[pid] = elem_ids[pid];
          ptcl_done[pid] = 1;
        } else {
          if(debug)
            printf("ptcl %d checking adj elms\n", ptcl);
          //get element ID
          auto dface_ind = dual_elems[elmId];
          const auto beg_face = elmId *4;
          const auto end_face = beg_face +4;
          o::LO f_index = 0;
          bool detected = false;
          for(auto iface = beg_face; iface < end_face; ++iface) {
            const auto face_id = down_r2fs[iface];
            auto xpoint = o::zero_vector<3>();
            bool exposed = side_is_exposed[face_id];
            auto fv2v = o::gather_verts<3>(face_verts, face_id);
            const auto face = gatherVectors3x3(coords, fv2v);
            o::LO matInd1 = getfmap(f_index*2);
            o::LO matInd2 = getfmap(f_index*2+1);
            bool inverse = true;
            if(fv2v[1] == tetv2v[matInd1] && fv2v[2] == tetv2v[matInd2])
              inverse = false;

            detected = line_triangle_intx_simple(face, orig, dest, xpoint, inverse);
            if(debug)
              printf("\t :ptcl %d faceid %d flipped %d exposed %d detected %d\n", ptcl, 
                face_id, inverse, exposed, detected);

            if(detected && exposed) {
              ptcl_done[pid] = 1;
              for(o::LO i=0; i<3; ++i)
                xpoints[pid*3+i] = xpoint[i];
              elem_ids_next[pid] = -1;
              if(debug) {
                printf("ptcl %d faceid %d detected and exposed, next parent elm %d\n",
                    ptcl, face_id, elem_ids_next[pid]);
              }
              break;
            } else if(detected && !exposed) {
              auto adj_elem  = dual_faces[dface_ind];
              elem_ids_next[pid] = adj_elem;
              if(debug) {
                printf("ptcl %d faceid %d detected and !exposed, next parent elm %d\n",
                    ptcl, face_id, elem_ids_next[pid]);
              }
              break;
            }
            // no line triangle intersection found for the current face
            // appears to be a guess at the next element based on the smallest BCC
            if(!exposed) {
              if(debug)
                printf("ptcl %d faceid %d !detected and !exposed\n", pid, face_id);
              ++dface_ind;
              const o::LO min_ind = min_index(bcc, 4);
              if(f_index == min_ind) {
                elem_ids_next[pid] = dual_faces[dface_ind];
              }
            }
            ++f_index;
          } //for iface

        } //else not in current element
      } //if active particle
    };

    scs->parallel_for(lamb, "adj_search");

    found = true;
    auto cp_elm_ids = OMEGA_H_LAMBDA( o::LO i) {
      elem_ids[i] = elem_ids_next[i];
    };
    o::parallel_for(elem_ids.size(), cp_elm_ids, "copy_elem_ids");

    o::LOs ptcl_done_r(ptcl_done);
    auto minFlag = o::get_min(ptcl_done_r);
    if(minFlag == 0)
      found = false;
    //Copy particle data from previous to next (adjacent) element
    ++loops;

    if(looplimit && loops > looplimit) {
      if (debug)
        fprintf(stderr, "ERROR:loop limit %d exceeded\n", looplimit);
      break;
    }
  }
  return found;
}

template < class ParticleType>
bool search_mesh_2d(o::Mesh& mesh, // (in) mesh
                 ps::SellCSigma< ParticleType >* scs, // (in) particle structure
                 Segment3d x_scs_d, // (in) starting particle positions
                 Segment3d xtgt_scs_d, // (in) target particle positions
                 SegmentInt pid_d, // (in) particle ids
                 o::Write<o::LO> elem_ids, // (out) parent element ids for the target positions
                 o::Write<o::Real> xpoints_d, // (out) particle-boundary intersection points
                 int looplimit=0) {
  Kokkos::Profiling::pushRegion("pumpipic_search_mesh_2d");

  const auto faces2edges = mesh.ask_down(o::FACE, o::EDGE);
  const auto edges2faces = mesh.ask_up(o::EDGE, o::FACE);
  const auto side_is_exposed = mark_exposed_sides(&mesh);
  const auto faces2verts = mesh.ask_elem_verts();
  const auto coords = mesh.coords();
  const auto edge_verts =  mesh.ask_verts_of(o::EDGE);
  const auto faceEdges = faces2edges.ab2b;
  const auto triArea = measure_elements_real(&mesh);

  const auto scsCapacity = scs->capacity();

  // ptcl_done[i] = 1 : particle i has hit a boundary or reached its destination
  o::Write<o::LO> ptcl_done(scsCapacity, 1, "ptcl_done");
  // store the next parent for each particle
  o::Write<o::LO> elem_ids_next(scsCapacity,-1);
  // store the last crossed edge
  o::Write<o::LO> lastEdge(scsCapacity,-1);
  auto lamb = SCS_LAMBDA(const int& e, const int& pid, const int& mask) {
    if(mask > 0) {
      elem_ids[pid] = e;
      ptcl_done[pid] = 0;
    } else {
      elem_ids[pid] = -1;
      ptcl_done[pid] = 1;
    }
  };
  scs->parallel_for(lamb);

  auto checkParent = SCS_LAMBDA(const int& e, const int& pid, const int& mask) {
    //inactive particle that is still moving to its target position
    if( mask > 0 && !ptcl_done[pid] ) {
      auto searchElm = elem_ids[pid];
      auto ptcl = pid_d(pid);
      OMEGA_H_CHECK(searchElm >= 0);
      auto faceVerts = o::gather_verts<3>(faces2verts, searchElm);
      const auto faceCoords = o::gather_vectors<3,2>(coords, faceVerts);
      auto ptclOrigin = makeVector2(pid, x_scs_d);
      Omega_h::Vector<3> faceBcc;
      barycentric_tri(triArea, faceCoords, ptclOrigin, faceBcc, searchElm);
      if(!all_positive(faceBcc)) {
        printf("Particle not in element! ptcl %d elem %d => %d "
          "orig %.3f %.3f bcc %.3f %.3f %.3f\n",
          ptcl, e, searchElm, ptclOrigin[0], ptclOrigin[1],
          faceBcc[0], faceBcc[1], faceBcc[2]);
        OMEGA_H_CHECK(false);
      }
    } //if active
  };
  scs->parallel_for(checkParent);

  bool found = false;
  int loops = 0;
  while(!found) {
    auto checkCurrentElm = SCS_LAMBDA(const int& e, const int& pid, const int& mask) {
      //active particle that is still moving to its target position
      if( mask > 0 && !ptcl_done[pid] ) {
        auto searchElm = elem_ids[pid];
        auto ptcl = pid_d(pid);
        OMEGA_H_CHECK(searchElm >= 0);
        const auto faceVerts = o::gather_verts<3>(faces2verts, searchElm);
        const auto faceCoords = o::gather_vectors<3,2>(coords, faceVerts);
        const auto ptclDest = makeVector2(pid, xtgt_scs_d);
        const auto ptclOrigin = makeVector2(pid, x_scs_d);
        Omega_h::Vector<3> faceBcc;
        barycentric_tri(triArea, faceCoords, ptclDest, faceBcc, searchElm);
        auto isDestInParentElm = all_positive(faceBcc);
        ptcl_done[pid] = isDestInParentElm;
        elem_ids_next[pid] = elem_ids[pid]; //NOT SURE ABOUT SETTING THIS UNCONDITIONALLY
      }
    };
    scs->parallel_for(checkCurrentElm);

    auto checkAdjElm = SCS_LAMBDA(const int& e, const int& pid, const int& mask) {
      //active particle that is still moving to its target position and is not
      //in the current element
      if( mask > 0 && !ptcl_done[pid] ) {
        auto searchElm = elem_ids[pid];
        auto ptcl = pid_d(pid);
        OMEGA_H_CHECK(searchElm >= 0);
        const auto edges = o::gather_down<3>(faceEdges, searchElm);
        const auto ptclDest = makeVector2(pid, xtgt_scs_d);
        const auto ptclOrigin = makeVector2(pid, x_scs_d);
        // check each edge of triangle for line-line intersection
        Segment2D ptclSeg, edgeSeg;
        ptclSeg.P0[0] = ptclOrigin[0];
        ptclSeg.P0[1] = ptclOrigin[1];
        ptclSeg.P1[0] = ptclDest[0];
        ptclSeg.P1[1] = ptclDest[1];

        struct {
          int isect[3] = {0,0,0};
          int isectEndPt[3] = {0,0,0};
          int exposed[3] = {0,0,0};
          int isLast[3] = {0,0,0};
        } edgeProps;
        for(int i = 0; i < 3; i++) {
          edgeProps.exposed[i] = side_is_exposed[edges[i]];
          edgeProps.isLast[i] = (lastEdge[pid] == edges[i]);
        }
        for(int i = 0; i < 3; i++) {
          const auto edgeVerts = o::gather_verts<2>(edge_verts, edges[i]);
          const auto edgeCoords = o::gather_vectors<2,2>(coords, edgeVerts);
          edgeSeg.P0[0] = edgeCoords(0,0);
          edgeSeg.P0[1] = edgeCoords(1,0);
          edgeSeg.P1[0] = edgeCoords(0,1);
          edgeSeg.P1[1] = edgeCoords(1,1);
          Point2D a,b;
          edgeProps.isect[i] = (intersect2D_2Segments(ptclSeg,edgeSeg,&a,&b) == 1);
          const auto isectEndPt0 = o::are_close(ptclSeg.P0[0],a.x) &&
                                   o::are_close(ptclSeg.P0[1],a.y);
          const auto isectEndPt1 = o::are_close(ptclSeg.P1[0],a.x) &&
                                   o::are_close(ptclSeg.P1[1],a.y);
          edgeProps.isectEndPt[i] = isectEndPt0 || isectEndPt1;

        }
        auto nextEdge = 0;
        // first, check for an exposed edge that is intersected
        for(int i = 0; i < 3; i++) {
          const auto check = (
              edgeProps.isect[i] &&
              edgeProps.exposed[i] &&
              !edgeProps.isLast[i] );
          //nextEdge is zero when an edge is not found, there is an edge 0 so add 1 to the ids
          if(check) {
            nextEdge = edges[i]+1;
          }
        }
        // next, if we have not found an exposed edge,
        //  check for an internal edge that is intersected
        //  at a point other than one of the end points
        for(int i = 0; i < 3; i++) {
          const auto check = (
              !nextEdge &&
              edgeProps.isect[i] &&
              !edgeProps.isectEndPt[i] &&
              !edgeProps.isLast[i] );
          if(check) {
            nextEdge = edges[i]+1;
          }
        }
        // last, if we have not found an edge, then select
        //  one of the end point intersected internal edges
        for(int i = 0; i < 3; i++) {
          const auto check = (
              !nextEdge &&
              edgeProps.isect[i] &&
              !edgeProps.isLast[i] );
          if(check) {
            nextEdge = edges[i]+1;
          }
        }
        if(!nextEdge) {
          printf("ptcl %d elm %d src %f %f dest %f %f\n",
              ptcl, searchElm,
              ptclSeg.P0[0], ptclSeg.P0[1],
              ptclSeg.P1[0], ptclSeg.P1[1]);
        }
        assert(nextEdge);
        lastEdge[pid] = nextEdge-1; //remove 1 to get the 0-based edge id
      } //if active particle
    };
    scs->parallel_for(checkAdjElm, "pumipic_checkAdjElm");

    auto checkExposedEdges = SCS_LAMBDA(const int& e, const int& pid, const int& mask) {
      if( mask > 0 && !ptcl_done[pid] ) {
        auto searchElm = elem_ids[pid];
        auto ptcl = pid_d(pid);
        auto bridge = lastEdge[pid];
        auto exposed = side_is_exposed[bridge];
        ptcl_done[pid] = exposed;
        elem_ids_next[pid] = -1; //leaves domain if exposed
      }
    };
    scs->parallel_for(checkExposedEdges, "pumipic_checkExposedEdges");

    auto e2f_vals = edges2faces.ab2b; // CSR value array
    auto e2f_offsets = edges2faces.a2ab; // CSR offset array, index by mesh edge ids
    auto setNextElm = SCS_LAMBDA(const int& e, const int& pid, const int& mask) {
      if( mask > 0 && !ptcl_done[pid] ) {
        auto searchElm = elem_ids[pid];
        auto ptcl = pid_d(pid);
        auto bridge = lastEdge[pid];
        auto e2f_first = e2f_offsets[bridge];
        auto e2f_last = e2f_offsets[bridge+1];
        auto upFaces = e2f_last - e2f_first;
        assert(upFaces==2);
        auto faceA = e2f_vals[e2f_first];
        auto faceB = e2f_vals[e2f_first+1];
        assert(faceA != faceB);
        assert(faceA == searchElm || faceB == searchElm);
        auto nextElm = (faceA == searchElm) ? faceB : faceA;
        elem_ids_next[pid] = nextElm;
      }
    };
    scs->parallel_for(setNextElm, "pumipic_setNextElm");

    found = true;
    auto cp_elm_ids = OMEGA_H_LAMBDA( o::LO i) {
      elem_ids[i] = elem_ids_next[i];
    };
    o::parallel_for(elem_ids.size(), cp_elm_ids, "copy_elem_ids");

    o::LOs ptcl_done_r(ptcl_done);
    auto minFlag = o::get_min(ptcl_done_r);
    if(minFlag == 0)
      found = false;
    ++loops;

    if(looplimit && loops >= looplimit) {
      fprintf(stderr, "ERROR:loop limit %d exceeded\n", looplimit);
      break;
    }
  }
  Kokkos::Profiling::popRegion();
  return found;
}

} //namespace
#endif //define


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
    printf(" old:bccvals %g %g %g %g vol %g \n", vals[0]/6.0, vals[1]/6.0, 
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
    printf(" bccvals %g %g %g %g vol %g \n", vals[0], vals[1], vals[2], vals[3],vol);
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

  if(std::abs(area) < 1e-20) { //TODO
    printf("area is too small \n");
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
      printf("LTintX Surface normal is flipped\n");
  }
  auto snorm_unit = Omega_h::normalize(normv);
  Omega_h::Real dist2plane = o::inner_product(abc[0] - origin, snorm_unit);
  auto plane2dest = dest - abc[0];
  Omega_h::Real proj_end = o::inner_product(snorm_unit, plane2dest);
  if(debug)
    printf("LTintX dist2plane %.10f pro_end %.10f\n", dist2plane, proj_end);  
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
        printf("LTintX Found %d bcc+ %d par_t= %.10f dist2plane= %.10f "
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

template < class ParticleType, typename Segment3d, typename SegmentInt >
bool search_mesh_3d(o::Mesh& mesh, // (in) mesh
    ParticleStructure< ParticleType >* ptcls, // (in) particle structure
    Segment3d x_ps_d, // (in) starting particle positions
    Segment3d xtgt_ps_d, // (in) target particle positions
    SegmentInt pid_d, // (in) particle ids
    o::Write<o::LO>& elem_ids, // (out) parent element ids for the target positions
    o::Write<o::Real>& xpoints_d, // (out) particle-boundary intersection points
    o::Write<o::LO>& xface_d, // (out) face ids of boundary-intersecting points
    int looplimit=0, int debug=0) {
  const auto btime = pumipic_prebarrier();
  Kokkos::Profiling::pushRegion("pumpipic_search_mesh3d");
  Kokkos::Profiling::pushRegion("pumpipic_search_mesh_Init");

  Kokkos::Timer timer;
  const o::Real tol = 1.0e-20;
  int rank, comm_size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &comm_size);

  Kokkos::Profiling::pushRegion("pumpipic_search_mesh_omegah");
  const auto side_is_exposed = mark_exposed_sides(&mesh);
  const auto mesh2verts = mesh.ask_elem_verts();
  const auto coords = mesh.coords();
  const auto face_verts =  mesh.ask_verts_of(2);
  const auto elem_faces = mesh.ask_down(3, 2).ab2b;
  const auto dual_elems = mesh.ask_dual().ab2b;
  const auto dual_faces = mesh.ask_dual().a2ab;
  const auto psCapacity = ptcls->capacity();
  Kokkos::Profiling::popRegion();

  Kokkos::Profiling::pushRegion("pumpipic_ptcl-done_elem_ids");
  // ptcl_done[i] = 2 : particle i has hit a boundary or reached its destination
  o::Write<o::LO> ptcl_done(psCapacity, 1, "ptcl_done");
  // store the next parent for each particle
  o::Write<o::LO> elem_ids_next(psCapacity,-1, "elem_ids_next");
  Kokkos::Profiling::popRegion();

  auto fill = PS_LAMBDA(const int& e, const int& pid, const int& mask) {
    if(mask > 0) {
      elem_ids[pid] = e;
      ptcl_done[pid] = 0;
    } else {
      elem_ids[pid] = -1;
      ptcl_done[pid] = 2;
    }
  };
  parallel_for(ptcls, fill, "searchMesh_fill_elem_ids");

  auto checkParent = PS_LAMBDA(const int& e, const int& pid, const int& mask) {
    if( mask > 0) {
      const auto orig = makeVector3(pid, x_ps_d);
      if(!isPointWithinElemTet(mesh2verts, coords, orig, e, tol)) {
        if(debug)
          printf("Search1: ptcl %d not_in parent_element %d :pos %g %g %g \n", 
            pid_d(pid), e, orig[0], orig[1], orig[2]);
        OMEGA_H_CHECK(false);
      }
    }
  };
  parallel_for(ptcls, checkParent, "pumipic_checkParent");
  Kokkos::Profiling::popRegion();
  bool found = false;
  int loops = 0;
  
  //debug
  int nloops = 5;
  int lsize = 5;
  int hsize = 1;
  int nl = nloops*lsize;
  if(debug)
    hsize = psCapacity;
  auto el_hist = o::Write<o::LO>(hsize*nl, -2);

  while(!found) {
    auto checkCurrentElm = PS_LAMBDA(const int& e, const int& pid, const int& mask) {
      if( mask > 0 && !ptcl_done[pid] ) {
        const auto searchElm = elem_ids[pid];
        OMEGA_H_CHECK(searchElm >= 0);
        const auto dest = makeVector3(pid, xtgt_ps_d);
        auto inParent = isPointWithinElemTet(mesh2verts, coords, dest, searchElm, tol);
        ptcl_done[pid] = (inParent) ? 2:0;
        //if ptcl not done, this will be reset below
        elem_ids_next[pid] = searchElm;
        if(debug && loops<nloops) {
          el_hist[pid*nl+lsize*loops] = ptcl_done[pid];
          el_hist[pid*nl+lsize*loops+1] = elem_ids_next[pid];
        }
      }
    };
    parallel_for(ptcls, checkCurrentElm, "pumipic_checkCurrentElm");

    auto findIntersection = PS_LAMBDA(const int& e, const int& pid, const int& mask) {
      if( mask > 0 && ptcl_done[pid]<2 ) {
        const auto searchElm = elem_ids[pid];
        OMEGA_H_CHECK(searchElm >= 0);
        const auto tetv2v = o::gather_verts<4>(mesh2verts, searchElm);
        const auto dest = makeVector3(pid, xtgt_ps_d);
        const auto orig = makeVector3(pid, x_ps_d);
        const auto face_ids = o::gather_down<4>(elem_faces, searchElm);
        auto dual_elem_id = dual_faces[searchElm];
        int adj_id = -1, ind_exp = -1;
        auto projd = o::zero_vector<4>(); //not used
        auto xpts = o::zero_vector<3>();
        for(int fi=0; fi<4; ++fi) {
          const auto face_id = face_ids[fi];
          auto xpoint = o::zero_vector<3>();
          const auto fv2v = o::gather_verts<3>(face_verts, face_id);
          const auto face = gatherVectors3x3(coords, fv2v);
          const auto flip = isFaceFlipped(fi, fv2v, tetv2v);
          const auto det = line_triangle_intx_simple(face, orig, dest, xpoint, 
            projd[fi], flip, tol);//, debug);
          auto exposed = side_is_exposed[face_id];
          if(det && exposed) {
            ind_exp = fi;
            for(o::LO i=0; i<3; ++i)
              xpts[i] = xpoint[i];
          }
          if(det && !exposed)
            adj_id = dual_elem_id;
          if(!exposed)
            ++dual_elem_id;
        } //for

        //wall collision
        if(ind_exp >= 0) {
          for(o::LO i=0; i<3; ++i)
            xpoints_d[pid*3+i] = xpts[i];
          xface_d[pid] = face_ids[ind_exp];
          if(debug)
            printf("Search: ptcl %d hit boundary %d  to-be stopped/reflected\n", 
              pid_d(pid), xface_d[pid]);
          elem_ids_next[pid] = -1;
          ptcl_done[pid] = 2;
          if(debug && loops<nloops)
            el_hist[pid*nl+lsize*loops+3] = elem_ids_next[pid];
        }

        //interior
        if(adj_id >= 0) {
          elem_ids_next[pid] = dual_elems[adj_id];
          ptcl_done[pid] = 1; // reset below to 0/non-zero
          if(debug && loops<nloops) {
            el_hist[pid*nl+lsize*loops+2] = adj_id;
            el_hist[pid*nl+lsize*loops+3] = elem_ids_next[pid];
          }
        }
      }
    };
    parallel_for(ptcls, findIntersection, "pumipic_findIntersection");

    auto processUndetected = PS_LAMBDA(const int& e, const int& pid, const int& mask) {
      auto done = ptcl_done[pid];
      ptcl_done[pid] = (done <2) ? 0: 2;
      if( mask > 0 && done < 1) {
        const auto searchElm = elem_ids[pid];
        OMEGA_H_CHECK(searchElm >= 0);
        const auto tetv2v = o::gather_verts<4>(mesh2verts, searchElm);
        const auto dest = makeVector3(pid, xtgt_ps_d);
        const auto orig = makeVector3(pid, x_ps_d);
        const auto face_ids = o::gather_down<4>(elem_faces, searchElm);
        o::Real projd[4] = {-1,-1,-1,-1};
        auto xpoints = o::zero_vector<12>();
        for(int fi=0; fi<4; ++fi) {
          const auto face_id = face_ids[fi];
          auto xpoint = o::zero_vector<3>();
          const auto fv2v = o::gather_verts<3>(face_verts, face_id);
          const auto face = gatherVectors3x3(coords, fv2v);
          const auto flip = isFaceFlipped(fi, fv2v, tetv2v);
          const auto det = line_triangle_intx_simple(face, orig, dest, xpoint, 
            projd[fi], flip, tol);//, debug);
          for(int i=0; i<3; ++i) 
            xpoints[fi*3+i] = xpoint[i];
        }
        const o::LO max_ind = max_index(projd, 4);
        OMEGA_H_CHECK(max_ind >= 0);
        const auto face_id = face_ids[max_ind];
        const auto exposed = side_is_exposed[face_id];
        if(exposed) {
          elem_ids_next[pid] = -1;
          if(debug && loops<nloops)
            el_hist[pid*nl+lsize*loops+4] = elem_ids_next[pid];          
          for(o::LO i=0; i<3; ++i)
            xpoints_d[pid*3+i] = xpoints[max_ind*3+i];            
          xface_d[pid] = face_id;
          if(debug)
            printf("Search: ptcl %d hit boundary tobe reflected/stopped\n", pid_d(pid));
          ptcl_done[pid] = 2;
        } else {
          elem_ids_next[pid] = dual_elems[face_id];
          if(debug && loops<nloops)
            el_hist[pid*nl+lsize*loops+4] = elem_ids_next[pid];
        }
      }
    };
    parallel_for(ptcls, processUndetected, "pumipic_processUndetected");
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
      auto ptclsNotFound = PS_LAMBDA(const int& e, const int& pid, const int& mask) {
        if( mask > 0 && !ptcl_done[pid] ) {
          auto elm = elem_ids[pid];
          auto ptcl = pid_d(pid);
          const auto dest = makeVector3(pid, xtgt_ps_d);
          const auto orig = makeVector3(pid, x_ps_d);
          printf("rank %d : el %d next_elm %d ptcl %d  %.15e %.15e %.15e "
            "=> %.15e %.15e %.15e \n", rank, e, elm, ptcl, orig[0], 
            orig[1], orig[2], dest[0], dest[1],dest[2]);
          if(debug)
            for(int il=0; il<nloops; ++il) {
              auto nn = pid*nloops*lsize+lsize*il;
              printf(" pid %d  loop %d el_hist  : %d %d %d %d %d\n", pid, il, 
               el_hist[nn], el_hist[nn+1], el_hist[nn+2], el_hist[nn+3], el_hist[nn+4]);
            }
        }
      };
      parallel_for(ptcls, ptclsNotFound, "ptclsNotFound");
      fprintf(stderr, "ERROR:loop limit %d exceeded\n", looplimit);
      break;
    }
  } //while
  Kokkos::Profiling::popRegion(); //whole
  //fprintf(stderr, "loop-time seconds %f\n", timer.seconds()); 
  pumipic::RecordTime("Search Mesh 3d", timer.seconds(), btime);
  return found;   
}


template < class ParticleType, typename Segment3d, typename SegmentInt >
bool search_mesh(o::Mesh& mesh, ParticleStructure< ParticleType >* ptcls,
  Segment3d x_ps_d, Segment3d xtgt_ps_d, SegmentInt pid_d,
  o::Write<o::LO>& elem_ids, o::Write<o::Real>& xpoints_d,
  o::Write<o::LO>& xface_d, int looplimit=0, int debug=0) {
           
  o::Real tol = 1.0e-10;
  
  Kokkos::Profiling::pushRegion("pumipic_search");
  Kokkos::Profiling::pushRegion("pumipic_search_Init");
  const auto side_is_exposed = mark_exposed_sides(&mesh);
  const auto mesh2verts = mesh.ask_elem_verts();
  const auto coords = mesh.coords();
  const auto face_verts =  mesh.ask_verts_of(2);
  const auto dual_r2fs = mesh.ask_down(3, 2).ab2b;
  const auto dual_elems = mesh.ask_dual().ab2b;
  const auto dual_faces = mesh.ask_dual().a2ab;
  const auto psCapacity = ptcls->capacity();

  // ptcl_done[i] = 1 : particle i has hit a boundary or reached its destination
  o::Write<o::LO> ptcl_done(psCapacity);//, 1, "ptcl_done");
  // store the next parent for each particle
  o::Write<o::LO> elem_ids_next(psCapacity);//,-1);
  auto fill = PS_LAMBDA(const int& e, const int& pid, const int& mask) {
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
  parallel_for(ptcls, fill, "searchMesh_fill_elem_ids");
  Kokkos::Profiling::popRegion();
  bool found = false;
  int loops = 0;
  while(!found) {
    if(debug) {
      fprintf(stderr, "------------ %d ------------\n", loops);
    }
    //pid is same for a particle between iterations in this while loop
    auto lamb = PS_LAMBDA(const int& e, const int& pid, const int& mask) {
      //inactive particle that is still moving to its target position
      if( mask > 0 && !ptcl_done[pid] ) {
        auto elmId = elem_ids[pid];
        OMEGA_H_CHECK(elmId >= 0);
        auto tetv2v = o::gather_verts<4>(mesh2verts, elmId);
        auto M = gatherVectors4x3(coords, tetv2v);
        if(debug)
          printf("pid %d in element %d\n", pid, elmId);
        auto dest = makeVector3(pid, xtgt_ps_d);
        auto orig = makeVector3(pid, x_ps_d);
        o::Vector<4> bcc;
        if(loops == 0) {
          //make sure particle origin is in initial element
          find_barycentric_tet(M, orig, bcc);
          if(!all_positive(bcc, tol)) {
            printf("Warning: Particle not in this element at loops=0"
              "\tpid %d elem %d\n", pid, elmId);
            OMEGA_H_CHECK(false);
          }
        }
        
        auto ptcl = pid_d(pid);
        bool intersected = false;
        find_barycentric_tet(M, dest, bcc);
        // TODO tolerance
        if(all_positive(bcc, tol)) {
          if(debug)
            printf("ptcl %d is in destination elm %d\n", ptcl, elmId);
          elem_ids_next[pid] = elmId;
          ptcl_done[pid] = 1;
        } else {
          if(debug)
            printf("ptcl %d  elemId %d checking adj elms:\n", ptcl, elmId);
          auto dproj = o::zero_vector<4>();
          auto xpoints = o::zero_vector<12>();
          o::LO exposed_faces[4];
          o::LO xface_ids[4];
          o::LO min_bcc_elem = -1;
          dproj[0] = dproj[1] = dproj[2] = dproj[3] = -1;
          //get element ID
          auto dual_elem_id = dual_faces[elmId];
          const auto beg_face = elmId *4;
          const auto end_face = beg_face +4;
          o::LO findex = 0;
          for(auto iface = beg_face; iface < end_face; ++iface) {
            const auto face_id = dual_r2fs[iface];
            auto xpoint = o::zero_vector<3>();
            o::LO exposed = side_is_exposed[face_id];
            exposed_faces[findex] = exposed;
            xface_ids[findex] = face_id;
            auto fv2v = o::gather_verts<3>(face_verts, face_id);
            const auto face = gatherVectors3x3(coords, fv2v);
            o::LO matInd1 = getFaceMap(findex*2);
            o::LO matInd2 = getFaceMap(findex*2+1);
            bool flip = true;
            if(fv2v[1] == tetv2v[matInd1] && fv2v[2] == tetv2v[matInd2])
              flip = false;
            intersected = line_triangle_intx_simple(face, orig, dest, xpoint, 
              dproj[findex], flip, tol);
            for(o::LO i=0; i<3; ++i)
              xpoints[findex*3+i] = xpoint[i];

            if(debug) {
              printf("\t :ptcl %d elmId %d faceid %d flipped %d exposed %d intersected %d"
              " findex %d\n", ptcl, elmId, face_id, flip, exposed, intersected, findex);
              for(int i=0; i<3; ++i)
               printf("\t ptcl %d face:%d %.15f %.15f %.15f\n", 
                ptcl, i, face[i][0], face[i][1], face[i][2]);
              printf("\t ptcl: %d orig,dest: %.15f %.15f %.15f %.15f %.15f %.15f \n", 
                ptcl, orig[0], orig[1], orig[2], dest[0],dest[1],dest[2]);
            }
            if(intersected && exposed) {
              ptcl_done[pid] = 1;
              for(o::LO i=0; i<3; ++i)
                xpoints_d[pid*3+i] = xpoint[i];
              xface_d[pid] = face_id;
              elem_ids_next[pid] = -1;
              if(debug)
                printf("\t ptcl %d e %d faceid %d intersected and exposed, next parent "
                  "elm %d findex %d\n", ptcl, elmId, face_id, elem_ids_next[pid], findex);
              break;
            } else if(intersected && !exposed) {
              auto adj_elem  = dual_elems[dual_elem_id];
              elem_ids_next[pid] = adj_elem;
              if(debug) {
                printf("\t ptcl %d e %d faceid %d intersected and !exposed, next parent "
                      "elm %d findex %d\n", ptcl, elmId, face_id, elem_ids_next[pid], findex);
              }
              break;
            }
            o::LO min_ind = min_index(bcc, 4);

            // save next element based on the smallest BCC,
            if(!exposed) {
              if(debug)
                printf("\t ptcl %d e %d faceid %d findex %d !intersected and !exposed\n",
                ptcl, elmId, face_id, findex);
              o::LO min_ind = min_index(bcc, 4);
              if(findex == min_ind) {
                min_bcc_elem = dual_elems[dual_elem_id];
              }
              ++dual_elem_id;
            }
            ++findex;
          } //for iface

          if(!intersected) {
            printf("\t ptcl %d e %d not intersected; using max dproj\n", ptcl, elmId);
            o::LO max_ind = max_index(dproj, 4);
            if(dproj[max_ind]>=0) {
              auto fid = xface_ids[max_ind];
              if(exposed_faces[max_ind]) {
                elem_ids_next[pid] = -1;
                for(o::LO i=0; i<3; ++i)
                  xpoints_d[pid*3+i] = xpoints[max_ind*3+i];            
                xface_d[pid] = fid;
                ptcl_done[pid] = 1;
              } else { //if(min_bcc_elem >= 0) {
                elem_ids_next[pid] = dual_elems[fid]; //min_bcc_elem;
                if(debug)
                  printf("\t ptcl %d e %d elem_ids_next %d min_bcc_elem %d\n", ptcl, 
                   elmId, elem_ids_next[pid], min_bcc_elem);
              }
            } else {
              // current elem, but bcc failed to detect it on face/corner
              printf("WARNING: particle %d leaked from e %d \n", ptcl, elmId);
              elem_ids_next[pid] = -1;
              ptcl_done[pid] = 1;
            }
          }
        } //else not in current element
      } //if active particle
    };

    parallel_for(ptcls, lamb, "adj_search");

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
      //if (debug)
        fprintf(stderr, "ERROR:loop limit %d exceeded\n", looplimit);
      break;
    }
  } //while
  
  if(debug)
    fprintf(stderr, "\t: loops %d\n", loops);
  Kokkos::Profiling::popRegion();
  return found;
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
      printf("bcc %.15f field %.15f val %.15f\n", bcc[fi], fv4[fd][0], val);
      printf("interp: %g %d %g %g \n", bcc[fi]*fv4[fd][0], d, 
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
    printf("d's:: %0.3f %0.3f %0.3f %0.3f %0.3f %0.3f \n", d1, d2, d3, d4, d5, d6);

  return ptq;
}

template < class ParticleStruct, typename CurrentCoordView,
           typename TargetCoordView, typename SegmentInt>
bool search_mesh_2d(o::Mesh& mesh, // (in) mesh
                    ParticleStruct* ptcls, // (in) particle structure
                    CurrentCoordView x_ps_d, // (in) starting particle positions
                    TargetCoordView xtgt_ps_d, // (in) target particle positions
                    SegmentInt pid_d, // (in) particle ids
                    o::Write<o::LO> elem_ids, // (out) parent element ids for the target positions
                    int looplimit=0,  // (in) [optional] number of loops before giving up
                    bool debug = false) {

  const auto btime = pumipic_prebarrier();
  Kokkos::Profiling::pushRegion("pumpipic_search_mesh_2d");
  Kokkos::Timer timer;

  int rank, comm_size;
  MPI_Comm_rank(MPI_COMM_WORLD,&rank);
  MPI_Comm_size(MPI_COMM_WORLD,&comm_size);

  const auto faces2edges = mesh.ask_down(o::FACE, o::EDGE);
  const auto edges2faces = mesh.ask_up(o::EDGE, o::FACE);
  const auto side_is_exposed = mark_exposed_sides(&mesh);
  const auto faces2verts = mesh.ask_elem_verts();
  const auto coords = mesh.coords();
  const auto edge_verts =  mesh.ask_verts_of(o::EDGE);
  const auto faceEdges = faces2edges.ab2b;
  const auto triArea = measure_elements_real(&mesh);

  const auto psCapacity = ptcls->capacity();

  // ptcl_done[i] = 1 : particle i has hit a boundary or reached its destination
  o::Write<o::LO> ptcl_done(psCapacity, 1, "ptcl_done");
  // store the last crossed edge
  o::Write<o::LO> lastEdge(psCapacity,-1);
  const o::LO nelems = mesh.nelems();
  auto lamb = PS_LAMBDA(const int& e, const int& pid, const int& mask) {
    if(mask > 0) {
      if (elem_ids[pid] == -1) {
        elem_ids[pid] = e;
      }
      ptcl_done[pid] = 0;
      // handle situations where particle may be outside the simulation domain
      // after field-following based operation
      if (elem_ids[pid] == -nelems) {
        elem_ids[pid] = -1;
        ptcl_done[pid] = 1;
      }
    } else {
      elem_ids[pid] = -1;
      ptcl_done[pid] = 1;
    }
  };
  parallel_for(ptcls, lamb);

  Omega_h::Write<o::LO> numNotInElem(1, 0);
  auto checkParent = PS_LAMBDA(const int& e, const int& pid, const int& mask) {
    //inactive particle that is still moving to its target position
    if( mask > 0 && !ptcl_done[pid] ) {
      auto searchElm = elem_ids[pid];
      auto ptcl = pid_d(pid);
      OMEGA_H_CHECK(searchElm >= 0);
      auto faceVerts = o::gather_verts<3>(faces2verts, searchElm);
      const auto faceCoords = o::gather_vectors<3,2>(coords, faceVerts);
      auto ptclOrigin = makeVector2(pid, x_ps_d);
      Omega_h::Vector<3> faceBcc;
      barycentric_tri(triArea, faceCoords, ptclOrigin, faceBcc, searchElm);
      //Note: particles are not necessarily in the correct element to start
      // with, due to field-following based particle->mesh association.
    } //if active
  };
  ps::parallel_for(ptcls, checkParent);
  Omega_h::HostWrite<o::LO> numNotInElem_h(numNotInElem);
  if (numNotInElem_h[0] > 0) {
    fprintf(stderr, "[WARNING] Rank %d: %d particles are not located in their "
            "starting elements. Deleting them...\n", rank, numNotInElem_h[0]);
  }
  bool found = false;
  int loops = 0;
  while(!found) {
    auto checkCurrentElm = PS_LAMBDA(const int& e, const int& pid, const int& mask) {
      //active particle that is still moving to its target position
      if( mask > 0 && !ptcl_done[pid] ) {
        auto searchElm = elem_ids[pid];
        OMEGA_H_CHECK(searchElm >= 0);
        const auto edges = o::gather_down<3>(faceEdges, searchElm);
        const auto faceVerts = o::gather_verts<3>(faces2verts, searchElm);
        const auto faceCoords = o::gather_vectors<3,2>(coords, faceVerts);
        const auto ptclDest = makeVector2(pid, xtgt_ps_d);
        const auto ptclOrigin = makeVector2(pid, x_ps_d);
        Omega_h::Vector<3> faceBcc;
        barycentric_tri(triArea, faceCoords, ptclDest, faceBcc, searchElm);
        auto isDestInParentElm = all_positive(faceBcc);
        ptcl_done[pid] = isDestInParentElm;
        const int idx = min3(faceBcc);
        lastEdge[pid] = edges[idx];
      }
    };
    parallel_for(ptcls, checkCurrentElm);

    auto checkExposedEdges = PS_LAMBDA(const int& e, const int& pid, const int& mask) {
      if( mask > 0 && !ptcl_done[pid] ) {
        auto searchElm = elem_ids[pid];
        assert(lastEdge[pid] != -1);
        auto bridge = lastEdge[pid];
        auto exposed = side_is_exposed[bridge];
        ptcl_done[pid] = exposed;
        elem_ids[pid] = exposed ? -1 : elem_ids[pid]; //leaves domain if exposed
      }
    };
    parallel_for(ptcls, checkExposedEdges, "pumipic_checkExposedEdges");

    auto e2f_vals = edges2faces.ab2b; // CSR value array
    auto e2f_offsets = edges2faces.a2ab; // CSR offset array, index by mesh edge ids
    auto setNextElm = PS_LAMBDA(const int& e, const int& pid, const int& mask) {
      if( mask > 0 && !ptcl_done[pid] ) {
        auto searchElm = elem_ids[pid];
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
        elem_ids[pid] = nextElm;
      }
    };
    parallel_for(ptcls, setNextElm, "pumipic_setNextElm");

    found = true;
    o::LOs ptcl_done_r(ptcl_done);
    auto minFlag = o::get_min(ptcl_done_r);
    if(minFlag == 0)
      found = false;
    ++loops;

    if(looplimit && loops >= looplimit) {
      Omega_h::Write<o::LO> numNotFound(1,0);
      auto ptclsNotFound = PS_LAMBDA(const int& e, const int& pid, const int& mask) {
        if( mask > 0 && !ptcl_done[pid] ) {
          auto searchElm = elem_ids[pid];
          auto ptcl = pid_d(pid);
          const auto ptclDest = makeVector2(pid, xtgt_ps_d);
          const auto ptclOrigin = makeVector2(pid, x_ps_d);
          if (debug) {
            printf("rank %d elm %d ptcl %d notFound %.15f %.15f to %.15f %.15f\n",
                   rank, searchElm, ptcl, ptclOrigin[0], ptclOrigin[1],
                   ptclDest[0], ptclDest[1]);
          }
          elem_ids[pid] = -1;
          Kokkos::atomic_add(&(numNotFound[0]), 1);
        }
      };
      ps::parallel_for(ptcls, ptclsNotFound, "ptclsNotFound");
      Omega_h::HostWrite<o::LO> numNotFound_h(numNotFound);
      fprintf(stderr, "ERROR:Rank %d: loop limit %d exceeded. %d particles were "
              "not found. Deleting them...\n", rank, looplimit, numNotFound_h[0]);
      break;
    }
  }

  RecordTime("pumipic search_2d", timer.seconds(), btime);
  char buffer[1024];
  sprintf(buffer, "%d pumipic search_2d loops %d", rank, loops);
  PrintAdditionalTimeInfo(buffer, 1);
  Kokkos::Profiling::popRegion();
  return found;
}

} //namespace
#endif //define

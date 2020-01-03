#ifndef PUMIPIC_ADJACENCY_HPP
#define PUMIPIC_ADJACENCY_HPP

#include <iostream>
#include <cstdlib>
#include <ctime>
#include "Omega_h_for.hpp"
#include "Omega_h_adj.hpp"
#include "Omega_h_element.hpp"
#include "Omega_h_shape.hpp"

#include <SellCSigma.h>
#include <SCS_Macros.h>

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
*/

#define TriVerts 3
#define TriDim 2
//compute the area coordinates formed by each edge of searchElm
//the coordinates are returned in the order of the edges bounding
//searchElm
OMEGA_H_DEVICE void barycentric_tri(
    const o::Reals triArea,
    const o::Matrix<TriDim, TriVerts> &faceCoords,
    const o::Vector<TriDim> &pos,
    o::Vector<TriVerts> &bcc,
    const int searchElm) {
  const auto parent_area = triArea[searchElm];
  for(int i=0; i<3; i++) {
    const auto kIdx = simplex_down_template(o::FACE, o::EDGE, i, 0);
    const auto lIdx = simplex_down_template(o::FACE, o::EDGE, i, 1);
    const auto kxy = faceCoords[kIdx];
    const auto lxy = faceCoords[lIdx];
    o::Few<o::Vector<2>, 2> tri;
    tri[0] = lxy - kxy;
    tri[1] = pos - kxy;
    const auto area = o::triangle_area_from_basis(tri);
    bcc[i] = area/parent_area;
  }
}

// BC coords are not in order of its corresp. opp. vertexes. Bccoord of tet(iface, xpoint)
//TODO Warning: Check opposite_template use in this before using
OMEGA_H_DEVICE bool find_barycentric_tet( const Omega_h::Matrix<DIM, 4> &Mat,
     const Omega_h::Vector<DIM> &pos, Omega_h::Vector<4> &bcc, 
     bool debug=false) {
  for(Omega_h::LO i=0; i<4; ++i) 
    bcc[i] = -1;

  Omega_h::Real vals[4];
  Omega_h::Few<Omega_h::Vector<DIM>, 3> abc;
  for(Omega_h::LO iface=0; iface<4; ++iface) {
    get_face_from_face_index_of_tet(Mat, iface, abc);
    auto vab = abc[1] - abc[0]; //b - a;
    auto vac = abc[2] - abc[0]; //c - a;
    auto vap = pos - abc[0]; // p - a;
    vals[iface] = osh_dot(vap, Omega_h::cross(vac, vab)); //ac, ab
  }
  //volume using bottom face=0
  get_face_from_face_index_of_tet(Mat, 0, abc);
  auto vtx3 = Omega_h::simplex_opposite_template(DIM, FDIM, 0);
  OMEGA_H_CHECK(3 == vtx3);
  // abc in order, for bottom face: M[0], M[2](=abc[1]), M[1](=abc[2])
  auto cross_ac_ab = Omega_h::cross(abc[2]-abc[0], abc[1]-abc[0]);
  Omega_h::Real vol6 = osh_dot(Mat[vtx3]-Mat[0], cross_ac_ab);
  // use o::tet_volume_from_basis()
  if(debug)
    print_few_vectors(abc);
  Omega_h::Real inv_vol = 0.0;
  if(vol6 > EPSILON) // TODO tolerance
    inv_vol = 1.0/vol6;
  else {
    return 0;
  }
  //bcc[0] for face0 corresp to its opp vtx, so on.
  for(int i=0; i<4; ++i)
    bcc[i] = inv_vol * vals[i]; 

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
  Omega_h::Real area = osh_dot(norm, cross);

  if(std::abs(area) < 1e-9) { //TODO
    printf("area is too small \n");
    return 0;
  }
  Omega_h::Real fac = 1/(area*2.0);
  bc[0] = fac * osh_dot(norm, Omega_h::cross(b-a, xpoint-a));
  bc[1] = fac * osh_dot(norm, Omega_h::cross(c-b, xpoint-b));
  bc[2] = fac * osh_dot(norm, Omega_h::cross(xpoint-a, c-a));

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
  Omega_h::Real dist2plane = osh_dot(abc[0] - origin, snorm_unit);
  auto plane2dest = dest - abc[0];
  Omega_h::Real proj_end = osh_dot(snorm_unit, plane2dest);
  if(debug)
    printf("LTintX dist2plane %.10f pro_end %.10f\n", dist2plane, proj_end);  
  // equal required if tol=0 is used
  if(dist2plane >= -tol && proj_end >= -tol) {
    dproj = osh_dot(line, snorm_unit);
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

template < class ParticleType >
bool search_mesh_3d(o::Mesh& mesh, // (in) mesh
    ps::SellCSigma< ParticleType >* scs, // (in) particle structure
    Segment3d x_scs_d, // (in) starting particle positions
    Segment3d xtgt_scs_d, // (in) target particle positions
    SegmentInt pid_d, // (in) particle ids
    o::Write<o::LO>& elem_ids, // (out) parent element ids for the target positions
    o::Write<o::Real>& xpoints_d, // (out) particle-boundary intersection points
    o::Write<o::LO>& xface_d, // (out) face ids of boundary-intersecting points
    int looplimit=0, int debug=0) {
  Kokkos::Profiling::pushRegion("pumpipic_search_mesh3d");
  Kokkos::Profiling::pushRegion("pumpipic_search_mesh_Init");

  Kokkos::Timer timer;
  const o::Real tol = 1.0e-10;
  int rank, comm_size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &comm_size);

  Kokkos::Profiling::pushRegion("pumpipic_search_mesh_omegah");
  const auto side_is_exposed = mark_exposed_sides(&mesh);
  const auto mesh2verts = mesh.ask_elem_verts();
  const auto coords = mesh.coords();
  const auto face_verts =  mesh.ask_verts_of(2);
  const auto elemFaces = mesh.ask_down(3, 2).ab2b;
  const auto dual_elems = mesh.ask_dual().ab2b;
  const auto dual_faces = mesh.ask_dual().a2ab;
  const auto scsCapacity = scs->capacity();
  Kokkos::Profiling::popRegion();

  Kokkos::Profiling::pushRegion("pumpipic_ptcl-done_elem_ids");
  // ptcl_done[i] = 2 : particle i has hit a boundary or reached its destination
  o::Write<o::LO> ptcl_done(scsCapacity, 1, "ptcl_done");
  // store the next parent for each particle
  o::Write<o::LO> elem_ids_next(scsCapacity,-1, "elem_ids_next");
  Kokkos::Profiling::popRegion();

  auto fill = SCS_LAMBDA(const int& e, const int& pid, const int& mask) {
    if(mask > 0) {
      elem_ids[pid] = e;
      ptcl_done[pid] = 0;
      auto ptcl = pid_d(pid);
    } else {
      elem_ids[pid] = -1;
      ptcl_done[pid] = 2;
    }
  };
  scs->parallel_for(fill, "searchMesh_fill_elem_ids");

  auto checkParent = SCS_LAMBDA(const int& e, const int& pid, const int& mask) {
    if( mask > 0 && !ptcl_done[pid] ) {
      const auto searchElm = elem_ids[pid];
      OMEGA_H_CHECK(searchElm >= 0);
      const auto ptcl = pid_d(pid);
      const auto tetv2v = o::gather_verts<4>(mesh2verts, searchElm);
      const auto tetCoords = gatherVectors4x3(coords, tetv2v);
      const auto orig = makeVector3(pid, x_scs_d);
      auto bcc = o::zero_vector<4>();
      //make sure particle origin is in initial element
      find_barycentric_tet(tetCoords, orig, bcc);
      if(!all_positive(bcc, tol))
        OMEGA_H_CHECK(false);
    }
  };
  scs->parallel_for(checkParent, "pumipic_checkParent");

  Kokkos::Profiling::popRegion();

  bool found = false;
  int loops = 0;
  
  while(!found) {
    auto checkCurrentElm = SCS_LAMBDA(const int& e, const int& pid, const int& mask) {
      if( mask > 0 && !ptcl_done[pid] ) {
        const auto searchElm = elem_ids[pid];
        const auto ptcl = pid_d(pid);
        OMEGA_H_CHECK(searchElm >= 0);
        const auto tetv2v = o::gather_verts<4>(mesh2verts, searchElm);
        const auto tetCoords = gatherVectors4x3(coords, tetv2v);
        const auto dest = makeVector3(pid, xtgt_scs_d);
        auto bcc = o::zero_vector<4>();

        find_barycentric_tet(tetCoords, dest, bcc);
        const auto isDestInParentElm = all_positive(bcc, tol);
        auto done = (isDestInParentElm >0) ? 2:0;
        ptcl_done[pid] = done; //isDestInParentElm;
        elem_ids_next[pid] = searchElm; //if ptcl not done, this will be reset below
      }
    };
    scs->parallel_for(checkCurrentElm, "pumipic_checkCurrentElm");

    auto findIntersection = SCS_LAMBDA(const int& e, const int& pid, const int& mask) {
      if( mask > 0 && ptcl_done[pid]<2 ) {
        const auto searchElm = elem_ids[pid];
        const auto ptcl = pid_d(pid);
        OMEGA_H_CHECK(searchElm >= 0);
        const auto tetv2v = o::gather_verts<4>(mesh2verts, searchElm);
        const auto dest = makeVector3(pid, xtgt_scs_d);
        const auto orig = makeVector3(pid, x_scs_d);
        const auto face_ids = o::gather_down<4>(elemFaces, searchElm);
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
            projd[fi], flip, tol, debug);
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
            xpoints_d[ptcl*3+i] = xpts[i];
          xface_d[ptcl] = face_ids[ind_exp];
          elem_ids_next[pid] = -1;
          ptcl_done[pid] = 2; 
        }

        //interior
        if(adj_id >= 0) {
          elem_ids_next[pid] = dual_elems[adj_id];
          ptcl_done[pid] = 1; // reset below to 0/non-zero
        }
      }
    };
    scs->parallel_for(findIntersection, "pumipic_findIntersection");

    auto processUndetected = SCS_LAMBDA(const int& e, const int& pid, const int& mask) {
      auto done = ptcl_done[pid];
      ptcl_done[pid] = (done <2) ? 0: 2;
      if( mask > 0 && done < 1) {
        const auto ptcl = pid_d(pid);
        const auto searchElm = elem_ids[pid];
        OMEGA_H_CHECK(searchElm >= 0);
        const auto tetv2v = o::gather_verts<4>(mesh2verts, searchElm);
        const auto dest = makeVector3(pid, xtgt_scs_d);
        const auto orig = makeVector3(pid, x_scs_d);
        const auto face_ids = o::gather_down<4>(elemFaces, searchElm);
        o::Real projd[4] = {-1,-1,-1,-1};
        auto xpoints = o::zero_vector<12>();
        for(int fi=0; fi<4; ++fi) {
          const auto face_id = face_ids[fi];
          auto xpoint = o::zero_vector<3>();
          const auto fv2v = o::gather_verts<3>(face_verts, face_id);
          const auto face = gatherVectors3x3(coords, fv2v);
          const auto flip = isFaceFlipped(fi, fv2v, tetv2v);
          const auto det = line_triangle_intx_simple(face, orig, dest, xpoint, 
            projd[fi], flip, tol, debug);
          for(int i=0; i<3; ++i) 
            xpoints[fi*3+i] = xpoint[i];
        }
        const o::LO max_ind = max_index(projd, 4);
        OMEGA_H_CHECK(max_ind >= 0);
        const auto face_id = face_ids[max_ind];
        const auto exposed = side_is_exposed[face_id];
        if(exposed) {
          elem_ids_next[pid] = -1;
          for(o::LO i=0; i<3; ++i)
            xpoints_d[ptcl*3+i] = xpoints[max_ind*3+i];            
          xface_d[ptcl] = face_id;
          ptcl_done[pid] = 2;        
        } else {
          elem_ids_next[pid] = dual_elems[face_id];
        }
      }
    };
    scs->parallel_for(processUndetected, "pumipic_processUndetected");

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
      auto ptclsNotFound = SCS_LAMBDA(const int& e, const int& pid, const int& mask) {
        if( mask > 0 && !ptcl_done[pid] ) {
          auto searchElm = elem_ids[pid];
          auto ptcl = pid_d(pid);
          const auto ptclDest = makeVector3(pid, xtgt_scs_d);
          const auto ptclOrigin = makeVector3(pid, x_scs_d);
        }
      };
      scs->parallel_for(ptclsNotFound, "ptclsNotFound");
      fprintf(stderr, "ERROR:loop limit %d exceeded\n", looplimit);
      break;
    }
  } //while
  Kokkos::Profiling::popRegion(); //whole
  fprintf(stderr, "loop-time seconds %f\n", timer.seconds()); 
  return found;   
}


template < class ParticleType>
bool search_mesh(o::Mesh& mesh, ps::SellCSigma< ParticleType >* scs,
  Segment3d x_scs_d, Segment3d xtgt_scs_d, SegmentInt pid_d,
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
  const auto scsCapacity = scs->capacity();

  // ptcl_done[i] = 1 : particle i has hit a boundary or reached its destination
  o::Write<o::LO> ptcl_done(scsCapacity);//, 1, "ptcl_done");
  // store the next parent for each particle
  o::Write<o::LO> elem_ids_next(scsCapacity);//,-1);
  auto fill = SCS_LAMBDA(const int& e, const int& pid, const int& mask) {
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

  scs->parallel_for(fill, "searchMesh_fill_elem_ids");
  Kokkos::Profiling::popRegion();

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
        auto elmId = elem_ids[pid];
        OMEGA_H_CHECK(elmId >= 0);
        auto tetv2v = o::gather_verts<4>(mesh2verts, elmId);
        auto M = gatherVectors4x3(coords, tetv2v);
        if(debug)
          printf("pid %d in element %d\n", pid, elmId);
        auto dest = makeVector3(pid, xtgt_scs_d);
        auto orig = makeVector3(pid, x_scs_d);
        o::Vector<4> bcc;
        if(loops == 0) {
          //make sure particle origin is in initial element
          find_barycentric_tet(M, orig, bcc);
          if(!all_positive(bcc, tol)) {
            printf("Warning: Particle not in this element at loops=0"
              "\tpid %d elem %d\n", pid, elmId);
            print_osh_vector(orig, "orig");
            print_osh_vector(dest, "dest");
            print_osh_vector(bcc, "bcc");
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
                xpoints_d[ptcl*3+i] = xpoint[i];
              xface_d[ptcl] = face_id;
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
                  xpoints_d[ptcl*3+i] = xpoints[max_ind*3+i];            
                xface_d[ptcl] = fid;
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
    if(debug)
      printf("interp: %g %d %g %g \n", bcc[fi]*fv4[fd][0], d, 
        bcc[fi], fv4[fd][0]);
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

OMEGA_H_DEVICE void findTetCoords(const Omega_h::LOs &mesh2verts,
const Omega_h::Reals &coords, const Omega_h::LO elem, 
Omega_h::Matrix<DIM, 4> &mat) {
  const auto tetv2v = Omega_h::gather_verts<4>(mesh2verts, elem);
  mat = Omega_h::gather_vectors<4, 3>(coords, tetv2v);
}

OMEGA_H_DEVICE void findBCCoordsInTet(const Omega_h::Reals &coords, 
  const Omega_h::LOs &mesh2verts, const Omega_h::Vector<3> &xyz, 
  const Omega_h::LO elem, Omega_h::Vector<4> &bcc) {
  Omega_h::Matrix<3, 4> mat;
  findTetCoords(mesh2verts, coords, elem, mat);
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
OMEGA_H_DEVICE o::LO find_closest_point_on_triangle_with_normal(
  const o::Few< o::Vector<3>, 3> &abc,
  const o::Vector<3> &p, o::Vector<3> &q, o::LO verbose = 0) {

  // Check if P in vertex region outside A
  o::Vector<3> a = abc[0];
  o::Vector<3> b = abc[1];
  o::Vector<3> c = abc[2];

  o::Vector<3> ab = b - a;
  o::Vector<3> ac = c - a;
  o::Vector<3> bc = c - b;
  // Compute parametric position s for projection P’ of P on AB,
  // P’ = A + s*AB, s = snom/(snom+sdenom)
  float snom = osh_dot(p - a, ab);
  float sdenom = osh_dot(p - b, a - b);
  // Compute parametric position t for projection P’ of P on AC,
  // P’ = A + t*AC, s = tnom/(tnom+tdenom)
  float tnom = osh_dot(p - a, ac);
  float tdenom = osh_dot(p - c, a - c);
  if (snom <= 0.0 && tnom <= 0.0){
    q = a;
    return VTXA;
  } // Vertex region early out
  // Compute parametric position u for projection P’ of P on BC,
  // P’ = B + u*BC, u = unom/(unom+udenom)
  float unom = osh_dot(p - b, bc);
  float  udenom = osh_dot(p - c, b - c);
  if (sdenom <= 0.0 && unom <= 0.0){
    q = b;
    return VTXB; // Vertex region early out
  }
  if (tdenom <= 0.0 && udenom <= 0.0){
    q = c;
    return VTXC; // Vertex region early out
  }
  // P is outside (or on) AB if the triple scalar product [N PA PB] <= 0
  o::Vector<3> n = o::cross(b - a, c - a);
  o::Vector<3> temp = o::cross(a - p, b - p);
  float vc = osh_dot(n, temp);
  // If P outside AB and within feature region of AB,
  // return projection of P onto AB
  if (vc <= 0.0 && snom >= 0.0 && sdenom >= 0.0){
    q = a + snom / (snom + sdenom) * ab;
    return EDGEAB;
  }
  // P is outside (or on) BC if the triple scalar product [N PB PC] <= 0
  o::Vector<3> temp1 = o::cross(b - p, c - p);
  float va = osh_dot(n, temp1);
  // If P outside BC and within feature region of BC,
  // return projection of P onto BC
  if (va <= 0.0 && unom >= 0.0 && udenom >= 0.0){
    q = b + unom / (unom + udenom) * bc;
    return EDGEBC;
  }
  // P is outside (or on) CA if the triple scalar product [N PC PA] <= 0
  o::Vector<3> temp2 = o::cross(c - p, a - p);
  float vb = osh_dot(n, temp2);
  // If P outside CA and within feature region of CA,
  // return projection of P onto CA
  if (vb <= 0.0 && tnom >= 0.0 && tdenom >= 0.0){
    q =  a + tnom / (tnom + tdenom) * ac;
    return EDGEAC;
  }
  // P must project inside face region. Compute Q using barycentric coordinates
  float u = va / (va + vb + vc);
  float v = vb / (va + vb + vc);
  float w = 1.0 - u - v; // = vc / (va + vb + vc)
  q = u * a + v * b + w * c;
  return TRIFACE;

}


//Ref: Real-time Collision Detection by Christer Ericson, 2005.
//ptp = ref point; ptq = nearest point on triangle; abc = triangle
OMEGA_H_DEVICE o::Vector<3> closest_point_on_triangle( const o::Few< o::Vector<3>, 3> &abc, 
   const o::Vector<3> &ptp, o::LO* reg=nullptr) {
  o::LO debug = 0;
  o::LO region = -1;
  auto ptq = o::zero_vector<3>();
  // Check if P in vertex region outside A
  o::Vector<3> pta = abc[0];
  o::Vector<3> ptb = abc[1];
  o::Vector<3> ptc = abc[2];

  o::Vector<3> vab = ptb - pta;
  o::Vector<3> vac = ptc - pta;
  o::Vector<3> vap = ptp - pta;
  o::Real d1 = osh_dot(vab, vap);
  o::Real d2 = osh_dot(vac, vap);
  if (d1 <= 0 && d2 <= 0) {
    // barycentric coordinates (1,0,0)
    for(int i=0; i<3; ++i)
      ptq[i] = pta[i];
    region = VTXA;
    if(reg)
      *reg = region;
    return ptq; 
  }

  // Check if P in vertex region outside B
  o::Vector<3> vbp = ptp - ptb;
  o::Real d3 = osh_dot(vab, vbp);
  o::Real d4 = osh_dot(vac, vbp);
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
  o::Real vc = d1*d4 - d3*d2;
  if(region <0 && vc <= 0 && d1 >= 0 && d3 <= 0) {
    o::Real v = d1 / (d1 - d3);
    // barycentric coordinates (1-v,v,0)
    ptq = v*vab;
    ptq = ptq + pta; 
    region = EDGEAB;
    return ptq;
  }

  // Check if P in vertex region outside C
  o::Vector<3> vcp = ptp - ptc;
  o::Real d5 = osh_dot(vab, vcp);
  o::Real d6 = osh_dot(vac, vcp);
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
  o::Real va = d3*d6 - d5*d4;
  if(region <0 && va <= 0 && (d4 - d3) >= 0 && (d5 - d6) >= 0) {
    o::Real w = (d4 - d3) / ((d4 - d3) + (d5 - d6));
    // barycentric coordinates (0,1-w,w)
    ptq =  ptb + w * (ptc - ptb); 
    region = EDGEBC;
    if(reg)
      *reg = region;
    return ptq;
  }

  // P inside face region. Compute Q through its barycentric coordinates (u,v,w)
  if(region <0) {
    o::Real inv = 1 / (va + vb + vc);
    o::Real v = vb * inv;
    o::Real w = vc * inv;
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


template < class ParticleType>
bool search_mesh_2d(o::Mesh& mesh, // (in) mesh
                 ps::SellCSigma< ParticleType >* scs, // (in) particle structure
                 Segment3d x_scs_d, // (in) starting particle positions
                 Segment3d xtgt_scs_d, // (in) target particle positions
                 SegmentInt pid_d, // (in) particle ids
                 o::Write<o::LO> elem_ids, // (out) parent element ids for the target positions
                 o::Write<o::Real> xpoints_d, // (out) particle-boundary intersection points
                 int looplimit=0) {
  const auto btime = pumipic_prebarrier();
  Kokkos::Profiling::pushRegion("pumpipic_search_mesh_2d");
  Kokkos::Timer timer;

  int rank, comm_size;
  MPI_Comm_rank(MPI_COMM_WORLD,&rank);
  MPI_Comm_size(MPI_COMM_WORLD,&comm_size);
  const auto rank_d = rank;

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
      if(!all_positive(faceBcc,1e-8)) {
        printf("%d Particle not in element! ptcl %d elem %d => %d "
          "orig %.15f %.15f bcc %.3f %.3f %.3f\n",
          rank_d, ptcl, e, searchElm, ptclOrigin[0], ptclOrigin[1],
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
        const auto edges = o::gather_down<3>(faceEdges, searchElm);
        const auto faceVerts = o::gather_verts<3>(faces2verts, searchElm);
        const auto faceCoords = o::gather_vectors<3,2>(coords, faceVerts);
        const auto ptclDest = makeVector2(pid, xtgt_scs_d);
        const auto ptclOrigin = makeVector2(pid, x_scs_d);
        Omega_h::Vector<3> faceBcc;
        barycentric_tri(triArea, faceCoords, ptclDest, faceBcc, searchElm);
        auto isDestInParentElm = all_positive(faceBcc);
        ptcl_done[pid] = isDestInParentElm;
        elem_ids_next[pid] = elem_ids[pid];
        const int idx = min3(faceBcc);
        lastEdge[pid] = edges[idx];
      }
    };
    scs->parallel_for(checkCurrentElm);

    auto checkExposedEdges = SCS_LAMBDA(const int& e, const int& pid, const int& mask) {
      if( mask > 0 && !ptcl_done[pid] ) {
        auto searchElm = elem_ids[pid];
        auto ptcl = pid_d(pid);
        assert(lastEdge[pid] != -1);
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
      auto ptclsNotFound = SCS_LAMBDA(const int& e, const int& pid, const int& mask) {
        if( mask > 0 && !ptcl_done[pid] ) {
          auto searchElm = elem_ids[pid];
          auto ptcl = pid_d(pid);
          const auto ptclDest = makeVector2(pid, xtgt_scs_d);
          const auto ptclOrigin = makeVector2(pid, x_scs_d);
          printf("rank %d elm %d ptcl %d notFound %.15f %.15f to %.15f %.15f\n",
              rank_d,
              searchElm, ptcl,
              ptclOrigin[0], ptclOrigin[1],
              ptclDest[0], ptclDest[1]);
        }
      };
      scs->parallel_for(ptclsNotFound, "ptclsNotFound");
      fprintf(stderr, "ERROR:loop limit %d exceeded\n", looplimit);
      break;
    }
  }
  if(!rank || rank == comm_size/2) {
    fprintf(stderr, "%d pumipic search_2d (seconds) %f pre-barrier (seconds) %f\n",
        rank, timer.seconds(), btime);
    fprintf(stderr, "%d pumipic search_2d loops %d\n", rank, loops);
  }
  int maxLoops = 0;
  MPI_Allreduce(&loops, &maxLoops, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
  int minLoops = 0;
  MPI_Allreduce(&loops, &minLoops, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
  long int totLoops = 0;
  long int loops_li = loops;
  MPI_Allreduce(&loops_li, &totLoops, 1, MPI_LONG, MPI_SUM, MPI_COMM_WORLD);
  int ranksWithPtcls = 0;
  const int hasPtcls = (scsCapacity > 0);
  MPI_Allreduce(&hasPtcls, &ranksWithPtcls, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
  const double avgLoops = (double) totLoops / ranksWithPtcls;
  if(maxLoops == loops)
    fprintf(stderr, "pumipic search_2d maxLoops %d on rank %d\n", maxLoops, rank);
  if(minLoops == loops)
    fprintf(stderr, "pumipic search_2d minLoops %d on rank %d\n", minLoops, rank);
  if(!rank)
    fprintf(stderr, "pumipic search_2d totLoops %ld ranksWithPtcls %d average loops %f\n",
     totLoops, ranksWithPtcls, avgLoops);
  Kokkos::Profiling::popRegion();
  return found;
}

} //namespace

#endif //define

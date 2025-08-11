#ifndef PUMIPIC_ADJACENCY_NEW_HPP
#define PUMIPIC_ADJACENCY_NEW_HPP
#include <Omega_h_array.hpp>
#include <Omega_h_defines.hpp>
#include <Omega_h_macros.h>
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
  OMEGA_H_DEVICE void barycentric_tri(
    const o::Real parentArea,
    const o::Matrix<TriDim, TriVerts> &faceCoords,
    const o::Vector<TriDim> &pos,
    o::Vector<TriVerts> &bcc) {
    for(int i=0; i<3; i++) {
      const auto kIdx = simplex_down_template(o::FACE, o::EDGE, i, 0);
      const auto lIdx = simplex_down_template(o::FACE, o::EDGE, i, 1);
      const auto kxy = faceCoords[kIdx];
      const auto lxy = faceCoords[lIdx];
      o::Few<o::Vector<2>, 2> tri;
      tri[0] = lxy - kxy;
      tri[1] = pos - kxy;
      const auto area = o::triangle_area_from_basis(tri);
      bcc[i] = area/parentArea;
    }
  }

  OMEGA_H_DEVICE bool barycentric_tet(
    const o::Real parentVol,
    const Omega_h::Matrix<DIM, 4> &mat,
    const Omega_h::Vector<DIM> &pos,
    Omega_h::Vector<4> &bcc) {
  for(Omega_h::LO i=0; i<4; ++i) 
    bcc[i] = -1;

  Omega_h::Real vals[4];
  Omega_h::Few<Omega_h::Vector<DIM>, 3> abc;
  for(Omega_h::LO iface=0; iface<4; ++iface) {
    get_face_from_face_index_of_tet(mat, iface, abc);
    auto vab = abc[1] - abc[0]; //b - a;
    auto vac = abc[2] - abc[0]; //c - a;
    auto vap = pos - abc[0]; // p - a;
    vals[iface] = o::inner_product(vap, Omega_h::cross(vac, vab)); //ac, ab
  }

  Omega_h::Real inv_vol = 0.0;
  if(parentVol > 0)
    inv_vol = 1.0/parentVol;
  else {
    return 0;
  }
  //bcc[0] for face0 corresp to its opp vtx, so on.
  for(int i=0; i<4; ++i)
    bcc[i] = inv_vol * vals[i];
  return 1; //success
}

  //Check that every particle is within the initial parent element
  template <class ParticleType, typename Segment3d, typename SegmentInt>
  o::LO check_initial_parents(o::Mesh mesh, ParticleStructure<ParticleType>* ptcls,
                              Segment3d x_ps_orig, SegmentInt pids,
                              o::Write<o::LO> elem_ids, o::Write<o::LO> ptcl_done,
                              o::Reals elmArea, const o::Real& tol, bool debug = false) {
    const auto dim = mesh.dim();
    const auto elm2verts = mesh.ask_elem_verts();
    const auto coords = mesh.coords();
    int rank;
    MPI_Comm_rank(mesh.comm()->get_impl(), &rank);
    Omega_h::Write<o::LO> numNotInElem(1, 0, "search_numNotInElem");
    if (dim == 2) {
      auto checkParent = PS_LAMBDA(const int e, const int pid, const int mask) {
        //inactive particle that is still moving to its target position
        if( mask > 0 && !ptcl_done[pid] ) {
          auto searchElm = elem_ids[pid];
          auto ptcl = pids(pid);
          OMEGA_H_CHECK(searchElm >= 0);
          auto elmVerts = o::gather_verts<3>(elm2verts, searchElm);
          const auto elmCoords = o::gather_vectors<3,2>(coords, elmVerts);
          auto ptclOrigin = makeVector2(pid, x_ps_orig);
          Omega_h::Vector<3> faceBcc;
          barycentric_tri(elmArea[searchElm], elmCoords, ptclOrigin, faceBcc);
          if(!all_positive(faceBcc, tol)) {
            if (debug) {
              Kokkos::printf("%d Particle not in element! ptcl %d: %d elem %d => %d "
                     "orig %.15f %.15f bcc %.15f %.15f %.15f\n",
                     rank, pid, ptcl, e, searchElm, ptclOrigin[0], ptclOrigin[1],
                     faceBcc[0], faceBcc[1], faceBcc[2]);
              Kokkos::printf("Element <%f %f> <%f %f> <%f %f>\n", elmCoords[0][0], elmCoords[0][1],
                     elmCoords[1][0], elmCoords[1][1], elmCoords[2][0], elmCoords[2][1]);
            }
            Kokkos::atomic_add(&(numNotInElem[0]), 1);
            elem_ids[pid] = -1;
            ptcl_done[pid] = 1;
          }
        } //if active
      };
      parallel_for(ptcls, checkParent, "search_checkParent");
    }
    else if (dim == 3) {
      auto checkParent = PS_LAMBDA(const int e, const int pid, const int mask) {
        if( mask > 0 && !ptcl_done[pid] ) {
          auto searchElm = elem_ids[pid];
          auto ptcl = pids(pid);
          OMEGA_H_CHECK(searchElm >= 0);
          auto elmVerts = o::gather_verts<4>(elm2verts, searchElm);
          const auto elmCoords = o::gather_vectors<4,3>(coords, elmVerts);
          auto ptclOrigin = makeVector3(pid, x_ps_orig);
          Omega_h::Vector<4> bcc;
          barycentric_tet(elmArea[searchElm], elmCoords, ptclOrigin, bcc);
          if(!all_positive(bcc, tol)) {
            if (debug) {
              Kokkos::printf("%d Particle not in element! ptcl %d: %d elem %d => %d "
                     "orig %.15f %.15f %.15f bcc %.15f %.15f %.15f %.15f\n",
                     rank, pid, ptcl, e, searchElm,
                     ptclOrigin[0], ptclOrigin[1], ptclOrigin[2],
                     bcc[0], bcc[1], bcc[2], bcc[3]);
            }
            Kokkos::atomic_add(&(numNotInElem[0]), 1);
            elem_ids[pid] = -1;
            ptcl_done[pid] = 1;
          }
        } //if active
      };
      parallel_for(ptcls, checkParent, "search_checkParent");
    }
    Omega_h::HostWrite<o::LO> numNotInElem_h(numNotInElem);
    if (numNotInElem_h[0] > 0) {
      printError( "[WARNING] Rank %d: %d particles are not located in their "
              "starting elements. Deleting them...\n", rank, numNotInElem_h[0]);
    }
    return numNotInElem_h[0];
  }

  // Uses Moller Trumbore line triangle intersection method
  /*
    Möller and Trumbore, « Fast, Minimum Storage Ray-Triangle Intersection », Journal of Graphics Tools, vol. 2,‎ 1997, p. 21–28
    https://cadxfem.org/inf/Fast%20MinimumStorage%20RayTriangle%20Intersection.pdf
   */
  OMEGA_H_DEVICE bool ray_intersects_triangle(const o::Few<o::Vector<3>, 3>& faceVerts,
                                                    const o::Vector<3>& orig, const o::Vector<3>& dest,
                                                    o::Vector<3>& xpoint, const o::Real tol, const o::LO flip,
                                                    o::Real& dproj, o::Real& closeness, o::Real& intersection_parametric_coord) {
    const o::LO vtx1 = 2 - flip;
    const o::LO vtx2 = flip + 1;
    const o::Vector<3> edge1 = faceVerts[vtx1] - faceVerts[0];
    const o::Vector<3> edge2 = faceVerts[vtx2] - faceVerts[0];
    const o::Vector<3> displacement = dest-orig;
    const o::Real seg_length = o::norm(displacement);
    const o::Vector<3> dir = displacement/seg_length;
    const o::Vector<3> faceNorm = o::cross(edge2, edge1);
    const o::Vector<3> pvec = o::cross(dir, edge2);
    dproj = o::inner_product(dir, faceNorm);
    const o::Real invdet = 1.0/dproj;
    //u, v, (1-u-v) define the intersection point on the plane of the triangle
    const o::Vector<3> tvec = orig - faceVerts[0];
    const o::Real u = invdet * o::inner_product(tvec, pvec);
    const o::Vector<3> qvec = o::cross(tvec, edge1);
    const o::Real v = invdet * o::inner_product(dir, qvec);
    //t is the distance the intersection point is along the particle path
    const o::Real t = invdet * o::inner_product(edge2, qvec);
    intersection_parametric_coord = t/seg_length;
    xpoint = orig + dir * t;
    closeness = Kokkos::max(Kokkos::max(Kokkos::min(Kokkos::fabs(u), Kokkos::fabs(1 - u)), Kokkos::min(Kokkos::fabs(v), Kokkos::fabs(1 - v))), Kokkos::min(Kokkos::fabs(u + v), Kokkos::fabs(1 - u - v)));
    return (dproj >= tol) && (t >= -tol) && (u >= -tol) && (v >= -tol) && (u+v <= 1.0 + 2 * tol);
  }

  [[deprecated("[Deprecated] Consider using ray_intersects_triangle or "
               "line_segment_intersects_triangle instead.")]]
  OMEGA_H_DEVICE bool moller_trumbore_line_triangle(
      const o::Few<o::Vector<3>, 3> &faceVerts, const o::Vector<3> &orig,
      const o::Vector<3> &dest, o::Vector<3> &xpoint, const o::Real tol,
      const o::LO flip, o::Real &dproj, o::Real &closeness) {
    o::Real intersection_parametric_coord;
    return ray_intersects_triangle(faceVerts, orig, dest, xpoint, tol, flip,
                                   dproj, closeness,
                                   intersection_parametric_coord);
  }

  OMEGA_H_DEVICE bool line_segment_intersects_triangle(
      const o::Few<o::Vector<3>, 3> &faceVerts, const o::Vector<3> &orig,
      const o::Vector<3> &dest, o::Vector<3> &xpoint, const o::Real tol,
      const o::LO flip, o::Real &dproj, o::Real &closeness,
      o::Real &intersection_parametric_coord) {
    bool ray_intersects =
        ray_intersects_triangle(faceVerts, orig, dest, xpoint, tol, flip, dproj,
                                closeness, intersection_parametric_coord);
    return ray_intersects && intersection_parametric_coord <= 1 + tol;
  }

  //Simple 2d line segment intersection routine
  OMEGA_H_DEVICE bool line_edge_2d(const o::Few<o::Vector<2>, 2>& edgeVerts,
                                   const o::Vector<2>& orig, const o::Vector<2>& dest,
                                   o::Vector<2>& xpoint, o::Real tol, o::LO flip) {
    const o::LO vtx1 = flip;
    const o::LO vtx2 = !flip;
    const o::Vector<2> path = dest - orig;
    const o::Vector<2> edge = edgeVerts[vtx2] - edgeVerts[vtx1];
    const o::Vector<2> norm = o::perp(edge);
    const o::Vector<2> normp = o::perp(path);
    const o::Real det = -o::inner_product(norm, path);
    const o::Real s = o::inner_product(normp, orig - edgeVerts[vtx1]);
    const o::Real t = o::inner_product(norm, orig - edgeVerts[vtx1]);
    xpoint = orig + (t / det) * path;
    return det >= tol && s >= -tol && s <= det+tol && t >= -tol && t <= det+tol;
  }

  OMEGA_H_DEVICE o::LO find_exit_face_bcc_3d(const o::Real elmArea, const o::Matrix<3, 4>& elmCoords,
                                             const o::Vector<3>& ptclOrigin, o::LO& ptcl_done) {
    Omega_h::Vector<4> bcc;
    barycentric_tet(elmArea, elmCoords, ptclOrigin, bcc);
    auto isDestInParentElm = all_positive(bcc);
    ptcl_done = isDestInParentElm;
    //Find min and set exit face
    return min_index(bcc, 4);
  }
  

  template <class ParticleType, typename Segment3d>
  void find_exit_face(o::Mesh mesh, ParticleStructure<ParticleType>* ptcls,
                      Segment3d x_ps_orig, Segment3d x_ps_tgt,
                      o::Write<o::LO> elem_ids, o::Write<o::LO> ptcl_done,
                      o::Reals elmArea, bool useBcc, o::Write<o::LO> lastExit,
                      o::Write<o::Real> xPoints, const o::Real& tol) {
    const auto dim = mesh.dim();
    const auto elm2verts = mesh.ask_elem_verts();
    const auto coords = mesh.coords();
    const auto elm2faces = mesh.ask_down(dim, dim-1);
    const auto elmDown = elm2faces.ab2b;

    if (useBcc) {
      if (dim == 2) {
        auto findExitFace = PS_LAMBDA(const int, const int pid, const int mask) {
          if( mask > 0 && !ptcl_done[pid] ) {
            auto searchElm = elem_ids[pid];
            OMEGA_H_CHECK(searchElm >= 0);
            //Calculate BCC
            auto elmVerts = o::gather_verts<3>(elm2verts, searchElm);
            const auto elmCoords = o::gather_vectors<3,2>(coords, elmVerts);
            auto ptclOrigin = makeVector2(pid, x_ps_tgt);
            Omega_h::Vector<3> faceBcc;
            barycentric_tri(elmArea[searchElm], elmCoords, ptclOrigin, faceBcc);
            auto isDestInParentElm = all_positive(faceBcc);
            ptcl_done[pid] = isDestInParentElm;
            //Find min and set exit edge
            const int idx = min3(faceBcc);
            const auto edges = o::gather_down<3>(elmDown, searchElm);
            lastExit[pid] = edges[idx];
          }
        };
        parallel_for(ptcls, findExitFace, "search_findExitFace_bcc_2d");
      }
      else if (dim == 3) {
        auto findExitFace = PS_LAMBDA(const int e, const int pid, const int mask) {
          if( mask > 0 && !ptcl_done[pid] ) {
            auto searchElm = elem_ids[pid];
            OMEGA_H_CHECK(searchElm >= 0);
            auto elmVerts = o::gather_verts<4>(elm2verts, searchElm);
            const auto elmCoords = o::gather_vectors<4,3>(coords, elmVerts);
            auto ptclOrigin = makeVector3(pid, x_ps_tgt);
            const o::LO face_id = find_exit_face_bcc_3d(elmArea[searchElm], elmCoords, ptclOrigin, ptcl_done[pid]);
            const auto faces = o::gather_down<4>(elmDown, searchElm);
            lastExit[pid] = faces[face_id];
          }
        };
        parallel_for(ptcls, findExitFace, "search_findExitFace_bcc_3d");
      }
    }
    else {
      const auto bridgeVerts =  mesh.ask_verts_of(dim-1);
      //Use line face intersection to determine exit face
      if (dim == 2) {
        auto findExitFace = PS_LAMBDA(const lid_t, const lid_t ptcl, const bool mask) {
          if (mask > 0 && !ptcl_done[ptcl]) {
            const auto searchElm = elem_ids[ptcl];
            OMEGA_H_CHECK(searchElm >= 0);
            const auto faceVerts = o::gather_verts<3>(elm2verts, searchElm);
            const auto faceCoords = o::gather_vectors<3,2>(coords, faceVerts);
            const auto dest = makeVector2(ptcl, x_ps_tgt);
            const auto orig = makeVector2(ptcl, x_ps_orig);
            const auto edge_ids = o::gather_down<3>(elmDown, searchElm);
            auto xpts = o::zero_vector<2>();
            const o::LO prevExit = lastExit[ptcl];
            lastExit[ptcl] = -1;
            for(int ei=0; ei<3; ++ei) {
              const auto edge_id = edge_ids[ei];
              if (edge_id == prevExit)
                continue;
              const auto ev2v = o::gather_verts<2>(bridgeVerts, edge_id);
              const auto edge = o::gather_vectors<2, 2>(coords, ev2v);
              const o::LO flip = isFaceFlipped(ei, ev2v, faceVerts);
              const bool success = line_edge_2d(edge, orig, dest, xpts, tol, flip);
              if (success) {
                lastExit[ptcl] = edge_id;
                xPoints[2*ptcl] = xpts[0]; xPoints[2*ptcl+1] = xpts[1];
              }
            }
            ptcl_done[ptcl] = (lastExit[ptcl] == -1);
          }
        };
        parallel_for(ptcls, findExitFace, "search_findExitFace_intersect_2d");
      }
      else if (dim == 3) {
        auto findExitFace = PS_LAMBDA(const lid_t, const lid_t ptcl, const bool mask) {
          if (mask > 0 && !ptcl_done[ptcl]) {
            const auto searchElm = elem_ids[ptcl];
            OMEGA_H_CHECK(searchElm >= 0);
            const auto tetv2v = o::gather_verts<4>(elm2verts, searchElm);
              const auto tetCoords = o::gather_vectors<4,3>(coords, tetv2v);
            const auto dest = makeVector3(ptcl, x_ps_tgt);
            const auto orig = makeVector3(ptcl, x_ps_orig);
            const auto face_ids = o::gather_down<4>(elmDown, searchElm);
            auto xpts = o::zero_vector<3>();
            const o::LO prevExit = lastExit[ptcl];
            lastExit[ptcl] = -1;
            o::Real quality = -1;
            o::LO bestFace = -1;
            for(int fi=0; fi<4; ++fi) {
              const auto face_id = face_ids[fi];
              if (face_id == prevExit)
                continue;
              const auto fv2v = o::gather_verts<3>(bridgeVerts, face_id);
              const auto face = o::gather_vectors<3,3>(coords, fv2v);
              const o::LO flip = isFaceFlipped(fi, fv2v, tetv2v);
              o::Real dproj;
              o::Real closeness;
              o::Real intersection_parametric_coord;
              const bool success = ray_intersects_triangle(face, orig, dest, xpts, tol, flip, dproj,
                                                           closeness, intersection_parametric_coord);
              if (success) {
                lastExit[ptcl] = face_id;
                xPoints[3*ptcl] = xpts[0]; xPoints[3*ptcl + 1] = xpts[1]; xPoints[3*ptcl + 2] = xpts[2];
              }
              if (dproj > -tol && (quality < 0 || closeness < quality) && lastExit[ptcl] == -1) {
                quality = closeness;
                bestFace = face_id;
                xPoints[3*ptcl] = xpts[0]; xPoints[3*ptcl + 1] = xpts[1]; xPoints[3*ptcl + 2] = xpts[2];
              }
            }

            //If line intersection fails then use BCC method
            if (lastExit[ptcl] == -1) {              
                lastExit[ptcl] = bestFace;
            }
            ptcl_done[ptcl] = (lastExit[ptcl] == -1);
          }
        };
        parallel_for(ptcls, findExitFace, "search_findExitFace_intersect_3d");
      }
    }
  }

  template <typename ParticleType, typename Segment3d>
  void check_model_intersection(o::Mesh mesh, ParticleStructure<ParticleType>* ptcls,
                                Segment3d x_ps_orig, Segment3d x_ps_tgt,
                                o::Write<o::LO> elem_ids, o::Write<o::LO> ptcl_done,
                                o::Write<o::LO> lastExit, o::Bytes side_is_exposed,
                                bool requireIntersection,
                                o::Write<o::LO> xFace) {
    auto checkExposedEdges = PS_LAMBDA(const int e, const int pid, const int mask) {
      if( mask > 0 && !ptcl_done[pid] ) {
        assert(lastExit[pid] != -1);
        const o::LO bridge = lastExit[pid];
        const bool exposed = side_is_exposed[bridge];
        ptcl_done[pid] = exposed;
        if (exposed && requireIntersection) {
          xFace[pid] = lastExit[pid];
        }
        else {
          elem_ids[pid] = exposed ? -1 : elem_ids[pid]; //leaves domain if exposed
        }
      }
    };
    parallel_for(ptcls, checkExposedEdges, "pumipic_checkExposedEdges");
  }

  template <class ParticleType>
  void set_new_element(o::Mesh& mesh, ParticleStructure<ParticleType>* ptcls,
                       o::Write<o::LO> elem_ids, o::LOs ptcl_done,
                       o::Write<o::LO> lastExit) {
    int dim = mesh.dim();
    const auto faces2elms = mesh.ask_up(dim-1, dim);
    auto e2f_vals = faces2elms.ab2b; // CSR value array
    auto e2f_offsets = faces2elms.a2ab; // CSR offset array, index by mesh edge ids
    auto setNextElm = PS_LAMBDA(const int& e, const int& pid, const int& mask) {
      if( mask > 0 && !ptcl_done[pid] ) {
        auto searchElm = elem_ids[pid];
        auto bridge = lastExit[pid];
        auto e2f_first = e2f_offsets[bridge];
        #ifdef _DEBUG
          auto e2f_last = e2f_offsets[bridge+1];
          auto upFaces = e2f_last - e2f_first;
          assert(upFaces==2);
        #endif
        auto faceA = e2f_vals[e2f_first];
        auto faceB = e2f_vals[e2f_first+1];
        assert(faceA != faceB);
        assert(faceA == searchElm || faceB == searchElm);
        auto nextElm = (faceA == searchElm) ? faceB : faceA;
        elem_ids[pid] = nextElm;
      }
    };
    parallel_for(ptcls, setNextElm, "pumipic_setNextElm");
  }

  template <typename Array>
  o::Real compute_tolerance_from_area(Array elmArea) {
    o::Real min_area;
    Kokkos::parallel_reduce(elmArea.size(), OMEGA_H_LAMBDA(const o::LO elm, o::Real& area) {
        if (elmArea[elm] < area)
          area = elmArea[elm];
      }, Kokkos::Min<o::Real>(min_area));
    o::Real tol = Kokkos::max(1e-15 / min_area, 1e-8);
    printInfo("Min area is: %.15f, Planned tol is %.15f\n", min_area, tol);
    return tol;
  }

  /**
  * @brief Particle tracing through mesh
  *
  * Starting at the initial position, the particles are traced through the mesh (both 2D and 3D)
  * until they are all marked "done" by the particle handler "func" at element boundary or destination. 
  * It gives back the parent element ids for the new positions of the particles,
  * the exit face ids for the particles that leave the domain, and the last intersection point for 
  * the particles.
  *
  * Note: Particle trajectories are considered as rays (not line segments).
  *
  *
  * @tparam ParticleType Particle type
  * @tparam Segment3d Segment type for particle position
  * @tparam SegmentInt Segment type for particle ids
  * @tparam Func Callable type object
  * @param mesh Omega_h mesh to search on
  * @param ptcls Particle structure
  * @param x_ps_orig Particle starting position
  * @param x_ps_tgt Particle target position
  * @param pids Particle ids
  * @param elem_ids Paricle parent element ids
  * @param requireIntersection True if intersection is required
  * @param inter_faces Exit faces for particles at domain boundary
  * @param inter_points Stores intersection points for particles at each face
  * @param looplimit Maximum number of iterations
  * @param debug True if debug information is printed
  * @param func Callable object to handle particles at element sides or destination
  * @return True if all particles are found at destination or left domain
  */
  template <class ParticleType, typename Segment3d, typename SegmentInt, typename Func>
  bool trace_particle_through_mesh(o::Mesh& mesh, ParticleStructure<ParticleType>* ptcls,
                   Segment3d x_ps_orig, Segment3d x_ps_tgt, SegmentInt pids,
                   o::Write<o::LO>& elem_ids,
                   bool requireIntersection,
                   o::Write<o::LO>& inter_faces,
                   o::Write<o::Real>& inter_points,
                   int looplimit,
                   bool debug,
                   Func& func) {
    static_assert(
        std::is_invocable_r_v<
            void, Func, o::Mesh &, ParticleStructure<ParticleType> *,
            o::Write<o::LO> &, o::Write<o::LO> &, o::Write<o::LO> &,
            o::Write<o::Real> &, o::Write<o::LO> &,
            Segment3d, Segment3d>,
        "Functional must accept <mesh> <ps> <elem_ids> <inter_faces> <lastExit> <inter_points> <ptcl_done> <x_ps_orig> <x_ps_tgt>\n");

    //Initialize timer
    const auto btime = pumipic_prebarrier(mesh.comm()->get_impl());
    Kokkos::Profiling::pushRegion("pumipic_search_mesh");
    Kokkos::Timer timer;

    //Initial setup
    const auto psCapacity = ptcls->capacity();
    // True if particle has reached new parent element
    o::Write<o::LO> ptcl_done(psCapacity, 0, "search_ptcl_done");
    // Store the last exit face
    o::Write<o::LO> lastExit(psCapacity,-1, "search_last_exit");
    const auto elmArea = measure_elements_real(&mesh);
    bool useBcc = !requireIntersection;
    o::Real tol = compute_tolerance_from_area(elmArea);
    
    int rank;
    MPI_Comm_rank(mesh.comm()->get_impl(), &rank);
    
    const auto dim = mesh.dim();
    const auto edges2faces = mesh.ask_up(dim-1, dim);
    const auto side_is_exposed = mark_exposed_sides(&mesh);
    const auto faces2verts = mesh.ask_elem_verts();
    const auto coords = mesh.coords();
    const auto edge_verts =  mesh.ask_verts_of(dim - 1);
    
    //Setup the output information
    if (elem_ids.size() == 0) {
      //Setup new parent id arrays
      elem_ids = o::Write<o::LO>(psCapacity, -1, "search_elem_ids");
      auto setInitial = PS_LAMBDA(const int e, const int pid, const int mask) {
        if(mask) {
          elem_ids[pid] = e;
        }
        else
          ptcl_done[pid] = 1;
      };
      parallel_for(ptcls, setInitial, "search_setInitial");
    }
    else {
      auto setInitial = PS_LAMBDA(const int e, const int pid, const int mask) {
        if((mask && elem_ids[pid] == -1) || (!mask))
          ptcl_done[pid] = 1;
      };
      parallel_for(ptcls, setInitial, "search_setInitial");
    }

    //Finish particles that didn't move
    auto finishUnmoved = PS_LAMBDA(const int e, const int pid, const int mask) {
      if  (mask){
        const o::Vector<3> start = makeVector3(pid, x_ps_orig);
        const o::Vector<3> end = makeVector3(pid, x_ps_tgt);
        if (o::norm(end - start) < tol)
          ptcl_done[pid] = 1;
      }
    };
    parallel_for(ptcls, finishUnmoved, "search_finishUnmoved");
    
    if (requireIntersection) {
      //Setup intersection arrays
      if (inter_points.size() == 0 || inter_faces.size() == 0) {
        inter_points = o::Write<o::Real>(dim*ptcls->capacity(), 0);
        inter_faces = o::Write<o::LO>(ptcls->capacity(), -1);
      }
      else {
        auto initializeIntersection = PS_LAMBDA(const int& e, const int& pid, const int& mask) {
          for (int i = 0; i < dim; ++i)
            inter_points[dim * pid + i] = 0;
          inter_faces[pid] = -1;
        };
        parallel_for(ptcls, initializeIntersection, "search_initializeIntersection");
      }
   }

    //Ensure all particles are within their starting element
    check_initial_parents(mesh, ptcls, x_ps_orig, pids, elem_ids, ptcl_done, elmArea, tol, debug);

    bool found = false;
    int loops = 0;

    //Iteratively find the next element until parent element is reached for each particle
    while (!found) {
      //Find intersection face
      find_exit_face(mesh, ptcls, x_ps_orig, x_ps_tgt, elem_ids, ptcl_done, elmArea, useBcc, lastExit, inter_points, tol);
      //Check if intersection face is exposed
      func(mesh, ptcls, elem_ids, inter_faces, lastExit, inter_points, ptcl_done, x_ps_orig, x_ps_tgt);
      
      // Move to next element
      set_new_element(mesh, ptcls, elem_ids, ptcl_done, lastExit);

      //Check if all particles are found
      found = true;
      o::LOs ptcl_done_r(ptcl_done);
      auto minFlag = o::get_min(ptcl_done_r);
      if(minFlag == 0)
        found = false;
      ++loops;

      // o::LO nr;
      // Kokkos::parallel_reduce(ptcl_done.size(), OMEGA_H_LAMBDA(const o::LO ptcl, o::LO& count) {
      //     count += (ptcl_done[ptcl] == 0);
      //     if (loops > 10000 && ptcl_done[ptcl] == 0)
      //       printInfo("  Remains %d in %d\n", ptcl, elem_ids[ptcl]);
      //   }, nr);
      // if (loops % 10 == 0)
      //   printInfo("Loop %d: %d remaining\n", loops, nr);
      //Check iteration count
      if(looplimit && loops >= looplimit) {
        Omega_h::Write<o::LO> numNotFound(1,0);
        auto ptclsNotFound = PS_LAMBDA(const int& e, const int& pid, const int& mask) {
          if( mask > 0 && !ptcl_done[pid] ) {
            auto searchElm = elem_ids[pid];
            auto ptcl = pids(pid);
            const auto ptclDest = makeVector2(pid, x_ps_orig);
            const auto ptclOrigin = makeVector2(pid, x_ps_tgt);
            if (debug) {
              Kokkos::printf("rank %d elm %d ptcl %d notFound %.15f %.15f to %.15f %.15f\n",
                     rank, searchElm, ptcl, ptclOrigin[0], ptclOrigin[1],
                     ptclDest[0], ptclDest[1]);
            }
            elem_ids[pid] = -1;
            Kokkos::atomic_add(&(numNotFound[0]), 1);
          }
        };
        ps::parallel_for(ptcls, ptclsNotFound, "ptclsNotFound");
        Omega_h::HostWrite<o::LO> numNotFound_h(numNotFound);
        printError( "ERROR:Rank %d: loop limit %d exceeded. %d particles were "
                "not found. Deleting them...\n", rank, looplimit, numNotFound_h[0]);
        break;
      }

    }
    RecordTime("pumipic search_mesh", timer.seconds(), btime);
    char buffer[1024];
    sprintf(buffer, "%d pumipic search_mesh loops %d", rank, loops);
    PrintAdditionalTimeInfo(buffer, 1);
    Kokkos::Profiling::popRegion();
    return found;
  }

  template <typename ParticleType, typename Segment3d>
  struct RemoveParticleOnGeometricModelExit {
    RemoveParticleOnGeometricModelExit(o::Mesh &mesh, bool requireIntersection)
        : requireIntersection_(requireIntersection) {
      side_is_exposed_ = mark_exposed_sides(&mesh);
    }

    void operator()(o::Mesh& mesh, ParticleStructure<ParticleType>* ptcls,
                    o::Write<o::LO>& elem_ids, o::Write<o::LO>& inter_faces,
                    o::Write<o::LO>& lastExit, o::Write<o::Real>& inter_points,
                    o::Write<o::LO>& ptcl_done,
                    Segment3d x_ps_orig,
                    Segment3d x_ps_tgt) const {
      // Check if intersection face is exposed
      check_model_intersection(mesh, ptcls, x_ps_orig, x_ps_tgt, elem_ids,
                               ptcl_done, lastExit, side_is_exposed_,
                               requireIntersection_, inter_faces);
    }

    private:
    bool requireIntersection_;
    o::Bytes side_is_exposed_;
  };

  template <class ParticleType, typename Segment3d, typename SegmentInt>
  bool search_mesh(o::Mesh& mesh, ParticleStructure<ParticleType>* ptcls,
                   Segment3d x_ps_orig, Segment3d x_ps_tgt, SegmentInt pids,
                   o::Write<o::LO>& elem_ids,
                   bool requireIntersection,
                   o::Write<o::LO>& inter_faces,
                   o::Write<o::Real>& inter_points,
                   int looplimit,
                   int debug) {
    RemoveParticleOnGeometricModelExit<ParticleType, Segment3d> native_handler(mesh, requireIntersection);

    return trace_particle_through_mesh(mesh, ptcls, x_ps_orig, x_ps_tgt, pids, elem_ids, requireIntersection,
                           inter_faces, inter_points, looplimit, debug, native_handler);
  }
}
#endif

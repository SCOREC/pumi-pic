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
#include "pumipic_adjacency.hpp"

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

  //Check that every particle is within the initial parent element. A particle
  //that fails the check is usually sitting on a shared face with a barycentric
  //coordinate that rounds slightly negative. A particle on a face can head into
  //either of the two elements adjacent to it, so in 3D the element on the side
  //of the destination is picked instead of blindly relocating or deleting it.
  //Particles far from any face of the recorded element, or heading out of the
  //domain over a boundary face, are deleted.
  template <class ParticleType, typename Origin, typename SegmentInt, typename Target>
  o::LO check_initial_parents(o::Mesh mesh, ParticleStructure<ParticleType>* ptcls,
                              Origin x_ps_orig, Target x_ps_tgt, SegmentInt pids,
                              o::Write<o::LO> elem_ids, o::Write<o::LO> ptcl_done,
                              o::Reals elmArea, const o::Real& tol, bool debug = false) {
    const auto dim = mesh.dim();
    const auto elm2verts = mesh.ask_elem_verts();
    const auto coords = mesh.coords();
    const auto elm2faces = mesh.ask_down(dim, dim-1);
    const auto elmDown = elm2faces.ab2b;
    const auto faces2elms = mesh.ask_up(dim-1, dim);
    const auto faces2elms_vals = faces2elms.ab2b;
    const auto faces2elms_offsets = faces2elms.a2ab;
    const auto verts2elems = mesh.ask_up(0, dim);
    const auto verts2elems_vals = verts2elems.ab2b;
    const auto verts2elems_offsets = verts2elems.a2ab;
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
              printInfo("%d Particle not in element! ptcl %d: %d elem %d => %d "
                     "orig %.15f %.15f bcc %.15f %.15f %.15f\n",
                     rank, pid, ptcl, e, searchElm, ptclOrigin[0], ptclOrigin[1],
                     faceBcc[0], faceBcc[1], faceBcc[2]);
              printInfo("Element <%f %f> <%f %f> <%f %f>\n", elmCoords[0][0], elmCoords[0][1],
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
          OMEGA_H_CHECK(searchElm >= 0);
          auto elmVerts = o::gather_verts<4>(elm2verts, searchElm);
          const auto elmCoords = o::gather_vectors<4,3>(coords, elmVerts);
          auto ptclOrigin = makeVector3(pid, x_ps_orig);
          auto ptclDest = makeVector3(pid, x_ps_tgt);
          Omega_h::Vector<4> bcc;
          barycentric_tet(elmArea[searchElm], elmCoords, ptclOrigin, bcc);
          if(!all_positive(bcc, tol)) {
            const o::LO min_bcc_idx = min_index(bcc, 4);
            //Only recover particles that are essentially on the face with the
            //most negative bcc (round-off); a large violation means the
            //recorded element is genuinely wrong and the particle is deleted
            if (bcc[min_bcc_idx] > -1e-2) { // ~0.17% of the element height
              //A particle on a face can head into either of the two elements
              //adjacent to it, so pick the one the destination lies in
              const auto faces = o::gather_down<4>(elmDown, searchElm);
              const o::LO face = faces[min_bcc_idx];
              const auto first = faces2elms_offsets[face];
              const auto last = faces2elms_offsets[face + 1];
              o::LO otherElm = -1;
              for (o::LO ai = first; ai < last; ++ai)
                if (faces2elms_vals[ai] != searchElm)
                  otherElm = faces2elms_vals[ai];
              //signed distance of the destination from the face plane in the
              //same winding as barycentric_tet (negative = beyond the face)
              Omega_h::Few<Omega_h::Vector<DIM>, 3> abc;
              get_face_from_face_index_of_tet(elmCoords, min_bcc_idx, abc);
              const auto vab = abc[1] - abc[0];
              const auto vac = abc[2] - abc[0];
              const o::Real destSide = o::inner_product(ptclDest - abc[0],
                                                        o::cross(vac, vab));
              const o::Real originSide = bcc[min_bcc_idx] * elmArea[searchElm];
              if (destSide < 0) {
                if (otherElm != -1) {
                  //heading into the adjacent element
                  elem_ids[pid] = otherElm;
                } else if (destSide < originSide) {
                  //heading out of the domain over the boundary face
                  Kokkos::atomic_add(&(numNotInElem[0]), 1);
                  elem_ids[pid] = -1;
                  ptcl_done[pid] = 1;
                }
                //else: zero-length move or heading back in over the boundary
                //face; keep the recorded element
              }
              //else: heading into the recorded element; keep it
            } else {
              //The recorded element is genuinely wrong: a near-edge crossing
              //in an earlier trace sent the particle to a wrong element where
              //no exit face was found. Before giving up on the particle, look
              //for the element containing the origin among the elements
              //sharing a vertex with the recorded element (the true element
              //is an edge neighbour of the element the trace stopped in).
              o::LO foundElm = -1;
              for (int vi = 0; vi < 4 && foundElm == -1; ++vi) {
                const o::LO v = elmVerts[vi];
                const auto first = verts2elems_offsets[v];
                const auto last = verts2elems_offsets[v + 1];
                for (o::LO ai = first; ai < last; ++ai) {
                  const o::LO cand = verts2elems_vals[ai];
                  if (cand == searchElm)
                    continue;
                  const auto candVerts = o::gather_verts<4>(elm2verts, cand);
                  const auto candCoords =
                      o::gather_vectors<4,3>(coords, candVerts);
                  Omega_h::Vector<4> candBcc;
                  if (barycentric_tet(elmArea[cand], candCoords, ptclOrigin,
                                      candBcc) &&
                      all_positive(candBcc, tol))
                    foundElm = cand;
                }
              }
              if (foundElm != -1) {
                elem_ids[pid] = foundElm;
              } else {
                Kokkos::atomic_add(&(numNotInElem[0]), 1);
                elem_ids[pid] = -1;
                ptcl_done[pid] = 1;
              }
            }
          }
        } //if active
      };
      parallel_for(ptcls, checkParent, "search_checkParent");
    }
    Omega_h::HostWrite<o::LO> numNotInElem_h(numNotInElem);
    if (numNotInElem_h[0] > 0) {
      printError( "[WARNING] Rank %d: %d particles are not located in their "
              "starting elements and could not be recovered. Deleting them...\n",
              rank, numNotInElem_h[0]);
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
    if (seg_length == 0) {
      xpoint = orig;
      intersection_parametric_coord = 0;
      dproj = closeness = 0;
      return false;
    }
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

  template <int Dim>
  OMEGA_H_DEVICE o::LO find_bcc_side(o::Real measure,
      const o::Matrix<Dim, Dim + 1>& vertices,
      const o::Vector<Dim>& destination, o::Real tol) {
    o::Vector<Dim + 1> bcc;
    if constexpr (Dim == 2)
      barycentric_tri(measure, vertices, destination, bcc);
    else
      barycentric_tet(measure, vertices, destination, bcc);
    return all_positive(bcc, tol) ? -1 : min_index(bcc, Dim + 1);
  }

  template <class ParticleType, typename Origin, typename Target>
  void find_exit_face(o::Mesh mesh, ParticleStructure<ParticleType>* ptcls,
                      Origin x_ps_orig, Target x_ps_tgt,
                      o::Write<o::LO> elem_ids, o::Write<o::LO> ptcl_done,
                      o::Reals elmArea, bool useBcc, o::Write<o::LO> lastExit,
                      o::Write<o::Real> xPoints, const o::Real& tol) {
    const auto dim = mesh.dim();
    const auto elm2verts = mesh.ask_elem_verts();
    const auto coords = mesh.coords();
    const auto elm2faces = mesh.ask_down(dim, dim-1);
    const auto elmDown = elm2faces.ab2b;

    if (useBcc) {
      auto walk = PS_LAMBDA(const int, const int pid, const bool mask) {
        if (!mask || ptcl_done[pid]) return;
        const auto elem = elem_ids[pid];
        OMEGA_H_CHECK(elem >= 0);
        o::LO side;
        if (dim == 2) {
          const auto verts = o::gather_verts<3>(elm2verts, elem);
          side = find_bcc_side<2>(elmArea[elem],
              o::gather_vectors<3, 2>(coords, verts), makeVector2(pid, x_ps_tgt), tol);
        } else {
          const auto verts = o::gather_verts<4>(elm2verts, elem);
          side = find_bcc_side<3>(elmArea[elem],
              o::gather_vectors<4, 3>(coords, verts), makeVector3(pid, x_ps_tgt), tol);
        }
        ptcl_done[pid] = (side == -1);
        lastExit[pid] = side == -1 ? -1 : elmDown[elem * (dim + 1) + side];
      };
      parallel_for(ptcls, walk, "search_findExitFace_bcc");
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
              const o::LO flip = isFaceFlipped(ei, ev2v, faceVerts) ^ (elmArea[searchElm] < 0);
              const bool success = line_edge_2d(edge, orig, dest, xpts, tol, flip);
              if (success) {
                lastExit[ptcl] = edge_id;
                xPoints[2*ptcl] = xpts[0]; xPoints[2*ptcl+1] = xpts[1];
              }
            }
            if (lastExit[ptcl] == -1) {
              xPoints[2*ptcl] = dest[0];
              xPoints[2*ptcl+1] = dest[1];
            }
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
            o::Real bestInteriority = -1e30;
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
              const bool success = line_segment_intersects_triangle(face, orig, dest, xpts, tol, flip, dproj,
                                                           closeness, intersection_parametric_coord);
              if (success) {
                //A ray passing within tolerance of an element edge (or vertex)
                //makes several faces report an intersection. The true exit
                //face is the one whose triangle contains the crossing point in
                //its interior, so pick the face with the most interior
                //crossing instead of the last successful face.
                Omega_h::Vector<3> faceBcc;
                o::Real interiority = -1.0;
                if (find_barycentric_tri_simple(face, xpts, faceBcc)) {
                  interiority = Kokkos::min(Kokkos::min(faceBcc[0], faceBcc[1]),
                                            faceBcc[2]);
                }
                if (interiority > bestInteriority) {
                  bestInteriority = interiority;
                  lastExit[ptcl] = face_id;
                  xPoints[3*ptcl] = xpts[0]; xPoints[3*ptcl + 1] = xpts[1]; xPoints[3*ptcl + 2] = xpts[2];
                }
              }
            }

            if (lastExit[ptcl] == -1){ // reached destination
                xPoints[3*ptcl] = x_ps_tgt(ptcl,0);
                xPoints[3*ptcl+1] = x_ps_tgt(ptcl,1);
                xPoints[3*ptcl+2] = x_ps_tgt(ptcl,2);
            }
          }
        };
        parallel_for(ptcls, findExitFace, "search_findExitFace_intersect_3d");
      }
    }
  }

  template <class ParticleType>
  void find_next_element(o::Mesh& mesh, ParticleStructure<ParticleType>* ptcls,
                         o::Write<o::LO> elem_ids, o::Write<o::LO> next_element, o::LOs ptcl_done,
                         o::Write<o::LO> lastExit) {
    int dim = mesh.dim();
    const auto faces2elms = mesh.ask_up(dim-1, dim);
    auto e2f_vals = faces2elms.ab2b; // CSR value array
    auto e2f_offsets = faces2elms.a2ab; // CSR offset array, index by mesh edge ids
    auto setNextElm = PS_LAMBDA(const int& e, const int& pid, const int& mask) {
      if( mask > 0 && !ptcl_done[pid] ) {
        auto bridge = lastExit[pid];
        if (bridge != -1) {
            auto searchElm = elem_ids[pid];
            auto e2f_first = e2f_offsets[bridge];
            auto e2f_last = e2f_offsets[bridge+1];
            auto upFaces = e2f_last - e2f_first;
            if (upFaces == 1) { // particle at Geometry boundary
                next_element[pid] = -1;
            }
            else if (upFaces==2) { // particle at element boundary to enter next element
                assert(upFaces == 2);
                auto faceA = e2f_vals[e2f_first];
                auto faceB = e2f_vals[e2f_first + 1];
                assert(faceA != faceB);
                assert(faceA == searchElm || faceB == searchElm);
                auto nextElm = (faceA == searchElm) ? faceB : faceA;
                next_element[pid] = nextElm;
            } else {
                OMEGA_H_CHECK_PRINTF(false, "Found invalid number of adjacent elements %d\n", upFaces);
            }
        }
      }
    };
    parallel_for(ptcls, setNextElm, "pumipic_setNextElm");
  }

  template <typename Array>
  o::Real compute_tolerance_from_area(Array elmArea) {
    o::Real min_area;
    Kokkos::parallel_reduce(elmArea.size(), OMEGA_H_LAMBDA(const o::LO elm, o::Real& area) {
        if (Kokkos::fabs(elmArea[elm]) < area)
          area = Kokkos::fabs(elmArea[elm]);
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
  * Intersection mode follows line segments; BCC mode locates destinations without promising the geometric path.
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
  * @param next_element Next element over the exit face
  * @param requireIntersection True if intersection is required
  * @param inter_faces Exit faces for particles at domain boundary
  * @param inter_points Stores intersection points for particles at each face
  * @param looplimit Maximum number of iterations
  * @param debug True if debug information is printed
  * @param func Callable object to handle particles at element sides or destination
  * @param elmArea Element areas/volumes based on the mesh dimension
  * @param pctl_done if particle done (internal state) if not sure, pass empty array of size 0
  * @param given_tol Tolerance for intersection. If not positive, it is computed from the minimum element area
  * @return True if all particles are found at destination or left domain
  */
  template <class ParticleType, typename Origin, typename SegmentInt, typename Func, typename Target>
  bool trace_particle_through_mesh(o::Mesh& mesh, ParticleStructure<ParticleType>* ptcls,
                   Origin x_ps_orig, Target x_ps_tgt, SegmentInt pids,
                   o::Write<o::LO>& elem_ids,
                   o::Write<o::LO>& next_element,
                   bool requireIntersection,
                   o::Write<o::LO>& inter_faces,
                   o::Write<o::Real>& inter_points,
                   o::Write<o::LO>& lastExit,
                   int looplimit,
                   bool debug,
                   Func& func,
                   o::Reals& elmArea,
                   o::Write<o::LO>& ptcl_done,
                   std::optional<o::Real> given_tol = std::nullopt) {
    static_assert(
        std::is_invocable_r_v<
            void, Func, o::Mesh &, ParticleStructure<ParticleType> *,
            o::Write<o::LO> &, o::Write<o::LO> &, o::Write<o::LO> &, o::Write<o::LO> &,
            o::Write<o::Real> &, o::Write<o::LO> &,
            Origin, Target>,
        "Functional must accept <mesh> <ps> <elem_ids> <next_element> <inter_faces> <lastExit> <inter_points> <ptcl_done> <x_ps_orig> <x_ps_tgt>\n");

    //Initialize timer
    const auto btime = pumipic_prebarrier(mesh.comm()->get_impl());
    Kokkos::Profiling::pushRegion("pumipic_search_mesh");
    Kokkos::Timer timer;

    //Initial setup
    const auto psCapacity = ptcls->capacity();

    if (lastExit.size() == 0){
        lastExit = Omega_h::Write<Omega_h::LO>(psCapacity, -1, "search_last_exit");
    }
    bool useBcc = !requireIntersection;

    o::Real tol = 0;
    if (!given_tol.has_value() || given_tol.value() <= 0) {
        tol = compute_tolerance_from_area(elmArea);
    } else {
        tol = given_tol.value();
    }
    
    int rank;
    MPI_Comm_rank(mesh.comm()->get_impl(), &rank);
    
    const auto dim = mesh.dim();
    OMEGA_H_CHECK(mesh.dim() == 2 || mesh.dim() == 3);
    const bool initialize_parents = elem_ids.size() == 0;
    if (initialize_parents)
      elem_ids = o::Write<o::LO>(psCapacity, -1, "search_elem_ids");
    if (ptcl_done.size() == 0)
      ptcl_done = o::Write<o::LO>(psCapacity, 0, "search_ptcl_done");
    if (next_element.size() == 0)
      next_element = o::Write<o::LO>(psCapacity, -1, "next_element");
    OMEGA_H_CHECK(elem_ids.size() == psCapacity && ptcl_done.size() == psCapacity);
    OMEGA_H_CHECK(next_element.size() == psCapacity && lastExit.size() == psCapacity);
    auto initialize = PS_LAMBDA(const int parent, const int pid, const bool mask) {
      if (initialize_parents && mask) elem_ids[pid] = parent;
      if (!mask) elem_ids[pid] = -1;
      if (!mask || elem_ids[pid] == -1) ptcl_done[pid] = 1;
    };
    parallel_for(ptcls, initialize, "search_initialize");

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
    // Destination-only walking accepts any valid element as its initial guess.
    if (!useBcc)
      check_initial_parents(mesh, ptcls, x_ps_orig, x_ps_tgt, pids, elem_ids, ptcl_done, elmArea, tol, debug);
    auto initialize_next = OMEGA_H_LAMBDA(o::LO i) { next_element[i] = elem_ids[i]; };
    o::parallel_for(psCapacity, initialize_next, "search_initialize_next");

    //The prevExit skip in find_exit_face must only skip faces crossed within
    //this trace; a new segment may legitimately go back through the face the
    //previous segment exited (or now sits on), so start each trace fresh
    auto resetLastExit = OMEGA_H_LAMBDA(o::LO i) { lastExit[i] = -1; };
    o::parallel_for(lastExit.size(), resetLastExit, "reset_last_exit");

    bool found = false;
    int loops = 0;

    //Iteratively find the next element until parent element is reached for each particle
    while (!found) {
      //Find intersection face
      find_exit_face(mesh, ptcls, x_ps_orig, x_ps_tgt, elem_ids, ptcl_done, elmArea, useBcc, lastExit, inter_points, tol);
      // find next element
      find_next_element(mesh, ptcls, elem_ids, next_element, ptcl_done, lastExit);
      //Check if intersection face is exposed
      func(mesh, ptcls, elem_ids, next_element, inter_faces, lastExit, inter_points, ptcl_done, x_ps_orig, x_ps_tgt);

      //Check if all particles are found
      found = true;
      o::LOs ptcl_done_r(ptcl_done);
      auto minFlag = o::get_min(ptcl_done_r);
      if(minFlag == 0)
        found = false;
      ++loops;

      //Check iteration count
      if(!found && looplimit && loops >= looplimit) {
        Omega_h::Write<o::LO> numNotFound(1,0);
        auto ptclsNotFound = PS_LAMBDA(const int& e, const int& pid, const int& mask) {
          if( mask > 0 && !ptcl_done[pid] ) {
            auto searchElm = elem_ids[pid];
            auto ptcl = pids(pid);
            const auto ptclDest = makeVector2(pid, x_ps_orig);
            const auto ptclOrigin = makeVector2(pid, x_ps_tgt);
            if (debug) {
              printInfo("rank %d elm %d ptcl %d notFound %.15f %.15f to %.15f %.15f\n",
                     rank, searchElm, ptcl, ptclOrigin[0], ptclOrigin[1],
                     ptclDest[0], ptclDest[1]);
            }
            Kokkos::atomic_add(&(numNotFound[0]), 1);
          }
        };
        ps::parallel_for(ptcls, ptclsNotFound, "ptclsNotFound");
        Omega_h::HostWrite<o::LO> numNotFound_h(numNotFound);
        printError( "ERROR:Rank %d: loop limit %d exceeded. %d particles were "
                "not found.\n", rank, looplimit, numNotFound_h[0]);
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
    RemoveParticleOnGeometricModelExit(o::Mesh&, bool requireIntersection)
        : requireIntersection_(requireIntersection) {}

    template <typename Target>
    void operator()(o::Mesh&, ParticleStructure<ParticleType>* ptcls,
                    o::Write<o::LO>& elem_ids, o::Write<o::LO>& next_element,
                    o::Write<o::LO>& inter_faces, o::Write<o::LO>& lastExit,
                    o::Write<o::Real>&, o::Write<o::LO>& ptcl_done,
                    Segment3d, Target) const {
      const bool record_intersections = requireIntersection_;
      auto advance = PS_LAMBDA(const int, const int pid, const bool mask) {
        if (!mask || ptcl_done[pid]) return;
        const bool at_destination = lastExit[pid] == -1;
        const auto next = at_destination ? elem_ids[pid] : next_element[pid];
        const bool outside = next == -1;
        if (record_intersections)
          inter_faces[pid] = outside ? lastExit[pid] : -1;
        elem_ids[pid] = next;
        ptcl_done[pid] = at_destination || outside;
      };
      parallel_for(ptcls, advance, "search_advance");
    }

    private:
    bool requireIntersection_;
  };

  template <class ParticleType, typename Origin, typename SegmentInt, typename Target>
  bool search_mesh(o::Mesh& mesh, ParticleStructure<ParticleType>* ptcls,
                   Origin x_ps_orig, Target x_ps_tgt, SegmentInt pids,
                   o::Write<o::LO>& elem_ids,
                   bool requireIntersection,
                   o::Write<o::LO>& inter_faces,
                   o::Write<o::Real>& inter_points,
                   int looplimit,
                   int debug) {
    RemoveParticleOnGeometricModelExit<ParticleType, Origin> native_handler(mesh, requireIntersection);
    o::Reals elmArea = measure_elements_real(&mesh);
    o::Write<o::LO> lastExit (ptcls->capacity(), -1, "lastExit");
    o::Write<o::LO> next_element(0, "next_element");
    o::Write<o::LO> ptcl_done(0, "ptcl_done");
    return trace_particle_through_mesh(mesh, ptcls, x_ps_orig, x_ps_tgt, pids, elem_ids, next_element, requireIntersection,
                           inter_faces, inter_points, lastExit, looplimit, debug, native_handler, elmArea, ptcl_done);
  }
}
#endif

#ifndef PUMIPIC_ADJACENCY_HPP
#define PUMIPIC_ADJACENCY_HPP

#include <iostream>
#include <cstdlib>
#include <ctime>
#include "Omega_h_for.hpp"
#include "Omega_h_adj.hpp"
#include "Omega_h_element.hpp"

#include <SellCSigma.h>
#include <SCS_Macros.h>

#include "pumipic_utils.hpp"
#include "pumipic_constants.hpp"
#include "pumipic_kktypes.hpp"

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
          const Omega_h::LO iface, Omega_h::Few<Omega_h::Vector<DIM>, 3> &abc) {
   //face_vert:0,2,1; 0,1,3; 1,2,3; 2,0,3
    OMEGA_H_CHECK(iface<4 && iface>=0);
    abc[0] = M[Omega_h::simplex_down_template(DIM, FDIM, iface, 0)];
    abc[1] = M[Omega_h::simplex_down_template(DIM, FDIM, iface, 1)];
    abc[2] = M[Omega_h::simplex_down_template(DIM, FDIM, iface, 2)];
}

OMEGA_H_INLINE void get_edge_coords(const Omega_h::Few<Omega_h::Vector<DIM>, 3> &abc,
          const Omega_h::LO iedge, Omega_h::Few<Omega_h::Vector<DIM>, 2> &ab) {
   //edge_vert:0,1; 1,2; 2,0
    ab[0] = abc[Omega_h::simplex_down_template(FDIM, 1, iedge, 0)];
    ab[1] = abc[Omega_h::simplex_down_template(FDIM, 1, iedge, 1)];
}


// WARNING: this doesn't give vertex ordering right, so surface normal may be wrong
OMEGA_H_DEVICE o::Matrix<3, 3> get_face_of_tet(const o::LOs& mesh2verts, 
  const o::Reals& coords, o::LO elem, o::LO findex) {
  o::Matrix<3, 3> face;
  auto tetv2v = o::gather_verts<4>(mesh2verts, elem);
  auto tet = o::gather_vectors<4, 3>(coords, tetv2v);
  get_face_coords(tet, findex, face);
  return face;
}

// WARNING: check vertex ordering right, so surface normal may be wrong
OMEGA_H_INLINE void check_face(const Omega_h::Matrix<DIM, 4> &M,
    const Omega_h::Few<Omega_h::Vector<DIM>, 3>& face, const Omega_h::LO faceid) {
    Omega_h::Few<Omega_h::Vector<DIM>, 3> abc;
    get_face_coords( M, faceid, abc);
    OMEGA_H_CHECK(true == compare_array(abc[0].data(), face[0].data(), DIM)); //a
    OMEGA_H_CHECK(true == compare_array(abc[1].data(), face[1].data(), DIM)); //b
    OMEGA_H_CHECK(true == compare_array(abc[2].data(), face[2].data(), DIM)); //c
}

// BCC not in order of its corresp. opp. vertexes. BCC of tet(iface, xpoint)
//TODO Warning: Check opposite_template use in this before using
OMEGA_H_INLINE bool find_barycentric_tet( const Omega_h::Matrix<DIM, 4> &Mat,
     const Omega_h::Vector<DIM> &pos, Omega_h::Vector<4> &bcc, 
     bool debug=false) {
  for(Omega_h::LO i=0; i<4; ++i) bcc[i] = -1;

  Omega_h::Real vals[4];
  Omega_h::Few<Omega_h::Vector<DIM>, 3> abc;
  for(Omega_h::LO iface=0; iface<4; ++iface) {
    get_face_coords(Mat, iface, abc);
    auto vab = abc[1] - abc[0]; //b - a;
    auto vac = abc[2] - abc[0]; //c - a;
    auto vap = pos - abc[0]; // p - a;
    vals[iface] = osh_dot(vap, Omega_h::cross(vac, vab)); //ac, ab
  }
  //volume using bottom face=0
  get_face_coords(Mat, 0, abc);
  auto vtx3 = Omega_h::simplex_opposite_template(DIM, FDIM, 0);
  OMEGA_H_CHECK(3 == vtx3);
  // abc in order, for bottom face: M[0], M[2](=abc[1]), M[1](=abc[2])
  Omega_h::Vector<DIM> cross_ac_ab = Omega_h::cross(abc[2]-abc[0], abc[1]-abc[0]);
  Omega_h::Real vol6 = osh_dot(Mat[vtx3]-Mat[0], cross_ac_ab);
  if(debug)
    print_few_vectors(abc);
  Omega_h::Real inv_vol = 0.0;
  if(vol6 > EPSILON) // TODO tolerance
    inv_vol = 1.0/vol6;
  else {
    return 0;
  }
  bcc[0] = inv_vol * vals[0]; //for face0, cooresp. to its opp. vtx.
  bcc[1] = inv_vol * vals[1];
  bcc[2] = inv_vol * vals[2];
  bcc[3] = inv_vol * vals[3]; // 1-others
  return 1; //success
}


// BCC not in order of its corresp. vertexes. BCC of triangle (iedge, xpoint)
// corresp. to vertex obtained from simplex_opposite_template(FDIM, 1, iedge) ?
OMEGA_H_INLINE bool find_barycentric_tri_simple(
  const Omega_h::Few<Omega_h::Vector<DIM>, 3> &abc,
  const Omega_h::Vector<3> &xpoint, Omega_h::Vector<3> &bc) {
  Omega_h::Vector<DIM> a = abc[0];
  Omega_h::Vector<DIM> b = abc[1];
  Omega_h::Vector<DIM> c = abc[2];
  Omega_h::Vector<DIM> cross = 1/2.0 * Omega_h::cross(b-a, c-a); //NOTE order
  Omega_h::Vector<DIM> norm = Omega_h::normalize(cross);
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
OMEGA_H_INLINE bool line_triangle_intx_simple (
  const Omega_h::Few<Omega_h::Vector<3>, 3> &abc, 
  const Omega_h::Vector<3> &origin, const Omega_h::Vector<3> &dest,
  Omega_h::Vector<3> &xpoint, Omega_h::Real& dproj, bool reverse=false, 
  Omega_h::Real tol=0, bool debug=false) {
  for(int i=0; i<3; ++i)
    xpoint[i] = 0;

  bool found = false;
  Omega_h::Vector<3> line = dest - origin;
  Omega_h::Vector<3> edge0 = abc[1] - abc[0];
  Omega_h::Vector<3> edge1 = abc[2] - abc[0];
  Omega_h::Vector<3> normv = Omega_h::cross(edge0, edge1);
  if(reverse) {
    normv = -1*normv;
    if(debug)
      printf("Surface normal is flipped\n");
  }
  Omega_h::Vector<3> snorm_unit = Omega_h::normalize(normv);
  Omega_h::Real dist2plane = osh_dot(abc[0] - origin, snorm_unit);
  Omega_h::Vector<3> plane2dest = dest - abc[0];
  Omega_h::Real proj_end = osh_dot(snorm_unit, plane2dest);
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
        printf(" Found %d bcc+ %d par_t= %.10f dist2plane= %.10f "
           "projline= %.10f proj_out_line %.10f \n", found, res, 
           par_t, dist2plane, dproj, proj_end);
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

OMEGA_H_DEVICE o::Matrix<3, 3> gatherVectors3x3(o::Reals const& a, 
  o::Few<o::LO, 3> v) {
  return o::gather_vectors<3, 3>(a, v);
}
OMEGA_H_DEVICE o::Matrix<3, 4> gatherVectors4x3(o::Reals const& a, 
  o::Few<o::LO, 4> v) {
  return o::gather_vectors<4, 3>(a, v);
}

template < class ParticleType>
bool search_mesh(o::Mesh& mesh, ps::SellCSigma< ParticleType >* scs,
  Segment3d x_scs_d, Segment3d xtgt_scs_d, SegmentInt pid_scs_d,
  o::Write<o::LO>& elem_ids, o::Write<o::Real>& xpoints_d,
  o::Write<o::LO>& xface_ids_d, int looplimit=0) {
           
  bool debug =0;
  o::Real tol = 1.0e-10;

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
  // ptcl_done[i] = 1 : particle hit a boundary or reached its destination
  o::Write<o::LO> ptcl_done(scsCapacity, 1, "ptcl_done");
  // store the next parent for each particle
  o::Write<o::LO> elem_ids_next(scsCapacity,-1);
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
  scs->parallel_for(lamb);

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
        auto ptcl = pid_scs_d(pid);
        auto tetv2v = o::gather_verts<4>(mesh2verts, elmId);
        auto M = gatherVectors4x3(coords, tetv2v);
        auto dest = makeVector3(pid, xtgt_scs_d);
        auto orig = makeVector3(pid, x_scs_d);
        o::Vector<4> bcc;
        if(loops == 0) {
          //make sure particle origin is in initial element
          find_barycentric_tet(M, orig, bcc);
          if(!all_positive(bcc, tol)) {
            printf("Warning: Particle not in this element at loops=0"
              "\tptcl %d elem %d\n", ptcl, elmId);
            print_osh_vector(orig, "orig");
            print_osh_vector(dest, "dest");
            print_osh_vector(bcc, "bcc");
            //OMEGA_H_CHECK(false);
          }
        }
        bool detected = false;
        find_barycentric_tet(M, dest, bcc);
        // TODO tolerance
        if(all_positive(bcc, tol)) {
          if(debug)
            printf("ptcl %d is in destination elm %d\n", ptcl, elmId);
          elem_ids_next[pid] = elmId; //elem_ids[pid];
          ptcl_done[pid] = 1;
        } else {
          if(debug)
            printf("ptcl %d checking adj elms\n", ptcl);
          auto dproj = o::zero_vector<4>();
          auto xpoints = o::zero_vector<12>();
          o::LO exposed_faces[4];
          o::LO xface_ids[4];
          o::LO min_bcc_elem = -1;
          dproj[0] = dproj[1] = dproj[2] = dproj[3] = -1;
          //get element ID
          auto dface_ind = dual_elems[elmId];
          const auto beg_face = elmId *4;
          const auto end_face = beg_face +4;
          o::LO findex = 0;
          for(auto iface = beg_face; iface < end_face; ++iface) {
            const auto face_id = down_r2fs[iface];
            auto xpoint = o::zero_vector<3>();
            o::LO exposed = side_is_exposed[face_id];
            exposed_faces[findex] = exposed;
            xface_ids[findex] = face_id;
            auto fv2v = o::gather_verts<3>(face_verts, face_id);
            const auto face = gatherVectors3x3(coords, fv2v);
            o::LO matInd1 = getfmap(findex*2);
            o::LO matInd2 = getfmap(findex*2+1);
            bool flip = true;
            if(fv2v[1] == tetv2v[matInd1] && fv2v[2] == tetv2v[matInd2])
              flip = false;
            detected = line_triangle_intx_simple(face, orig, dest, xpoint, 
              dproj[findex], flip, true); //debug TODO
            for(o::LO i=0; i<3; ++i)
              xpoints[findex*3+i] = xpoint[i];

            if(debug) {
              printf("\t :ptcl %d faceid %d flipped %d exposed %d detected %d\n", ptcl, 
                face_id, flip, exposed, detected);
              for(int i=0; i<3; ++i)
               printf("face:%d %.15f %.15f %.15f\n", i, face[i][0], face[i][1], face[i][2]);
              printf("ptcl: %d orig,dest: %.15f %.15f %.15f %.15f %.15f %.15f \n", ptcl, orig[0], 
                orig[1], orig[2], dest[0],dest[1],dest[2]);
            }
            if(detected && exposed) {
              ptcl_done[pid] = 1;
              for(o::LO i=0; i<3; ++i)
                xpoints_d[pid*3+i] = xpoint[i];
              xface_ids_d[pid] = face_id;
              elem_ids_next[pid] = -1;
              if(debug)
                printf("ptcl %d faceid %d detected and exposed, next parent elm %d\n",
                    ptcl, face_id, elem_ids_next[pid]);
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
            o::LO min_ind = min_index(bcc, 4);

            // save next element based on the smallest BCC,
            if(!exposed) {
              if(debug)
                printf("ptcl %d faceid %d !detected and !exposed\n", ptcl, face_id);
              o::LO min_ind = min_index(bcc, 4);
              if(findex == min_ind) {
                min_bcc_elem = dual_faces[dface_ind];
              }
              ++dface_ind;
            }
            ++findex;
          } //for iface

          if(!detected) {
            printf("ptcl %d not detected; using max dproj\n", ptcl);
            o::LO max_ind = max_index(dproj, 4);
            if(dproj[max_ind]>=0) {
              auto fid = xface_ids[max_ind];
              if(exposed_faces[max_ind]) {
                elem_ids_next[pid] = -1;
                for(o::LO i=0; i<3; ++i)
                  xpoints_d[pid*3+i] = xpoints[max_ind*3+i];            
                xface_ids_d[pid] = fid;
                ptcl_done[pid] = 1;
              } else { //if(min_bcc_elem >= 0) {
                elem_ids_next[pid] = dual_faces[fid]; //min_bcc_elem;
                if(debug)
                  printf("ptcl %d elem_ids_next %d min_bcc_elem %d\n", ptcl, 
                   elem_ids_next[pid], min_bcc_elem);
              }
            } else {
              // current elem, but bcc failed to detect it on face/corner
              printf("WARNING: particle %d leaked \n", ptcl);
              elem_ids_next[pid] = -1;
              ptcl_done[pid] = 1;
            }
          }
        } //else not in current element
      } //if active particle
    };

    scs->parallel_for(lamb);
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
  }
  if(debug)
    fprintf(stderr, "\t: loops %d\n", loops);

  if(debug && !found) {
    auto lamb = SCS_LAMBDA(const int& e, const int& pid, const int& mask) {
      if(mask > 0 && ptcl_done[pid] ==0) {
        auto tetv2v = o::gather_verts<4>(mesh2verts, e);
        auto M = gatherVectors4x3(coords, tetv2v);
        auto dest = makeVector3(pid, xtgt_scs_d);
        auto orig = makeVector3(pid, x_scs_d);
        o::Vector<4> bcc;
        find_barycentric_tet(M, orig, bcc);
        auto ptcl = pid_scs_d(pid);
        printf("ptcl %d elem %d orig %.6f %.6f %.6f dest %.6f %.6f %.6f\n",
          ptcl, e, orig[0], orig[1], orig[2], dest[0], dest[1], dest[2]);
        printf("ERROR:notFound: pid %3d elem_ids %6d\n", ptcl, elem_ids[pid]);
      }
    };
    scs->parallel_for(lamb);
  }
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

typedef o::Vector<3> Vector;

//TODO test this function, is this needed ?
OMEGA_H_INLINE o::LO find_closest_point_on_triangle_with_normal(
  const o::Few< o::Vector<3>, 3> &abc,
  const o::Vector<3> &p, o::Vector<3> &q, o::LO verbose = 0) {

  // Check if P in vertex region outside A
  Vector a = abc[0];
  Vector b = abc[1];
  Vector c = abc[2];

  Vector ab = b - a;
  Vector ac = c - a;
  Vector bc = c - b;
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
  Vector n = o::cross(b - a, c - a);
  Vector temp = o::cross(a - p, b - p);
  float vc = osh_dot(n, temp);
  // If P outside AB and within feature region of AB,
  // return projection of P onto AB
  if (vc <= 0.0 && snom >= 0.0 && sdenom >= 0.0){
    q = a + snom / (snom + sdenom) * ab;
    return EDGEAB;
  }
  // P is outside (or on) BC if the triple scalar product [N PB PC] <= 0
  Vector temp1 = o::cross(b - p, c - p);
  float va = osh_dot(n, temp1);
  // If P outside BC and within feature region of BC,
  // return projection of P onto BC
  if (va <= 0.0 && unom >= 0.0 && udenom >= 0.0){
    q = b + unom / (unom + udenom) * bc;
    return EDGEBC;
  }
  // P is outside (or on) CA if the triple scalar product [N PC PA] <= 0
  Vector temp2 = o::cross(c - p, a - p);
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
OMEGA_H_DEVICE o::LO find_closest_point_on_triangle( const o::Few< o::Vector<3>, 3> &abc, 
  const o::Vector<3> &ptp, o::Vector<3> &ptq, o::LO verbose = 0) {
  
  //o::LO verbose = 1;
  o::LO region = -1;
  // Check if P in vertex region outside A
  o::Vector<3> pta = abc[0];
  o::Vector<3> ptb = abc[1];
  o::Vector<3> ptc = abc[2];

  if(verbose >2){
    print_osh_vector(pta, "pta");
    print_osh_vector(ptb, "ptb");
    print_osh_vector(ptc, "ptc");
    print_osh_vector(ptp, "ptp");
  }

  o::Vector<3> vab = ptb - pta;
  o::Vector<3> vac = ptc - pta;
  o::Vector<3> vap = ptp - pta;
  o::Real d1 = osh_dot(vab, vap);
  o::Real d2 = osh_dot(vac, vap);
  if (d1 <= 0 && d2 <= 0) {
    // barycentric coordinates (1,0,0)
    ptq = pta;
    region = VTXA;
    if(verbose >2){
      print_osh_vector(ptq, "QA");
      print_osh_vector(ptp, "P");
    }
    return VTXA; 
  }

  // Check if P in vertex region outside B
  o::Vector<3> vbp = ptp - ptb;
  o::Real d3 = osh_dot(vab, vbp);
  o::Real d4 = osh_dot(vac, vbp);
  if(region <0 && d3 >= 0 && d4 <= d3){ 
    // barycentric coordinates (0,1,0)
    ptq = ptb;
    region = VTXB;
    if(verbose >2)
      print_osh_vector(ptq, "QB");
    return VTXB; 
  }

  // Check if P in edge region of AB, if so return projection of P onto AB
  o::Real vc = d1*d4 - d3*d2;
  if(region <0 && vc <= 0 && d1 >= 0 && d3 <= 0) {
    o::Real v = d1 / (d1 - d3);
    // barycentric coordinates (1-v,v,0)
    ptq = pta + v * vab; 
    region = EDGEAB;
    //return EDGEAB;
  }

  // Check if P in vertex region outside C
  o::Vector<3> vcp = ptp - ptc;
  o::Real d5 = osh_dot(vab, vcp);
  o::Real d6 = osh_dot(vac, vcp);
  if(region <0 && d6 >= 0 && d5 <= d6) { 
    // barycentric coordinates (0,0,1)
    ptq = ptc; 
    region = VTXC;
    if(verbose >2)
      print_osh_vector(ptq, "QAB");
    return VTXC;
  }

  // Check if P in edge region of AC, if so return projection of P onto AC
  o::Real vb = d5*d2 - d1*d6;
  if(region <0 && vb <= 0 && d2 >= 0 && d6 <= 0) {
    o::Real w = d2 / (d2 - d6);
    // barycentric coordinates (1-w,0,w)
    ptq = pta + w * vac; 
    region = EDGEAC;
    if(verbose >2)
      print_osh_vector(ptq, "QAC");
    return EDGEAC;
  }

  // Check if P in edge region of BC, if so return projection of P onto BC
  o::Real va = d3*d6 - d5*d4;
  if(region <0 && va <= 0 && (d4 - d3) >= 0 && (d5 - d6) >= 0) {
    o::Real w = (d4 - d3) / ((d4 - d3) + (d5 - d6));
    // barycentric coordinates (0,1-w,w)
    ptq =  ptb + w * (ptc - ptb); 
    region = EDGEBC;
    if(verbose >2)
      print_osh_vector(ptq, "QBC");
    return EDGEBC;
  }

  // P inside face region. Compute Q through its barycentric coordinates (u,v,w)
  if(region <0) {
    o::Real inv = 1 / (va + vb + vc);
    o::Real v = vb * inv;
    o::Real w = vc * inv;
    // u*a + v*b + w*c, u = va * inv = 1 - v - w
    ptq =  pta + v * vab+ w * vac;
    region = TRIFACE;
    if(verbose >2) 
      print_osh_vector(ptq, "QABC");
    return TRIFACE;
  }
  if(verbose >2){
    print_osh_vector(ptq, "Q");
    printf("d's:: %0.3f %0.3f %0.3f %0.3f %0.3f %0.3f \n", d1, d2, d3, d4, d5, d6);
  }
  return region;
}


// type =1 for  interior, 2=bdry
OMEGA_H_DEVICE o::LO get_face_type_ids_of_elem(const o::LO elem, 
  const o::LOs &down_r2f, const o::Read<o::I8> &side_is_exposed, 
  o::LO (&fids)[4], const o::LO type) {

  o::LO nf = 0;
  const auto beg_face = elem *4;
  const auto end_face = beg_face +4;
  for(o::LO fi = beg_face; fi < end_face; ++fi){
    const auto fid = down_r2f[fi];
    if( (type==1 && !side_is_exposed[fid]) ||
        (type==2 &&  side_is_exposed[fid]) ) {
      fids[nf] = fid;
      ++nf;
    }
  }
  return nf;
}


//TODO remove print by argument
OMEGA_H_DEVICE void get_face_data_by_id(const Omega_h::LOs &face_verts, 
  const Omega_h::Reals &coords,  const o::LO face_id, o::Real (&fdat)[9],
  const bool p=false) {

  auto fv2v = Omega_h::gather_verts<3>(face_verts, face_id);
  const auto face = Omega_h::gather_vectors<3, 3>(coords, fv2v);
  for(auto i=0; i<3; ++i){
    for(auto j=0; j<3; ++j){
      fdat[i*3+j] = face[i][j];
      if(p){
        printf(" fdat[%d]=%.3f i_%d j_%d face[i][j]=%.3f \n", i*3+j, 
          fdat[i], i, j, face[i][j]);
      }
    }
  }
}


OMEGA_H_DEVICE bool check_if_face_within_dist_to_tet(const o::Matrix<DIM, 4> &tet, 
  const Omega_h::LOs &face_verts, const Omega_h::Reals &coords, const o::LO face_id,
  const o::Real depth = 0.001) {

  //TODO test this copying 
  /*
  o::Few<o::Vector<3>, 3> face;
  for(o::LO i=0; i< dim; ++i){
    //face[i] = {data[beg+i*3], data[beg+i*3+1], data[beg+i*3+2]};
    for(o::LO j=0; j< dim; ++j){
      face[i][j] = data[beg + j]; 
    }
  }
  */

  auto fv2v = Omega_h::gather_verts<3>(face_verts, face_id); //Few<LO, 3>
  const auto face = Omega_h::gather_vectors<3, 3>(coords, fv2v);

  for(o::LO i=0; i<3; ++i){ //3 vtx of face
    for(o::LO j=0; j<4; ++j){ //4 vtx of tet
      o::Vector<3> dv = face[i] - tet[j];
      o::Real d2 = osh_dot(dv, dv);
      

     // printf("face_%0.3f,%0.3f,%0.3f tet_%0.3f,%0.3f,%0.3f \n", 
     //   face[i][0],face[i][1], face[i][2],tet[j][0],tet[j][1],tet[j][2]);
      
      // printf("dist_%0.5f depth_%0.5f \n", sqrt(d2), depth);

      if(d2 <= depth*depth){
        return true;
      }
    }
  }
  return false;
}

// TODO use device to call device function
/*
inline void test_find_closest_point_on_triangle(){
  constexpr int nTris = 1;
  o::Few<o::Few< o::Vector<3>, 3>, nTris> abcs{{{1,0,0},{2,0,0},{1.5,1,0}}}; 

  constexpr int nPts = 7;
  o::Few<o::Vector<3>, nPts> pts{{0,-0.5,0}, {0,2,0}, {0,0.2,0},
                                 {1.2,0,0}, {1.2,0.3,0},
                                 {0, -1, 2}, {1.5, 0.5, -2}};
  o::LO regions[nTris * nPts] = {0, 2, 0, 3, 6, 0, 6};

  o::Vector<3> ptq;
  o::Few< o::Vector<3>, 3> abc;
  for(int i=0; i<nTris; ++i){
    abc = abcs[i];
    for(int j=0; j<nPts; ++j){
      o::Vector<3> ptp = pts[j];
      auto v = find_closest_point_on_triangle(abc, ptp, ptq, 3);
  
      std::cout << "\nTest_Pt_on_Tri: ";
      print_osh_vector(abc[0], "A");
      print_osh_vector(abc[1], "B");
      print_osh_vector(abc[2], "C");    
      print_osh_vector(ptp, "P");
      print_osh_vector(ptq, "Nearest_pt Q");
      std::cout << " reg: " << v << "\n";

      OMEGA_H_CHECK(v == regions[i*nTris + j]);
    }
  }
}
*/

} //namespace

#endif //define

#ifndef PUMIPIC_ADJACENCY_HPP
#define PUMIPIC_ADJACENCY_HPP

#include <iostream>

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

//TODO use .get() to access data ?
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

// BC coords are not in order of its corresp. opp. vertexes. Bccoord of tet(iface, xpoint)
//TODO Warning: Check opposite_template use in this before using
OMEGA_H_INLINE bool find_barycentric_tet( const Omega_h::Matrix<DIM, 4> &Mat,
     const Omega_h::Vector<DIM> &pos, Omega_h::Vector<4> &bcc)
{
  bool debug = false;
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

    if(debug) {
      printf("vol: %f for points_of_this_TET:\n", vals[iface]);
      print_array(abc[0].data(),3);
      print_array(abc[1].data(),3);
      print_array(abc[2].data(),3);
      print_array(pos.data(),3, "point");
      printf("\n");
    }
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
    if(debug)
      printf("Volume %f too low \n", vol6);

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
    Omega_h::Vector<DIM> &xpoint)
{
  const bool debug = 0;
  for(int i=0; i<DIM; ++i)
    xpoint[i] = 0;

  //Boundary exclusion. Don't set it globally and change randomnly.
  const Omega_h::Real bound_intol = 0;//SURFACE_EXCLUDE; //TODO optimum value ?

  bool found = false;
  const Omega_h::Vector<DIM> line = dest - origin;
  const Omega_h::Vector<DIM> edge0 = abc[1] - abc[0];
  const Omega_h::Vector<DIM> edge1 = abc[2] - abc[0];
  Omega_h::Vector<DIM> normv = Omega_h::cross(edge0, edge1);

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
    printf("Line and plane are parallel \n");
  }
  return found;
}

OMEGA_H_DEVICE o::Vector<3> makeVector3(int pid, kkFp3View xyz) {
  o::Vector<3> v;
  for(int i=0; i<3; ++i)
    v[i] = xyz(pid,i);
  return v;
}

//2,3 nodes of faces. 0,2,1; 0,1,3; 1,2,3; 2,0,3
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

OMEGA_H_DEVICE o::Matrix<3, 3> get_face_of_tet(const o::LOs& mesh2verts, 
  const o::Reals& coords, o::LO elem, o::LO findex) {
  o::Matrix<3, 3> face;
  auto tetv2v = o::gather_verts<4>(mesh2verts, elem);
  auto tet = gatherVectors4x3(coords, tetv2v);        
  get_face_coords(tet, findex, face);
  return face;
}

/*
TODO: Search should exclude particle if on surface of an element by SURFACE_EXCLUDE. 
When an intersection on bdry is found, that face id is to be stored to use in surface model.
*/

//How to avoid redefining the MemberType? each application will define it
//differently. Templating search_mesh with
//template < typename ParticleType >
//results in an error on getSCS<> as an unresolved function.

// TODO Temporary
typedef particle_structs::MemberTypes < Vector3d, Vector3d, int,  int, Vector3d, 
       Vector3d, Vector3d> ParticleType;

template < class ParticleType>
bool search_mesh(o::Mesh& mesh, particle_structs::SellCSigma< ParticleType >* scs,
    o::Write<o::LO>& elem_ids, int looplimit=0) {
  const auto dual = mesh.ask_dual();
  const auto down_r2f = mesh.ask_down(3, 2);
  const auto side_is_exposed = mark_exposed_sides(&mesh);
  const auto mesh2verts = mesh.ask_elem_verts();
  const auto coords = mesh.coords();
  const auto face_verts =  mesh.ask_verts_of(2);

  const auto down_r2fs = down_r2f.ab2b;
  const auto dual_faces = dual.ab2b;
  const auto dual_elems = dual.a2ab;

  scs->transferToDevice();  //TODO user tuples should be allocated on device by default
  const auto scsCapacity = scs->offsets[scs->num_slices];
  kkFp3View x_scs_d("x_scs_d", scsCapacity);
  hostToDeviceFp(x_scs_d, scs->template getSCS<0>() );
  kkFp3View xtgt_scs_d("xtgt_scs_d", scsCapacity);
  hostToDeviceFp(xtgt_scs_d, scs->template getSCS<1>() );

  kkLidView pid_d("pid_d", scsCapacity);
  hostToDeviceLid(pid_d, scs->template getSCS<2>() );

  // ptcl_done[i] = 1 : particle i has hit a boundary or reached its destination
  o::Write<o::LO> ptcl_done(scsCapacity, 1, "ptcl_done");
  // particle intersection points
  o::Write<o::Real> xpoints(3*scsCapacity, -1.0);
  // store the next parent for each particle
  o::Write<o::LO> elem_ids_next(scsCapacity,-1);
  auto lamb = SCS_LAMBDA(const int& e, const int& pid, const int& mask) {
    if(mask > 0) {
      elem_ids[pid] = e;
      ptcl_done[pid] = 0;
      printf("pid %3d mask %1d elem_ids %6d\n", pid, mask, elem_ids[pid]);
    } else {
      elem_ids[pid] = -1;
      ptcl_done[pid] = 1;
    }
  };
  scs->parallel_for(lamb);

  const int debug = 0;

  bool found = false;
  int loops = 0;
  while(!found) {
    if(debug) {
      fprintf(stderr, "------------ %d ------------\n", loops);
    }

    auto lamb = SCS_LAMBDA(const int& e, const int& pid, const int& mask) {
      auto elmId = e;
      auto tetv2v = o::gather_verts<4>(mesh2verts, elmId);
      auto M = gatherVectors4x3(coords, tetv2v);
      //inactive particle that is still moving to its target position
      if( mask > 0 && !ptcl_done[pid] ) {
        if(debug)
          printf("Elem %d ptcl: %d\n", elmId, pid);
        if(elmId != elem_ids[pid]) {
          elmId = elem_ids[pid];
          tetv2v = o::gather_verts<4>(mesh2verts, elmId);
          M = gatherVectors4x3(coords, tetv2v);
          if(debug)
            printf("Elem %d ptcl: %d\n", elmId, pid);
        }
        const o::Vector<3> orig = makeVector3(pid, x_scs_d);
        const o::Vector<3> dest = makeVector3(pid, xtgt_scs_d);
        Omega_h::Vector<4> bcc;
        if(loops == 0) {
          printf("orig %.3f %.3f %.3f dest %.3f %.3f %.3f\n",
              orig[0], orig[1], orig[2],
              dest[0], dest[1], dest[2]);
          //Check particle origin containment in beginning element
          find_barycentric_tet(M, orig, bcc);
          OMEGA_H_CHECK(all_positive(bcc, 0));
        }
        //check if the destination is in this element
        find_barycentric_tet(M, dest, bcc);
        if(all_positive(bcc, 0)) {
          if(debug)
            printf("ptcl %d is in destination elm %d\n", pid, elmId);
          elem_ids_next[pid] = elem_ids[pid];
          ptcl_done[pid] = 1;
        } else {
          if(debug)
            printf("ptcl %d checking adj elms\n", pid);
          //get element ID
          auto dface_ind = dual_elems[elmId];
          const auto beg_face = elmId *4;
          const auto end_face = beg_face +4;
          o::LO findex = 0;

          for(auto iface = beg_face; iface < end_face; ++iface) {
            const auto face_id = down_r2fs[iface];
            o::Vector<3> xpoint = o::zero_vector<3>();
            const auto face = get_face_of_tet(mesh2verts, coords, e, findex);
            bool detected = line_triangle_intx_simple(face, orig, dest, xpoint);
            if(debug)
              printf("ptcl %d faceid %d detected %d\n", pid, face_id, detected);

            if(detected && side_is_exposed[face_id]) {
              ptcl_done[pid] = 1;
              for(o::LO i=0; i<3; ++i)
                xpoints[pid*3+i] = xpoint[i];
              elem_ids_next[pid] = -1;
              if(debug) {
                printf("ptcl %d faceid %d detected and exposed, next parent elm %d\n",
                    pid, face_id, elem_ids_next[pid]);
              }
              break;
            } else if(detected && !side_is_exposed[face_id]) {
              auto adj_elem  = dual_faces[dface_ind];
              elem_ids_next[pid] = adj_elem;
              if(debug) {
                printf("ptcl %d faceid %d detected and !exposed, next parent elm %d\n",
                    pid, face_id, elem_ids_next[pid]);
              }
              break;
            }

            // No line triangle intersection found for the current face
            // appears to be a guess at the next element based on the smallest BCC
            if(!side_is_exposed[face_id]) {
              if(debug)
                printf("ptcl %d faceid %d !detected and !exposed\n", pid, face_id);
              ++dface_ind;
              const o::LO min_ind = min_index(bcc, 4);
              if(findex == min_ind) {
                elem_ids_next[pid] = dual_faces[dface_ind];
                if(debug) {
                  printf("WARNING ptcl %d faceid %d !detected and !exposed, next parent elm %d\n",
                    pid, face_id, elem_ids_next[pid]);
                }
                // No 'break' here, since other faces to be checked.
                // TODO This occurs by error, and this fallback option to be used for debugging
                // by setting correct tolerances that match with it in other function calls.
              }
            }

            ++findex;
          } //for iface
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
      if(debug) fprintf(stderr, "loop limit %d exceeded\n", looplimit);
      break;
    }
  }

  return found;
}

// Vector is not needed for bcc, TODO

// To interpoalte field stored at vertices. Field has dof components, and 
// stored in order 0,1,2,3 at tet's vertices. BCC in order of faces
OMEGA_H_DEVICE Omega_h::Real interpolateTet(const Omega_h::Reals &field, 
  const Omega_h::Vector<4> &bcc, const Omega_h::LO elem, o::LO dof=1, o::LO comp=0)
{
  OMEGA_H_CHECK(all_positive(bcc)==1);
  Omega_h::Real val = 0;
  for(Omega_h::LO fi=0; fi<4; ++fi) //faces
  {
      Omega_h::LO d = Omega_h::simplex_opposite_template(3,2,fi); //3,2,0,1
      Omega_h::LO fd = 4*dof*elem + d*dof + comp;
      val = val + bcc[fi]*field[fd];
  }
  return val;
}


OMEGA_H_DEVICE void interpolate3dFieldTet(const Omega_h::Reals &field, 
  const Omega_h::Vector<4> &bcc, const Omega_h::LO elem, Omega_h::Vector<3> &fv)
{
  for(int i=0; i<3; ++i) {
      fv[i] = interpolateTet(field, bcc, elem, 3, i);
  }
}


OMEGA_H_DEVICE void findTetCoords(const Omega_h::LOs &mesh2verts,
const Omega_h::Reals &coords, const Omega_h::LO elem, Omega_h::Matrix<DIM, 4> &mat)
{
  const auto tetv2v = Omega_h::gather_verts<4>(mesh2verts, elem);
  mat = Omega_h::gather_vectors<4, 3>(coords, tetv2v);
}

OMEGA_H_DEVICE void findBCCoordsInTet(const Omega_h::Reals &coords, 
  const Omega_h::LOs &mesh2verts, const Omega_h::Vector<3> &xyz, 
  const Omega_h::LO elem, Omega_h::Vector<4> &bcc)
{
  Omega_h::Matrix<3, 4> mat;
  findTetCoords(mesh2verts, coords, elem, mat);
  const bool res = find_barycentric_tet(mat, xyz, bcc);
  OMEGA_H_CHECK(res==0);
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
  const o::Vector<3> &p, o::Vector<3> &q, o::LO verbose = 0){

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
OMEGA_H_INLINE o::LO find_closest_point_on_triangle( const o::Few< o::Vector<3>, 3> &abc, 
  const o::Vector<3> &ptp, o::Vector<3> &ptq, o::LO verbose = 0) {
  
  //o::LO verbose = 1;
  o::LO region = -1;
  // Check if P in vertex region outside A
  o::Vector<3> pta = abc[0];
  o::Vector<3> ptb = abc[1];
  o::Vector<3> ptc = abc[2];

  if(verbose >2){
    print_osh_vector(pta, "pta", false);
    print_osh_vector(ptb, "ptb", false);
    print_osh_vector(ptc, "ptc", false);
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
      print_osh_vector(ptq, "QA", false);
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
    print_osh_vector(ptq, "Q", false);
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
  const bool p=false){

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
  const o::Real depth = 0.001){

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

// TODO move to unit_tests
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
      print_osh_vector(abc[0], "A", false);
      print_osh_vector(abc[1], "B", false);
      print_osh_vector(abc[2], "C", false);    
      print_osh_vector(ptp, "P", false);
      print_osh_vector(ptq, "Nearest_pt Q", false);
      std::cout << " reg: " << v << "\n";

      OMEGA_H_CHECK(v == regions[i*nTris + j]);
    }
  }
}


} //namespace

#endif //define

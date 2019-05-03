#ifndef PUMIPIC_ADJACENCY_HPP
#define PUMIPIC_ADJACENCY_HPP

#include <iostream>

#include "Omega_h_for.hpp"
#include "Omega_h_adj.hpp"
#include "Omega_h_element.hpp"

#include "pumipic_utils.hpp"
#include "pumipic_constants.hpp"

namespace o = Omega_h;
namespace pumipic
{
  const int verbose = 0; //TODO move to pumipic_constants.hpp as inline/static

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
//retrieve face coords in the o order
OMEGA_H_INLINE void get_face_coords(const o::Matrix<DIM, 4> &M,
          const o::LO iface, o::Few<o::Vector<DIM>, 3> &abc) {

   //face_vert:0,2,1; 0,1,3; 1,2,3; 2,0,3
    OMEGA_H_CHECK(iface<4 && iface>=0);
    abc[0] = M[o::simplex_down_template(DIM, FDIM, iface, 0)];
    abc[1] = M[o::simplex_down_template(DIM, FDIM, iface, 1)];
    abc[2] = M[o::simplex_down_template(DIM, FDIM, iface, 2)];

    if(verbose > 1)
        std::cout << "face " << iface << ": \n"; 
}

OMEGA_H_INLINE void get_edge_coords(const o::Few<o::Vector<DIM>, 3> &abc,
          const o::LO iedge, o::Few<o::Vector<DIM>, 2> &ab) {

   //edge_vert:0,1; 1,2; 2,0
    ab[0] = abc[o::simplex_down_template(FDIM, 1, iedge, 0)];
    ab[1] = abc[o::simplex_down_template(FDIM, 1, iedge, 1)];
    if(verbose > 2)
        std::cout << "abc_index " << ab[0].data() << ", " << ab[1].data()
                  << " iedge:" << iedge << "\n";
}

OMEGA_H_INLINE void check_face(const o::Matrix<DIM, 4> &M,
    const o::Few<o::Vector<DIM>, 3>& face, const o::LO faceid ){

    o::Few<o::Vector<DIM>, 3> abc;
    get_face_coords( M, faceid, abc);

    if(verbose > 2) {
      print_array(abc[0].data(),3, "a");
      print_array(face[0].data(),3, "face1");
      print_array(abc[1].data(), 3, "b");
      print_array(face[1].data(), 3, "face2");
      print_array(abc[2].data(), 3, "c");
      print_array(face[2].data(), 3, "face3");
    }
    OMEGA_H_CHECK(true == compare_array(abc[0].data(), face[0].data(), DIM)); //a
    OMEGA_H_CHECK(true == compare_array(abc[1].data(), face[1].data(), DIM)); //b
    OMEGA_H_CHECK(true == compare_array(abc[2].data(), face[2].data(), DIM)); //c
}

// BC coords are not in order of its corresp. opp. vertexes. Bccoord of tet(iface, xpoint)
//TODO Warning: Check opposite_template use in this before using
OMEGA_H_INLINE bool find_barycentric_tet( const Omega_h::Matrix<DIM, 4> &mat,
     const Omega_h::Vector<DIM> &pos, Omega_h::Write<Omega_h::Real> &bcc)
{
  OMEGA_H_CHECK(!(std::isnan(pos[0]) || std::isnan(pos[1]) || std::isnan(pos[2])));

  for(Omega_h::LO i=0; i<3; ++i) bcc[i] = 0;

  Omega_h::Real vals[4];
  Omega_h::Few<Omega_h::Vector<DIM>, 3> abc;
  for(Omega_h::LO iface=0; iface<4; ++iface) // TODO last not needed
  {
    get_face_coords(mat, iface, abc);
    auto vab = abc[1] - abc[0]; //b - a;
    auto vac = abc[2] - abc[0]; //c - a;
    auto vap = pos - abc[0]; // p - a;
    vals[iface] = osh_dot(vap, o::cross(vac, vab)); //ac, ab NOTE

    if(verbose > 2) {
      std::cout << "vol: " << vals[iface] << " for points_of_this_TET:\n" ;
      print_array(abc[0].data(),3);
      print_array(abc[1].data(),3);
      print_array(abc[2].data(),3);
      print_array(pos.data(),3, "point");
      std::cout << "\n";
    }
  }
  //volume using bottom face=0
  get_face_coords(mat, 0, abc);
  auto vtx3 = o::simplex_opposite_template(DIM, FDIM, 0);
  OMEGA_H_CHECK(3 == vtx3);
  // abc in order, for bottom face: M[0], M[2](=abc[1]), M[1](=abc[2])
  o::Vector<DIM> cross_ac_ab = o::cross(abc[2]-abc[0], abc[1]-abc[0]); //NOTE
  o::Real vol6 = osh_dot(mat[vtx3]-mat[0], cross_ac_ab);
  o::Real inv_vol = 0.0;
  if(vol6 > EPSILON) // TODO tolerance
    inv_vol = 1.0/vol6;
  else {
    if(verbose > 0)  
      std::cout << "Error: Volume " << vol6 << " too small \n";
    return 0;
  }
  bcc[0] = inv_vol * vals[0]; //for face0, cooresp. to its opp. vtx.
  bcc[1] = inv_vol * vals[1];
  bcc[2] = inv_vol * vals[2];
  bcc[3] = inv_vol * vals[3]; // 1-others

  return 0; //success
}


template <typename T>
OMEGA_H_INLINE double interpolateTet(const T &field, const double (&bcc)[4],
  const Omega_h::LO ielem)
{
  //bcc in face order, field in vertex order ? TODO
  OMEGA_H_CHECK(all_positive(bcc, 4)==1);
  double val = 0;
  for(Omega_h::LO fi=0; fi<4; ++fi) //faces
  {
      Omega_h::LO d = Omega_h::simplex_opposite_template(3,2,fi); //3,2,0,1
      Omega_h::LO fd = d + 4*ielem;
      val = val + bcc[fi]*field[fd];
  }
  return val;
}

template <typename T>
OMEGA_H_INLINE void interpolate3dFieldTet(const T &fieldx, const T &fieldy,
  const T &fieldz, const double (&bcc)[4], const Omega_h::LO ielem, Omega_h::Vector<3> &fv)
{
    Omega_h::Real fx = interpolateTet(fieldx,  bcc,  ielem);
    Omega_h::Real fy = interpolateTet(fieldy,  bcc,  ielem);
    Omega_h::Real fz = interpolateTet(fieldz,  bcc,  ielem);
    fv = {fx, fy, fz};
}


OMEGA_H_INLINE void findTetCoords(const Omega_h::LOs &mesh2verts,
const Omega_h::Reals &coords, const Omega_h::LO ielem, Omega_h::Matrix<DIM, 4> &mat)
{
  const auto tetv2v = Omega_h::gather_verts<4>(mesh2verts, ielem);
  mat = Omega_h::gather_vectors<4, 3>(coords, tetv2v);
}

OMEGA_H_INLINE void findBCCoordsInTet(const Omega_h::Reals &coords, const Omega_h::LOs &mesh2verts,
   const Omega_h::Vector<3> &xyz, const Omega_h::LO ielem, Omega_h::Write<Omega_h::Real> &bcc)
{
  Omega_h::Matrix<3, 4> mat;
  findTetCoords(mesh2verts, coords, ielem, mat);
  const bool res = find_barycentric_tet(mat, xyz, bcc);
  OMEGA_H_CHECK(res==0);
  OMEGA_H_CHECK(all_positive(bcc.data(), 4)==1);
}

// BC coords are not in order of its corresp. vertexes. Bccoord of triangle (iedge, xpoint)
// corresp. to vertex obtained from simplex_opposite_template(FDIM, 1, iedge) ?
OMEGA_H_INLINE bool find_barycentric_tri_simple(const o::Few<o::Vector<DIM>, 3> &abc,
     const o::Vector<3> &xpoint, o::Write<o::Real> &bc) {

  o::Vector<DIM> a = abc[0];
  o::Vector<DIM> b = abc[1];
  o::Vector<DIM> c = abc[2];
  o::Vector<DIM> cross = 1/2.0 * o::cross(b-a, c-a); //NOTE order
  o::Vector<DIM> norm = o::normalize(cross);
  o::Real area = osh_dot(norm, cross);

  if(std::abs(area) < 1e-6)  //TODO
    return 0;
  o::Real fac = 1/(area*2.0);
  bc[0] = fac * osh_dot(norm, o::cross(b-a, xpoint-a));
  bc[1] = fac * osh_dot(norm, o::cross(c-b, xpoint-b));
  bc[2] = fac * osh_dot(norm, o::cross(xpoint-a, c-a));

  return 1;
}

OMEGA_H_INLINE bool line_triangle_intx_simple(const o::Few<o::Vector<DIM>, 3> &abc,
    const o::Vector<DIM> &origin, const o::Vector<DIM> &dest,
    o::Vector<DIM> &xpoint, o::LO &edge, bool reverse=false ) {

  edge = -1;
  xpoint = {0, 0, 0};

  if(verbose > 1) {
    print_osh_vector(origin, "origin", false);
    print_osh_vector(dest, "dest");
  }
    
  //Boundary exclusion. Don't set it globally and change randomnly.
  const o::Real bound_intol = 0;//SURFACE_EXCLUDE; //TODO optimum value ?

  bool found = false;
  const o::Vector<DIM> line = dest - origin;
  const o::Vector<DIM> edge0 = abc[1] - abc[0];
  const o::Vector<DIM> edge1 = abc[2] - abc[0];
  o::Vector<DIM> normv = o::cross(edge0, edge1);

  if(reverse) {
    normv = -1*normv;
    if(verbose > 0)
      std::cout << "Surface normal reversed \n";

  }
  const o::Vector<DIM> snorm_unit = o::normalize(normv);
  const o::Real dist2plane = osh_dot(abc[0] - origin, snorm_unit);
  const o::Real proj_lined =  osh_dot(line, snorm_unit);
  const o::Vector<DIM> surf2dest = dest - abc[0];

  if(std::abs(proj_lined) >0) {
    const o::Real par_t = dist2plane/proj_lined;
    if(verbose > 2)
      std::cout << " abs(proj_lined)>0;  par_t= " << par_t << " dist2plane= "
             <<  dist2plane << "; proj_lined= " << proj_lined << ";\n";
    if (par_t > bound_intol && par_t <= 1.0) {//TODO test tol value
      xpoint = origin + par_t * line;
      o::Write<o::Real> bcc{3,0};
      bool res = find_barycentric_tri_simple(abc, xpoint, bcc);
      if(verbose > 2)
        print_array(bcc.data(), 3, "BCC");
      if(res) {
        if(bcc[0] < 0 || bcc[2] < 0 || bcc[0]+bcc[2] > 1.0) { //TODO all zeros ?
          edge = min_index(bcc.data(), 3, EPSILON); //TODO test tolerance
        }
        else {
          const o::Real proj = osh_dot(snorm_unit, surf2dest);
          if(proj >0) found = true;
          else if (proj<0) {
            if(verbose > 1)
              std::cout << "Particle Entering domain\n";
          }
          else if(almost_equal(proj,0.0)) {//TODO use tol
            if(verbose > 1)
              std::cout << "Particle path on surface\n";
          }
        }
      }
      if(verbose > 2)
        print_array(bcc.data(), 3, "BCCtri");
    }
    else if(par_t >1.0) {
      if(verbose > 0)
        std::cout << "Error** Line origin and destination are on the same side of face \n";
    }
    else if(par_t < bound_intol) {// dist2plane ~0. Line contained in plane, no intersection?
      if(verbose > 1)
        std::cout << "No/Self-intersection of ptcl origin with plane at origin. t= " << par_t << " "
                << dist2plane << " " << proj_lined << "\n";
    }
  }
  else if(verbose > 1) {
      std::cout << "Line and plane are parallel \n";
  }
  return found;
}

/*
Adjacent element IDs are stored for further searching if the ptcl not done. 
TODO Search should exclude particle if on surface of an element by SURFACE_EXCLUDE. 
When an intersection on bdry is found, that face id is to be stored to use in surface model.
NOTE: reset it before this function call, but not during the while loop within this.
*/


inline bool search_mesh(const Omega_h::Write<Omega_h::LO> pids, Omega_h::LO nelems, 
    const Omega_h::Write<Omega_h::Real> &x0,
 const Omega_h::Write<Omega_h::Real> &y0, const Omega_h::Write<Omega_h::Real> &z0, 
 const Omega_h::Write<Omega_h::Real> &x, const Omega_h::Write<Omega_h::Real> &y, 
 const Omega_h::Write<Omega_h::Real> &z, const Omega_h::Adj &dual, const Omega_h::Adj &down_r2f,
 const Omega_h::Read<Omega_h::I8> &side_is_exposed, const Omega_h::LOs &mesh2verts, 
 const Omega_h::Reals &coords, const Omega_h::LOs &face_verts, Omega_h::Write<Omega_h::LO> &part_flags,
 Omega_h::Write<Omega_h::LO> &elem_ids, Omega_h::Write<Omega_h::LO> &coll_adj_face_ids, 
 Omega_h::Write<Omega_h::Real> &bccs, Omega_h::Write<Omega_h::Real> &xpoints, Omega_h::LO &loops, 
 Omega_h::LO limit=0)
{
  const auto down_r2fs = &down_r2f.ab2b;
  const auto dual_faces = &dual.ab2b;
  const auto dual_elems = &dual.a2ab;

  const int debug = 0;

  const int totNumPtcls = elem_ids.size();
  Omega_h::Write<Omega_h::LO> elem_ids_next(totNumPtcls,-1);

  //particle search: adjacency + boundary crossing
  auto search_ptcl = OMEGA_H_LAMBDA( Omega_h::LO ielem)
  {
    // NOTE ielem is taken as sequential from 0 ... is it elementID ? TODO verify it
    const auto tetv2v = Omega_h::gather_verts<4>(mesh2verts, ielem);
    const auto M = Omega_h::gather_vectors<4, 3>(coords, tetv2v);

    // parallel_for loop for groups of remaining particles in this element
    //......

    // Each group of particles inside the parallel_for.
    // TODO Change ntpcl, ip start and limit. Update global(?) indices inside.
    for(Omega_h::LO ip = 0; ip < totNumPtcls; ++ip) //HACK - each element checks all particles
    {
      //skip if the particle is not in this element or has been found
      if(elem_ids[ip] != ielem || part_flags[ip] <= 0) continue;

      if(debug)
        std::cerr << "Elem " << ielem << " ptcl:" << ip << "\n";
        
      const Omega_h::Vector<3> orig{x0[ip], y0[ip], z0[ip]};
      const Omega_h::Vector<3> dest{x[ip], y[ip], z[ip]};
      
      Omega_h::Write<Omega_h::Real> bcc(4, -1.0);

      //TESTING. Check particle origin containment in current element
      find_barycentric_tet(M, orig, bcc);
      if(debug>3 && !(all_positive(bcc.data(), 4)))
          std::cerr << "ORIGIN ********NOT in elemet_id " << ielem << " \n";
      find_barycentric_tet(M, dest, bcc);

      //check if the destination is in this element
      if(all_positive(bcc.data(), 4, 0)) //SURFACE_EXCLUDE)) TODO
      {
        // TODO interpolate Fields to ptcl position, and store them, for push
        // interpolateFields(bcc, ptcls);
        elem_ids_next[ip] = elem_ids[ip];
        part_flags.data()[ip] = -1;
        if(debug) 
        {
            std::cerr << "********found in " << ielem << " \n";
            print_matrix(M);
        }
        continue;
      }
       //get element ID
      //TODO get map from omega methods. //2,3 nodes of faces. 0,2,1; 0,1,3; 1,2,3; 2,0,3
      Omega_h::LOs fmap{2,1,1,3,2,3,0,3}; 
      auto dface_ind = (*dual_elems)[ielem];
      const auto beg_face = ielem *4;
      const auto end_face = beg_face +4;
      Omega_h::LO f_index = 0;
      bool inverse;

      for(auto iface = beg_face; iface < end_face; ++iface) //not 0..3
      {
        const auto face_id = (*down_r2fs)[iface];
        if(debug >1)  
          std::cout << " \nFace: " << face_id << " dface_ind " <<  dface_ind << "\n";

        Omega_h::Vector<3> xpoint{0,0,0};
        auto fv2v = Omega_h::gather_verts<3>(face_verts, face_id); //Few<LO, 3>

        const auto face = Omega_h::gather_vectors<3, 3>(coords, fv2v);
        Omega_h::LO matInd1 = fmap[f_index*2], matInd2 = fmap[f_index*2+1];

        if(debug >3) {
          std::cout << "Face_local_index "<< fv2v.data()[0] << " " << fv2v.data()[1] << " " << fv2v.data()[2] << "\n";
          std::cout << "Mat index "<< tetv2v[matInd1] << " " << tetv2v[matInd2] << " " <<  matInd1 << " " << matInd2 << " \n";
          std::cout << "Mat dat ind " <<  tetv2v.data()[0] << " " << tetv2v.data()[1] << " "
                   << tetv2v.data()[2] << " " << tetv2v.data()[3] << "\n";
        }


        if(fv2v.data()[1] == tetv2v[matInd1] && fv2v.data()[2] == tetv2v[matInd2])
          inverse = false;
        else // if(fv2v.data()[1] == tetv2v[matInd2] && fv2v.data()[2] == tetv2v[matInd1])
        {
          inverse = true;
        }

        //TODO not useful
        auto fcoords = Omega_h::gather_vectors<3, 3>(coords, fv2v);
        auto base = Omega_h::simplex_basis<3, 2>(fcoords); //edgres = Matrix<2,3>
        auto snormal = Omega_h::normalize(Omega_h::cross(base[0], base[1]));

        Omega_h::LO dummy = -1;
        bool detected = line_triangle_intx_simple(face, orig, dest, xpoint, dummy, inverse);
        if(debug && detected)
            std::cout << " Detected: For el=" << ielem << "\n";

        if(detected && side_is_exposed[face_id])
        {
           part_flags.data()[ip] = -1;
           for(Omega_h::LO i=0; i<3; ++i)xpoints[ip*3+i] = xpoint.data()[i];
           //store current face_id and element_ids

           if(debug)
             print_osh_vector(xpoint, "COLLISION POINT");

           elem_ids_next[ip] = -1;
           break;
         }
         else if(detected && !side_is_exposed[face_id])
         {
          //OMEGA_H_CHECK(side2side_elems[side + 1] - side2side_elems[side] == 2);
           auto adj_elem  = (*dual_faces)[dface_ind];
           if(debug)
             std::cout << "Deletected For el=" << ielem << " ;face_id=" << (*down_r2fs)[iface]
                     << " ;ADJ elem= " << adj_elem << "\n";

           elem_ids_next[ip] = adj_elem;
           break;
         }

         if(!side_is_exposed[face_id])//TODO for DEBUG
         {
           if(debug)
             std::cout << "adj_element_across_this_face " << (*dual_faces)[dface_ind] << "\n";
           const Omega_h::LO min_ind = min_index(bcc.data(), 4);
           if(f_index == min_ind)
           {
             if(debug)
               std::cout << "Min_bcc el|face_id=" << ielem << "," << (*down_r2fs)[iface]
                     << " :unused adj_elem= " << (*dual_faces)[dface_ind] << "\n";
            if(!detected)
            {
              elem_ids_next[ip] = (*dual_faces)[dface_ind];
              if(debug)
                std::cout << "...  adj_elem=" << elem_ids[ip]  <<  "\n";
            }
           }

         }

         if( !side_is_exposed[face_id])
           ++dface_ind;

         ++f_index;
      } //iface 
 
    }//ip
  };

  bool found = false;
  loops = 0;
  while(!found)
  {
    if(debug) fprintf(stderr, "------------ %d ------------\n", loops);
    //TODO check if particle is on boundary and remove from list if so.

    // Searching all elements. TODO exclude those done ?
    Omega_h::parallel_for(nelems,  search_ptcl, "search_ptcl");
    found = true;
    auto cp_elm_ids = OMEGA_H_LAMBDA( Omega_h::LO i) {
      elem_ids[i] = elem_ids_next[i];
    };
    Omega_h::parallel_for(elem_ids.size(), cp_elm_ids, "copy_elem_ids");

    // TODO synchronize

    //TODO this could be a sequential bottle-neck
    for(int i=0; i<totNumPtcls; ++i){ if(part_flags[i] > 0) {found = false; break;} }
    //Copy particle data from previous to next (adjacent) element
    ++loops;

    if(limit && loops>limit) break;
  }

  std::cerr << "search iterations " << loops << "\n";

  return found;
} //search_mesh


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

//TODO test this function
OMEGA_H_INLINE o::LO find_closest_point_on_triangle_with_normal(
  const o::Few< o::Vector<3>, 3> &abc,
  const o::Vector<3> &p, o::Vector<3> &q, o::LO verbose = 0){

  o::LO region = -1;
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
  Vector n; 
  Vector temp;
  osh_cross(b - a, c - a, n);
  osh_cross(a - p, b - p, temp);
  float vc = osh_dot(n, temp);
  // If P outside AB and within feature region of AB,
  // return projection of P onto AB
  if (vc <= 0.0 && snom >= 0.0 && sdenom >= 0.0){
    q = a + snom / (snom + sdenom) * ab;
    return EDGEAB;
  }
  // P is outside (or on) BC if the triple scalar product [N PB PC] <= 0
  osh_cross(b - p, c - p, temp);
  float va = osh_dot(n, temp);
  // If P outside BC and within feature region of BC,
  // return projection of P onto BC
  if (va <= 0.0 && unom >= 0.0 && udenom >= 0.0){
    q = b + unom / (unom + udenom) * bc;
    return EDGEBC;
  }
  // P is outside (or on) CA if the triple scalar product [N PC PA] <= 0
  osh_cross(c - p, a - p, temp);
  float vb = osh_dot(n, temp);
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
OMEGA_H_INLINE o::LO get_face_type_ids_of_elem(const o::LO elem, 
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
OMEGA_H_INLINE void get_face_data_by_id(const Omega_h::LOs &face_verts, 
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


OMEGA_H_INLINE bool check_if_face_within_dist_to_tet(const o::Matrix<DIM, 4> &tet, 
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


inline void test_find_closest_point_on_triangle(){
  constexpr int nTris = 1;
  o::Few<o::Few< o::Vector<3>, 3>, nTris> abcs{ {{1,0,0},{2,0,0},{1.5,1,0}} }; 

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


inline void test_find_distance_to_bdry(){


}


} //namespace

#endif //define

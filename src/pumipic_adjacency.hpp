#ifndef PUMIPIC_ADJACENCY_HPP
#define PUMIPIC_ADJACENCY_HPP

#include <iostream>

#include "Omega_h_for.hpp"
#include "Omega_h_adj.hpp"
#include "Omega_h_element.hpp"

#include "pumipic_utils.hpp"
#include "pumipic_constants.hpp"


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

#if DEBUG >2
    std::cout << "face " << iface << ": \n"; 
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
//TODO Warning: Check opposite_template use in this before using
OMEGA_H_INLINE bool find_barycentric_tet( const Omega_h::Matrix<DIM, 4> &Mat,
     const Omega_h::Vector<DIM> &pos, Omega_h::Write<Omega_h::Real> &bcc)
{
  for(Omega_h::LO i=0; i<3; ++i) bcc[i] = -1;

  Omega_h::Real vals[4];
  Omega_h::Few<Omega_h::Vector<DIM>, 3> abc;
  for(Omega_h::LO iface=0; iface<4; ++iface) // TODO last not needed
  {
    get_face_coords(Mat, iface, abc);
    auto vab = abc[1] - abc[0]; //b - a;
    auto vac = abc[2] - abc[0]; //c - a;
    auto vap = pos - abc[0]; // p - a;
    vals[iface] = osh_dot(vap, Omega_h::cross(vac, vab)); //ac, ab NOTE

#if DEBUG >2
    std::cout << "vol: " << vals[iface] << " for points_of_this_TET:\n" ;
    print_array(abc[0].data(),3);
    print_array(abc[1].data(),3);
    print_array(abc[2].data(),3);
    print_array(pos.data(),3, "point");
    std::cout << "\n";
#endif // DEBUG
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
#if DEBUG >0  
    std::cout << vol6 << "too low \n";
#endif 
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
     const Omega_h::Vector<3> &xpoint, Omega_h::Write<Omega_h::Real> &bc)
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
    Omega_h::Vector<DIM> &xpoint, Omega_h::LO &edge)
{
  edge = -1;
  xpoint = {0, 0, 0};

  //Boundary exclusion. Don't set it globally and change randomnly.
  const Omega_h::Real bound_intol = SURFACE_EXCLUDE; //TODO optimum value ?

  bool found = false;
  const Omega_h::Vector<DIM> line = dest - origin;
  const Omega_h::Vector<DIM> edge0 = abc[1] - abc[0];
  const Omega_h::Vector<DIM> edge1 = abc[2] - abc[0];
  const Omega_h::Vector<DIM> normv = Omega_h::cross(edge0, edge1);
  const Omega_h::Vector<DIM> snorm_unit = Omega_h::normalize(normv);
  const Omega_h::Real dist2plane = osh_dot(abc[0] - origin, snorm_unit);
  const Omega_h::Real proj_lined =  osh_dot(line, snorm_unit);
  const Omega_h::Vector<DIM> surf2dest = dest - abc[0];

  if(std::abs(proj_lined >0))
  {
    const Omega_h::Real par_t = dist2plane/proj_lined;

    if (par_t > bound_intol && par_t <= 1.0) //TODO test tol value
    {
      xpoint = origin + par_t * line;
      Omega_h::Write<Omega_h::Real> bcc{3,0};
      bool res = find_barycentric_tri_simple(abc, xpoint, bcc);

      if(res)
      {
        if(bcc[0] < 0 || bcc[2] < 0 || bcc[0]+bcc[2] > 1.0) //TODO all zeros ?
        {
          edge = min_index(bcc.data(), 3, EPSILON); //TODO test tolerance
        }
        else
        {
          const Omega_h::Real proj = osh_dot(snorm_unit, surf2dest);
          if(proj >0) found = true;
#ifdef DEBUG
          else if(proj<0)
          {
            std::cout << "Particle Entering domain\n";
          }
          else if(almost_equal(proj,0.0)) //TODO use tol
          {
            std::cout << "Particle path on surface\n";
          }
#endif // DEBUG
        }
      }
#if DEBUG >1
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
// TODO Change if needed. Adjacency search excludes particle if on surface of an element by SURFACE_EXCLUDE. Because,
// a point on boundary surface should be intersection, which is searched for only if adjacency search does not find it.
//TODO Avoid passing object Adj ?

OMEGA_H_INLINE bool search_mesh(Omega_h::LO nptcl, Omega_h::LO nelems, const Omega_h::Write<Omega_h::Real> &x0,
 const Omega_h::Write<Omega_h::Real> &y0, const Omega_h::Write<Omega_h::Real> &z0, 
 const Omega_h::Write<Omega_h::Real> &x, const Omega_h::Write<Omega_h::Real> &y, 
 const Omega_h::Write<Omega_h::Real> &z, const Omega_h::Adj &dual, const Omega_h::Adj &down_r2f,
 const Omega_h::Adj &down_f2e, const Omega_h::Adj &up_e2f, const Omega_h::Adj &up_f2r,
 const Omega_h::Read<Omega_h::I8> &side_is_exposed, const Omega_h::LOs &mesh2verts, 
 const Omega_h::Reals &coords, const Omega_h::LOs &face_verts, Omega_h::Write<Omega_h::LO> &part_flags,
 Omega_h::Write<Omega_h::LO> &elem_ids, Omega_h::Write<Omega_h::LO> &coll_adj_face_ids, 
 Omega_h::Write<Omega_h::Real> &bccs, Omega_h::Write<Omega_h::Real> &xpoints, Omega_h::LO &loops, 
 Omega_h::LO limit=0)
{
  const auto up_e2f_edges = &up_e2f.a2ab;
  const auto up_e2f_faces = &up_e2f.ab2b;
  const auto down_f2es = &down_f2e.ab2b;
  const auto down_r2fs = &down_r2f.ab2b;
  const auto dual_faces = &dual.ab2b;
  const auto dual_elems = &dual.a2ab;
  const auto up_f2r_faces = &up_f2r.a2ab;
  const auto up_f2r_reg = &up_f2r.ab2b;

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

    assert(nptcl==1);
    // Each group of particles inside the parallel_for.
    // TODO Change ntpcl, ip start and limit. Update global(?) indices inside.
    for(Omega_h::LO ip = 0; ip < totNumPtcls; ++ip) //HACK - each element checks all particles
    {
      //skip if the particle is not in this element or has been found
      if(elem_ids[ip] != ielem || part_flags[ip] <= 0) continue;

      if(debug)
        std::cerr << "Elem " << ielem << " ptcl:" << ip << "\n";
      bool continue_coll = (coll_adj_face_ids[ip] !=-1) ? true:false;
      Omega_h::LO coll_face_id = -1;

      const Omega_h::Vector<3> orig{x0[ip], y0[ip], z0[ip]};
      const Omega_h::Vector<3> dest{x[ip], y[ip], z[ip]};
      bool do_collision = false;
      if(!continue_coll)
      {
        Omega_h::Write<Omega_h::Real> bcc(4, -1.0);

        //TESTING. Check particle origin containment in current element
        find_barycentric_tet(M, orig, bcc);
        if(debug>3 && !(all_positive(bcc.data(), 4)))
          std::cerr << "ORIGIN ********NOT in elemet_id " << ielem << " \n";
        find_barycentric_tet(M, dest, bcc);

        //check if the destination is in this element
        if(all_positive(bcc.data(), 4, 0)) //SURFACE_EXCLUDE)) TODO
        {
          part_flags.data()[ip] = 0;
          elem_ids_next[ip] = ielem;
          for(Omega_h::LO i=0; i<4; ++i) bccs[4*ip+i] = bcc[i];
          OMEGA_H_CHECK(almost_equal(bcc[0] + bcc[1] + bcc[2] +bcc[3], 1.0)); //?
          if(debug) {
            std::cerr << "********found in " << ielem << " \n";
            print_matrix(M);
          }
        }
        //destination is not in current element
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

            if(!side_is_exposed[face_id]) //interior face
            {
               auto adj_elem  = (*dual_faces)[dface_ind];
               if(debug>3)
                 std::cerr << "el " << ielem << " adj " << adj_elem << " face " << f_index << "\n";
               if(f_index == min_entry)
               {
                 if(debug>3)
                   std::cerr << "=====> For el|face_id=" << ielem << "," << (*down_r2fs)[iface]  << " :ADJ elem= " << adj_elem << "\n";
                 elem_ids_next[ip] = adj_elem;
                 break;
               }
               ++dface_ind;
            }
            else if(f_index == min_entry && bcc[f_index] < 0) // if exposed face
            {
              coll_face_id = face_id;
              do_collision = true;
              if(debug>1)
                std::cerr << "To_search_coll for face " << coll_face_id << " faceind:" << f_index << "\n";
              break;
            }
            if(debug>3)
              std::cerr << "faceind " << f_index << "\n";
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
        Omega_h::LO face_id = continue_coll? coll_adj_face_ids[ip]:coll_face_id;
        auto fv2v = Omega_h::gather_verts<3>(face_verts, face_id); //Few<LO, 3>
        const auto face = Omega_h::gather_vectors<3, 3>(coords, fv2v);

        if(debug>1)
          std::cerr << "********* \n Call Wall collision for el,face_id " << ielem << "," << face_id << "\n";
        bool detected = line_triangle_intx_simple(face, orig, dest, xpoint, cross_edge);

        if(detected)
        {
          //elem_ids[ip] = -1;
          //coll_adj_face_ids[ip] = -1;
          part_flags.data()[ip] = -1;
          for(Omega_h::LO i=0; i<3; ++i)xpoints[ip*3+i] = xpoint.data()[i];
          //store current face_id and element_ids
          if(debug)
            print_osh_vector(xpoint, "COLLISION POINT");
          break;
        }
        if(debug>1)
          std::cerr << "edge " << cross_edge << "\n";
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
            elem_ids_next[ip] = adj_elem_id;
            coll_adj_face_ids[ip] = adj_face_id;
            //part_flags.data()[ip] = 2; //collision
            next_elem = true;
          }
        }

        if(cross_edge <0 || !next_elem)
        {
          elem_ids_next[ip] = ielem; //current element
          if(debug>1)
            std::cerr << "Collision tracking lost.\n";
          //Error stop
          part_flags.data()[ip] = -2;
        }

      } //do_collision
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


} //namespace
#ifdef DEBUG
#undef DEBUG
#endif // DEBUG
#endif //define


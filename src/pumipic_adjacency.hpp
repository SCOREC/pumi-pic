#ifndef PUMIPIC_ADJACENCY_HPP
#define PUMIPIC_ADJACENCY_HPP

#include <iostream>

#include "Omega_h_for.hpp"
#include "Omega_h_adj.hpp"
#include "Omega_h_element.hpp"

#include "pumipic_utils.hpp"
#include "pumipic_constants.hpp"

#include <SellCSigma.h>
#include <SCS_Macros.h>

namespace o = Omega_h;

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

OMEGA_H_INLINE bool line_triangle_intx_simple(const Omega_h::Few<Omega_h::Vector<DIM>, 3> &abc,
    const Omega_h::Vector<DIM> &origin, const Omega_h::Vector<DIM> &dest,
    Omega_h::Vector<DIM> &xpoint, Omega_h::LO &edge, bool reverse=false )
{
  bool debug=0;
  edge = -1;
  xpoint = {0, 0, 0};

  if(debug) {
    print_osh_vector(origin, "origin", false);
    print_osh_vector(dest, "dest");
  }
    
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
      std::cout << "Surface normal reversed \n";

  }
  const Omega_h::Vector<DIM> snorm_unit = Omega_h::normalize(normv);
  const Omega_h::Real dist2plane = osh_dot(abc[0] - origin, snorm_unit);
  const Omega_h::Real proj_lined =  osh_dot(line, snorm_unit);
  const Omega_h::Vector<DIM> surf2dest = dest - abc[0];

  if(std::abs(proj_lined) >0)
  {
    const Omega_h::Real par_t = dist2plane/proj_lined;
    if(debug)
      std::cout << " abs(proj_lined)>0;  par_t= " << par_t << " dist2plane= "
             <<  dist2plane << "; proj_lined= " << proj_lined << ";\n";
    if (par_t > bound_intol && par_t <= 1.0) //TODO test tol value
    {
      xpoint = origin + par_t * line;
      Omega_h::Write<Omega_h::Real> bcc{3,0};
      bool res = find_barycentric_tri_simple(abc, xpoint, bcc);
      if(debug)
        print_array(bcc.data(), 3, "BCC");
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
          else if(proj<0)
          {
            if(debug)
              std::cout << "Particle Entering domain\n";
          }
          else if(almost_equal(proj,0.0)) //TODO use tol
          { 
            if(debug)
              std::cout << "Particle path on surface\n";
          }
        }
      }
      if(debug)
        print_array(bcc.data(), 3, "BCCtri");
    }
    else if(par_t >1.0)
    {
      if(debug)
        std::cout << "Error** Line origin and destination are on the same side of face \n";
    }
    else if(par_t < bound_intol) // dist2plane ~0. Line contained in plane, no intersection?
    {
      if(debug)
        std::cout << "No/Self-intersection of ptcl origin with plane at origin. t= " << par_t << " "
                << dist2plane << " " << proj_lined << "\n";
    }
  }
  else
  {
    std::cout << "Line and plane are parallel \n";
  }
  return found;
}

OMEGA_H_INLINE bool search_mesh(o::Mesh& mesh, SellCSigma<Particle>* scs) {
  return true;
}

OMEGA_H_INLINE bool search_mesh(const Omega_h::Write<Omega_h::LO> pids, Omega_h::LO nelems, const Omega_h::Write<Omega_h::Real> &x0,
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
      //skip inactive particles
      if(pids[ip] == -1) {
        continue;
      }

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

    Omega_h::LOs part_flags_r(part_flags);
    auto minFlag = Omega_h::get_min(part_flags_r);
    auto maxFlag = Omega_h::get_max(part_flags_r);
    fprintf(stderr, "%d 0.2 minFlag maxFlag %d %d\n", loops, minFlag, maxFlag);
    if(maxFlag > 0)
      found = false;
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


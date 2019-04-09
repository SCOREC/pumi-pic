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

OMEGA_H_INLINE o::Vector<3> makeVector(int pid, kkFp3View xyz) {
  return o::Vector<3>{xyz(pid,0), xyz(pid,1), xyz(pid,2)};
}

OMEGA_H_INLINE o::LO getfmap(int i) {
  assert(i>=0 && i<8);
  o::LOs fmap{2,1,1,3,2,3,0,3};
  return fmap[i];
}

//HACK to avoid having an unguarded comma in the SCS PARALLEL macro
OMEGA_H_INLINE o::Matrix<3, 3> gatherVectors3x3(o::Reals const& a, o::Few<o::LO, 3> v) {
  return o::gather_vectors<3, 3>(a, v);
}
OMEGA_H_INLINE o::Matrix<3, 4> gatherVectors4x3(o::Reals const& a, o::Few<o::LO, 4> v) {
  return o::gather_vectors<4, 3>(a, v);
}

//How to avoid redefining the MemberType? each application will define it
//differently. Templating search_mesh with
//template < typename ParticleType >
//results in an error on getSCS<> as an unresolved function.
//typedef particle_structs::MemberTypes<Vector3d, Vector3d, int> ParticleType;

template < class ParticleType>
bool search_mesh(o::Mesh& mesh, ps::SellCSigma< ParticleType >* scs,
    o::Write<o::LO>& elem_ids, int looplimit=0) {
  const auto dual = mesh.ask_dual();
  const auto down_r2f = mesh.ask_down(3, 2);
  const auto side_is_exposed = mark_exposed_sides(&mesh);
  const auto mesh2verts = mesh.ask_elem_verts();
  const auto coords = mesh.coords();
  const auto face_verts =  mesh.ask_verts_of(2);

  scs->transferToDevice();  //TODO user tuples should be allocated on device by default
  const auto scsCapacity = scs->offsets[scs->num_slices];
  kkFp3View x_scs_d("x_scs_d", scsCapacity);
  hostToDeviceFp(x_scs_d, scs->template getSCS<0>() );
  kkFp3View xtgt_scs_d("xtgt_scs_d", scsCapacity);
  hostToDeviceFp(xtgt_scs_d, scs->template getSCS<1>() );

  kkLidView pid_d("pid_d", scsCapacity);
  hostToDeviceLid(pid_d, scs->template getSCS<2>() );

  // ptcl_flags[i] < 0 : particle i has hit a boundary or reached its destination
  o::Write<o::LO> ptcl_flags(scsCapacity, 1, "ptcl_flags");
  // particle intersection points
  o::Write<o::Real> xpoints(3*scsCapacity, -1.0);
  // store the next parent for each particle
  o::Write<o::LO> elem_ids_next(scsCapacity,-1);
  PS_PARALLEL_FOR_ELEMENTS(scs, thread, e, {
    PS_PARALLEL_FOR_PARTICLES(scs, thread, pid, {
      if(particle_mask(pid)) {
        elem_ids[pid] = e;
      } else {
        elem_ids[pid] = -1;
      }
    });
  });

  const int debug = 0;

  bool found = false;
  int loops = 0;
  while(!found) {
    if(debug) {
      fprintf(stderr, "------------ %d ------------\n", loops);
    }
    PS_PARALLEL_FOR_ELEMENTS(scs, thread, e, {
      (void)e;
      const auto tetv2v = o::gather_verts<4>(mesh2verts, e);
      const auto M = gatherVectors4x3(coords, tetv2v);
      PS_PARALLEL_FOR_PARTICLES(scs, thread, pid, {
        //inactive particle that is still moving to its target position
        if( particle_mask(pid) && ptcl_flags[pid] > 0 ) {
          auto elmId = e;
          if(debug)
            std::cerr << "Elem " << elmId << " ptcl:" << pid << "\n";
          const o::Vector<3> orig = makeVector(pid, x_scs_d);
          const o::Vector<3> dest = makeVector(pid, xtgt_scs_d);
          o::Write<o::Real> bcc(4, -1.0);
          //Check particle origin containment in current element
          find_barycentric_tet(M, orig, bcc);
          find_barycentric_tet(M, dest, bcc);
          //check if the destination is in this element
          if(all_positive(bcc, 4, 0)) {
            elem_ids_next[pid] = elem_ids[pid];
            ptcl_flags[pid] = -1;
          } else {
            //get element ID
            //TODO get map from omega methods. //2,3 nodes of faces. 0,2,1; 0,1,3; 1,2,3; 2,0,3
            auto dface_ind = dual[elmId];
            const auto beg_face = e *4;
            const auto end_face = beg_face +4;
            o::LO f_index = 0;
            bool inverse;

            for(auto iface = beg_face; iface < end_face; ++iface) {
              const auto face_id = down_r2f[iface];

              o::Vector<3> xpoint(0);
              auto fv2v = o::gather_verts<3>(face_verts, face_id);

              const auto face = gatherVectors3x3(coords, fv2v);
              o::LO matInd1 = getfmap(f_index*2);
              o::LO matInd2 = getfmap(f_index*2+1);

              if(fv2v[1] == tetv2v[matInd1] && fv2v[2] == tetv2v[matInd2])
                inverse = false;
              else
                inverse = true;

              o::LO dummy = -1;
              bool detected = line_triangle_intx_simple(face, orig, dest, xpoint, dummy, inverse);

              if(detected && side_is_exposed[face_id]) {
                 part_flags[pid] = -1;
                 for(o::LO i=0; i<3; ++i)
                   xpoints[pid*3+i] = xpoint[i];
                 elem_ids_next[pid] = -1;
                 break;
              } else if(detected && !side_is_exposed[face_id]) {
                 auto adj_elem  = dual_faces[dface_ind];
                 elem_ids_next[pid] = adj_elem;
                 break;
              }

              if(!side_is_exposed[face_id]) {
                const o::LO min_ind = min_index(bcc, 4);
                if(f_index == min_ind && !detected)
                  elem_ids_next[pid] = dual_faces[dface_ind];
              }

              if( !side_is_exposed[face_id])
                ++dface_ind;

              ++f_index;
            } //iface
          }
        }
      });
    });

    found = true;
    auto cp_elm_ids = OMEGA_H_LAMBDA( o::LO i) {
      elem_ids[i] = elem_ids_next[i];
    };
    o::parallel_for(elem_ids.size(), cp_elm_ids, "copy_elem_ids");

    o::LOs ptcl_flags_r(ptcl_flags);
    auto minFlag = o::get_min(ptcl_flags_r);
    auto maxFlag = o::get_max(ptcl_flags_r);
    fprintf(stderr, "%d 0.2 minFlag maxFlag %d %d\n", loops, minFlag, maxFlag);
    if(maxFlag > 0)
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

} //namespace
#ifdef DEBUG
#undef DEBUG
#endif // DEBUG
#endif //define


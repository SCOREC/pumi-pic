#include <iostream>
#include <cmath>
#include <utility>

#include "Omega_h_for.hpp"
#include "Omega_h_file.hpp"  //gmsh
#include "Omega_h_tag.hpp"
#include "Omega_h_adj.hpp"
//#include "Omega_h_array.hpp"
#include "Omega_h_array_ops.hpp"
#include "Omega_h_element.hpp"
#include "Omega_h_scalar.hpp" //divide
#include "Omega_h_mark.hpp"
#include "Omega_h_fail.hpp" //assert

#include "Omega_h_mesh.hpp"
#include "Omega_h_shape.hpp"
#include "Omega_h_build.hpp"
#include "Omega_h_compare.hpp"
#include "Omega_h_reduce.hpp"

#include "gitrm_utils.hpp"
#include "gitrm_adjacency.hpp"
#include "unit_tests.hpp"

namespace o = Omega_h;
namespace g = GITRm;

#define DEBUG 1
#define DO_TEST 0
int main(int argc, char** argv) {

  auto lib = Omega_h::Library(&argc, &argv);
  const auto world = lib.world();
  auto mesh = Omega_h::gmsh::read(argv[1], world);
  const auto dim = mesh.dim();

#if DO_TEST==1
  test_unit(&lib);
  print_mesh_stat(mesh);
  test_barycentric1();
  test_barycentric2();
  test_barycentric_tri();
  test_line_tri_intx();
#endif

//Kokkos::View<int*> a("a" , 100);
//Kokkos::View<int*, Kokkos::MemoryTraits<Atomic> > a_atomic = a;
//a_atomic(1) += 1;

  //adjacent element
  const auto dual = mesh.ask_dual();
  auto dual_faces = dual.ab2b;
  auto dual_elems = dual.a2ab;
  auto up = mesh.ask_up(dim - 1, dim);
  const auto down= mesh.ask_down(dim, dim - 1);
  const auto down20 = mesh.ask_down(2,0);
  const auto downs= down.ab2b;
  auto side_is_exposed = mark_exposed_sides(&mesh);
  Omega_h::Int nsides_per_elem = dim + 1;
  Omega_h::Int nelems = mesh.nelems();

  //get coordinate values
  const auto mesh2verts = mesh.ask_elem_verts();
  const auto coords = mesh.coords();
  const auto face_verts =  mesh.ask_verts_of(2);//LOs

  const Omega_h::Few< Omega_h::Vector<3>, 2> dest{{0.10,0.3,0.1}, {12,0.2,0.2}};
  const Omega_h::Few< Omega_h::Vector<3>, 2> orig{{9.8,0.3,0.7}, {12,0.2,0.2}};

  Omega_h::LO nptcl = 2;

  Omega_h::Write<Omega_h::LO> part_flags(nptcl, 1);
  Omega_h::Write<Omega_h::LO> elemIds(nptcl);

nptcl=1; //error for 2 !

  elemIds[0] = 24; // begin with
  elemIds[1] = 150;

  //adjacency + boundary crossing search
  auto search_ptcl = OMEGA_H_LAMBDA( Omega_h::LO ielem)
  {
    if(!(ielem == elemIds[0])) return; //Assume all other elements are empty

     //Replace by particle parallel_for loop for all the remaining particles in this element
    for(Omega_h::LO iptcl = 0; iptcl < nptcl; ++iptcl)
    {
      auto ttv2v = Omega_h::gather_verts<4>(mesh2verts, ielem);
      const auto M = Omega_h::gather_vectors<4, 3>(coords, ttv2v);
      Omega_h::Write<Omega_h::Real> bcc(4, -1.0);

      bool res = g::find_barycentric_tet(M, dest[iptcl], bcc);
      std::cout << "-------\n elem " << ielem << " \n";

      if(g::all_positive(bcc.data(), 4))
      {
        part_flags.data()[iptcl] = 0;
        elemIds[iptcl] = ielem;

        std::cout << "********found in " << ielem << " \n";
        g::print_matrix(M);
        g::print_data(M, dest[iptcl], bcc);
        OMEGA_H_CHECK(g::almost_equal(bcc[0] + bcc[1] + bcc[2] +bcc[3], 1.0));
      }
      else
      {
        Omega_h::LO most_neg = g::most_negative_index(bcc.data(), 4);

        //get element ID
        auto dface_ind = dual_elems[ielem];

        auto beg_face = ielem * nsides_per_elem; //face index
        auto end_face = beg_face + nsides_per_elem;
        Omega_h::LO fIndex = 0;

        for(auto iface = beg_face; iface < end_face; ++iface) //not 0..3
        {
          Omega_h::LO faceID = downs[iface];

          auto fv2v = Omega_h::gather_verts<3>(face_verts, faceID);
          const auto face = Omega_h::gather_vectors<3, 3>(coords, fv2v);

         //g::get_face_coords( M, fIndex, abc);
          g::check_face(M, face, fIndex);

          if(!side_is_exposed[faceID])
          {
             //OMEGA_H_CHECK(side2side_elems[side + 1] - side2side_elems[side] == 2);
             auto adj_elem  = dual_faces[dface_ind];
             if(fIndex == most_neg)
             {
               elemIds[iptcl] = adj_elem;
               std::cout << "=====> For el|faceID=" << ielem << "," << downs[iface]  << " :ADJ elem= " << adj_elem << "\n";

               break;
             }
             ++dface_ind;
          }
          else if(bcc[fIndex] < 0) //fIndex == most_neg)
          {
            std::cout << "-------\n Call Wall collision for el,faceID " << ielem << "," << downs[iface] << "\n";
            Omega_h::Vector<3> xpoint;
            bool res = g::line_triangle_intx(face, orig[iptcl], dest[iptcl],   xpoint);

            if(res)
            {
              g::print_matrix(M);
              g::print_data(M, dest[iptcl], bcc);
              g::print_osh_vector(xpoint, "COLLISION POINT");
            }
          }
          ++fIndex;
        }
      }
    }
  };

  /*
  auto search_reduce = OMEGA_H_LAMBDA(Omega_h::LO ielem)
  {

    return 1;
  };*/


  bool done = false;
  while(!done )
  {
    Omega_h::parallel_for(mesh.nelems(),  search_ptcl, "search_ptcl");//mesh.nelems()
    // synchronize
    //done = Omega_h::parallel_reduce(nptcl, search_reduce, "search_reduce");
    //TODO replace by parallel_reduce or..
    done = true;
    for(int i=0; i<nptcl; ++i){ if(part_flags[i]) {done = false; break;} }

    //remain = true;
  }

  Omega_h::vtk::write_parallel("cube", &mesh);

  return 0;
}
//LO a....
//a.get(i)




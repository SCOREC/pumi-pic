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


#include "gitrm_utils.hpp"
#include "gitrm_adjacency.hpp"
#include "unit_tests.hpp"

namespace g = GITRm;

//#define DEBUG 1
// #define DO_TEST 1
int main(int argc, char** argv) {

  auto lib = Omega_h::Library(&argc, &argv);
  const auto world = lib.world();
  auto mesh = Omega_h::gmsh::read(argv[1], world);
  const auto dim = mesh.dim();

#if DO_TEST==1
  test_unit(&lib);
  print_mesh_stat(mesh);
  test_barycentric1(mesh);
  test_barycentric2(mesh);

#endif

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

  //barycentric
  auto bc_fun = OMEGA_H_LAMBDA(Omega_h::LO ielem)
  {
  //ielem = 32 ; //166
    auto ttv2v = Omega_h::gather_verts<4>(mesh2verts, ielem);
    const auto M = Omega_h::gather_vectors<4, 3>(coords, ttv2v);
    Omega_h::Write<Omega_h::Real> bc(4, -1.0);
    const Omega_h::Vector<3> dest{9.8,-0.3,0.5}; //{12,0.2,0.2};
    const Omega_h::Vector<3> orig{9.8,0.9,0.5}; //{12,0.2,0.2};


    bool res = g::find_barycentric(M, dest, bc);

#ifdef DEBUG
        std::cout << "-------\n elem " << ielem << " \n";
        g::print_matrix(M);
        g::print_osh_vector(dest, "point");
        g::print_array(bc.data(), 4, "BCoords");
    }
#endif // DEBUG

    if(g::all_positive(bc.data(), 4))
    {
      std::cout << "********found in " << ielem << " \n";
    }
    else
    {
      Omega_h::LO neg = g::most_negative_index(bc.data(), 4);

      //get element ID
      auto dface_ind = dual_elems[ielem];

      auto beg_face = ielem * nsides_per_elem; //face index
      auto end_face = beg_face + nsides_per_elem;
      Omega_h::LO fIndex = 0, indf=0;

      for(auto iface = beg_face; iface < end_face; ++iface)
      {
        Omega_h::LO faceID = downs[iface];

        auto fv2v = Omega_h::gather_verts<3>(face_verts, faceID);
        const auto face = Omega_h::gather_vectors<3, 3>(coords, fv2v);
#ifdef DEBUG
        g::check_face(M, face, indf);
        ++indf;
#endif // DEBUG

        if(!side_is_exposed[faceID])
        {
           //OMEGA_H_CHECK(side2side_elems[side + 1] - side2side_elems[side] == 2);
           auto adj_elem  = dual_faces[dface_ind];
           if(fIndex == neg)
           {
#ifdef DEBUG
             std::cout << "=====> For el|faceID=" << ielem << "," << downs[iface]
                        << " :ADJ elem= " << adj_elem << "\n";
#endif // DEBUG
             break;
           }
           ++dface_ind;
        }
        else if(fIndex == neg)
        {
          std::cout << "Call Wall collision for el|faceID " << ielem << "," << downs[iface] << "\n";
          Omega_h::Vector<3> xpoint;
          bool res = g::line_triangle_intx(M, face, indf, orig, dest,   xpoint);

          {
            std::cout << "-------\n collision out " << ielem << " \n";
            g::print_matrix(M);
            g::print_osh_vector(dest, "point");
            g::print_array(bc.data(), 4, "BCoords");
          }
          if(res)
          {
            std::cout << "FOUND \n";
            g::print_osh_vector(xpoint, "COLLISION POINT");
          }
        }
        ++fIndex;
      }

    }
  };

  Omega_h::parallel_for(204, bc_fun, "bc");//mesh.nelems()
  Omega_h::vtk::write_parallel("cube", &mesh);

  return 0;
}
//LO a....
//a.get(i)




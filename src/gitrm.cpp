#include <iostream>
#include <cmath>
#include <utility>

#include "Omega_h_for.hpp"
#include "Omega_h_file.hpp"  //gmsh
#include "Omega_h_tag.hpp"
#include "Omega_h_adj.hpp"
#include "Omega_h_array.hpp"
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

#include "gitrm_adjacency.hpp"
#include "unit_tests.hpp"

namespace o = Omega_h;
namespace g = GITRm;

#define DO_TEST 1
int main(int argc, char** argv) {

  auto lib = Omega_h::Library(&argc, &argv);
  const auto world = lib.world();
  auto mesh = Omega_h::gmsh::read(argv[1], world);

#if DO_TEST==1
  //test_unit(&lib);
  print_mesh_stat(mesh);
  test_barycentric1();
  test_barycentric2();
  test_barycentric_tri();
  test_line_tri_intx();
#endif

  // common mesh data
  // dual, up, down are Adj ~ Graph{arrays:LOs a2ab,ab2b}
  const auto dual = mesh.ask_dual();
  const auto down_r2f = mesh.ask_down(3, 2);
  const auto down_f2e = mesh.ask_down(2,1);
  const auto up_e2f = mesh.ask_up(1, 2);
  const auto up_f2r = mesh.ask_up(2, 3);
  //coordinates
  const auto mesh2verts = mesh.ask_elem_verts();
  const auto coords = mesh.coords();
  const auto face_verts =  mesh.ask_verts_of(2);//LOs
  const auto side_is_exposed = mark_exposed_sides(&mesh);

  const auto dim = mesh.dim();
  Omega_h::Int nelems = mesh.nelems();


  Omega_h::LO nptcl = 2; //total

  Omega_h::Write<Omega_h::Real> eFieldsPre(3*nptcl,0.1);
  Omega_h::Write<Omega_h::Real> bFieldsPre(3*nptcl,0.1);
  //mesh.add_tag(0, "EField", 3);


  //Simple particle data
  Omega_h::Few< Omega_h::Vector<3>, 3> dest{ {1.8,1.0,0.5}, {0.26,0.2,0.12}};
  Omega_h::Few< Omega_h::Vector<3>, 3> orig{ {0.8,0.3,0.7}, {0.15,0.1,0.2}};
  Omega_h::Write<Omega_h::Real> bccs(4*nptcl, -1.0);
  Omega_h::Write<Omega_h::Real> xpoints(3*nptcl, -1.0); //use sparse ?

  Omega_h::Write<Omega_h::LO> part_flags(nptcl, 1); // found or not
  //To store adj_elem for subsequent searches. Once ptcl is found/crossed, this is reset.
  Omega_h::Write<Omega_h::LO> elem_ids(nptcl); //next element to search for

  Omega_h::Write<Omega_h::LO> coll_adj_face_ids(nptcl, -1);
  // Particle ownership is not yet here.

  //Which particle belongs to which element ?
  elem_ids[0] = 26; //173{0.15,0.1,0.2}; 26{0.8,0.3,0.7}
  elem_ids[1] = 150; //temporary

  Omega_h::LO ptcls=1; //per group
  Omega_h::LO loops = 0;

  //.data() returns read only ?
  g::search_mesh(ptcls, nelems, orig.data(), dest.data(), dual, down_r2f, down_f2e, up_e2f, up_f2r, side_is_exposed, mesh2verts,
     coords, face_verts, part_flags, elem_ids, coll_adj_face_ids, bccs, xpoints, loops);

  g::print_array(&xpoints[0], 3, "XPOINT");
  g::print_array(&bccs[0], 3, "BCCS");
  Omega_h::vtk::write_parallel("cube", &mesh);
  std::cout << "LOOPS " << loops << "\n";
  return 0;
}
//LO a....
//a.get(i)




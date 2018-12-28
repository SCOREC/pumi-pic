#include <iostream>
#include <cmath>
#include <utility>
#include <cstdlib>
#include <sstream>

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

#define DO_TEST 0
int main(int argc, char** argv) {


  if(argc != 4)
  {
    std::cout << "Usage: ./search mesh init final\n";
    exit(1);
  }
  std::stringstream ss1(argv[2]);
  std::stringstream ss2(argv[3]);

  Omega_h::Vector<3> orig{0,0,0};
  Omega_h::Vector<3> dest{0,0,0};

  std::cout << "Origin: " << argv[2] << " Final " << argv[3] << "\n";
  int i=0;
  while(ss1.good())
  {
    std::string s;
    getline(ss1, s, ',');
    orig[i++] = atof(s.c_str());
  }

  i = 0;
  while(ss2.good())
  {
    std::string s;
    getline(ss2, s, ',');
    dest[i++] = atof(s.c_str());
  }

  auto lib = Omega_h::Library(&argc, &argv);
  const auto world = lib.world();
  auto mesh = Omega_h::gmsh::read(argv[1], world);

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

  Omega_h::LO nptcl = 1;
  Omega_h::Write<Omega_h::Real> bccs(4*nptcl, -1.0);
  Omega_h::Write<Omega_h::Real> xpoints(3*nptcl, -1.0);
  Omega_h::Write<Omega_h::LO> part_flags(nptcl, 1); // found or not
  //To store adj_elem for subsequent searches. Once ptcl is found/crossed, this is reset.
  Omega_h::Write<Omega_h::LO> elem_ids(nptcl); //next element to search for
  Omega_h::Write<Omega_h::LO> coll_adj_face_ids(nptcl, -1);
  //Which particle belongs to which element ?
  elem_ids[0] =0;

  Omega_h::LO ptcls=1; //per group
  Omega_h::LO loops = 0;

  //.data() returns read only ?
  g::search_mesh(ptcls, nelems, &orig, &dest, dual, down_r2f, down_f2e, up_e2f, up_f2r, side_is_exposed, mesh2verts,
     coords, face_verts, part_flags, elem_ids, coll_adj_face_ids, bccs, xpoints, loops);

  g::print_array(&xpoints[0], 3, "XPOINT");
  g::print_array(&bccs[0], 3, "BCCS");

  std::cout << "Element ID of Particle1 " << elem_ids[0] << " ;Xface " << coll_adj_face_ids[0] << " ;LOOPS " << loops << "\n";

  //search this particle in all elements in serial and compare with the above result

  Omega_h::Write<Omega_h::Real> bcc(4, -1.0);
  int found_in = -1;
  for(int ielem =0; ielem<nelems; ++ielem)
  {
    bcc= {-1,-1,-1,-1};
    const auto tetv2v = Omega_h::gather_verts<4>(mesh2verts, ielem);
    const auto M = Omega_h::gather_vectors<4, 3>(coords, tetv2v);
    const bool res = g::find_barycentric_tet(M, dest, bcc);
    if(g::all_positive(bcc.data(), 4))
      found_in = ielem;
  }

  OMEGA_H_CHECK(found_in == elem_ids[0]);
  return 0;
}
//origin: (0 0 0); dest: (-1 1 1); starts off on surface, not part of boundary, collision not detected
//-1,1,1 0,0,0  found in 144 !
// 0.1,0.1,0 11,1,-1 no xpt


#include "pumipic_adjacency.hpp"
#include "unit_tests.hpp"

namespace o = Omega_h;
namespace g = pumipic;

#define DO_TEST 0
int main(int argc, char** argv) {

  //Unexpected results: all on surface. origin: (0 0 0); dest: (-1 1 1); -1,1,1  0,0,0;  0.1,0.1,0  11,1,-1

  if(argc != 4)
  {
    std::cout << "Usage: ./search mesh init final\n Example: ./search cube.msh 2,0.5,0.2  4,0.9,0.3 \n";
    exit(1);
  }
  std::stringstream ss1(argv[2]);
  std::stringstream ss2(argv[3]);

  Omega_h::Vector<3> orig{0,0,0};
  Omega_h::Vector<3> dest{0,0,0};

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

  //only one particle is handled
  Omega_h::LO nptcl = 1;
  Omega_h::Write<Omega_h::Real> bccs(4*nptcl, -1.0);
  Omega_h::Write<Omega_h::Real> xpoints(3*nptcl, -1.0);
  Omega_h::Write<Omega_h::LO> part_flags(nptcl, 1); // found or not
  Omega_h::Write<Omega_h::LO> elem_ids(nptcl); //next element to search for
  Omega_h::Write<Omega_h::LO> coll_adj_face_ids(nptcl, -1);
  //Which particle belongs to which element ?
  elem_ids[0] =0;

  Omega_h::LO ptcls=1; //per group
  Omega_h::LO loops = 0;

  g::search_mesh(ptcls, nelems, &orig, &dest, dual, down_r2f, down_f2e, up_e2f, up_f2r, side_is_exposed, mesh2verts,
     coords, face_verts, part_flags, elem_ids, coll_adj_face_ids, bccs, xpoints, loops);

  Omega_h::Write<Omega_h::Real> bcc(4, -1.0);
  int found_in = -1;
  for(int ielem =0; ielem<nelems; ++ielem)
  {
    const auto tetv2v = Omega_h::gather_verts<4>(mesh2verts, ielem);
    const auto M = Omega_h::gather_vectors<4, 3>(coords, tetv2v);
    const bool res = g::find_barycentric_tet(M, dest, bcc);
    if(g::all_positive(bcc.data(), 4))
      found_in = ielem;
  }
#ifdef DEBUG
  std::cout << "Passed adjacency search test: origin: " << orig[0] << "," << orig[1] << "," << orig[2]
            <<  " Dest: " << dest[0] << "," << dest[1] << "," << dest[2] << ". Dest. element: "
            << elem_ids[0] << " found_in " << found_in << " #loops: " << loops << "\n";
#endif // DEBUG

  int status = (found_in == elem_ids[0])?0:1;
  return status;
}

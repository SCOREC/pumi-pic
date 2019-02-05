
#include "pumipic_adjacency.hpp"
#include "unit_tests.hpp"

namespace o = Omega_h;
namespace p = pumipic;

#define DO_TEST 0
int main(int argc, char** argv) {

  //TODO set points here X={0, 0.416667, 0.25}
  //Unexpected results : all on surface. origin: (0 0 0); dest: (-1 1 1); 0.1,0.1,0  11,1,-1

  Omega_h::Vector<3> orig{2, 0.5,0.3};
  Omega_h::Vector<3> dest{1.1,50,0};
  Omega_h::Write<Omega_h::Real> res(3,0);
  res[0] = 1.99091;
  res[1] = 1;
  res[2] = 0.29697;

  auto lib = Omega_h::Library(&argc, &argv);
  const auto world = lib.world();
  auto mesh = Omega_h::gmsh::read(argv[1], world);

  const auto dual = mesh.ask_dual();
  const auto down_r2f = mesh.ask_down(3, 2);
  //coordinates
  const auto mesh2verts = mesh.ask_elem_verts();
  const auto coords = mesh.coords();
  const auto face_verts =  mesh.ask_verts_of(2);//LOs
  const auto side_is_exposed = mark_exposed_sides(&mesh);

  Omega_h::Int nelems = mesh.nelems();


  const Omega_h::LO np = 1; 

  //Particle data
  Omega_h::Write<Omega_h::Real> x(np,0);
  Omega_h::Write<Omega_h::Real> y(np,0);
  Omega_h::Write<Omega_h::Real> z(np,0);
  Omega_h::Write<Omega_h::Real> xp(np,0);
  Omega_h::Write<Omega_h::Real> yp(np,0);
  Omega_h::Write<Omega_h::Real> zp(np,0);
  Omega_h::Write<Omega_h::Real> x0(np,0);
  Omega_h::Write<Omega_h::Real> y0(np,0);
  Omega_h::Write<Omega_h::Real> z0(np,0);

  Omega_h::Write<Omega_h::Real> bccs(4*np, -1.0);
  Omega_h::Write<Omega_h::Real> xpoints(3*np, -1.0);
  Omega_h::Write<Omega_h::LO> part_flags(np, 1); // to do or not
  Omega_h::Write<Omega_h::LO> elem_ids(np); //next element to search for
  Omega_h::Write<Omega_h::LO> coll_adj_face_ids(np, -1);

  elem_ids[0] = 87;
  x0[0] = orig[0];
  y0[0] = orig[1];
  z0[0] = orig[2];
  x[0] = dest[0];
  y[0] = dest[1];
  z[0] = dest[2];

  Omega_h::LO gpSize=1; //per group
  Omega_h::LO loops = 0;

  p::search_mesh(gpSize, nelems, x0, y0, z0, x, y, z, dual, down_r2f, side_is_exposed,
       mesh2verts, coords, face_verts, part_flags, elem_ids, coll_adj_face_ids, bccs, xpoints, loops);


  //for collision cross-check test non-containment in any element
  Omega_h::Write<Omega_h::Real> bcc(4, -1.0);
  int found_in = -1;
  for(int ielem =0; ielem<nelems; ++ielem)
  {
    const auto tetv2v = Omega_h::gather_verts<4>(mesh2verts, ielem);
    const auto M = Omega_h::gather_vectors<4, 3>(coords, tetv2v);
    g::find_barycentric_tet(M, dest, bcc);
    if(g::all_positive(bcc.data(), 4))
      found_in = ielem;
  }

  int status = 1;
  if(g::almost_equal(xpoints.data(), res.data(), 0) && found_in == -1)
    status = 0;
    
#if DEBUG>0
      std::cout << "Collision test: origin: " << orig[0] << "," << orig[1] << "," << orig[2]
            << " Dest: " << dest[0] << "," << dest[1] << "," << dest[2]
            << ". Xpt: " << xpoints[0] << "," << xpoints[1] << "," << xpoints[2]
            << " #loops: " << loops << " Element_id: " << found_in << " status " << status << "\n";
#endif // DEBUG

  return status;
}

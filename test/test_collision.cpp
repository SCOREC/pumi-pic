#include "Omega_h_file.hpp"
#include "Omega_h_for.hpp"
#include "Omega_h_mesh.hpp"
#include <psTypes.h>
#include <SellCSigma.h>
#include <SCS_Macros.h>
#include "pumipic_adjacency.hpp"
#include "unit_tests.hpp"
#include "pumipic_library.hpp"
namespace o = Omega_h;
namespace p = pumipic;

using particle_structs::Vector3d;
using particle_structs::SellCSigma;
using particle_structs::MemberTypes;
using particle_structs::lid_t;
using particle_structs::fp_t;
//To demonstrate push and adjacency search we store:
//-two fp_t[3] arrays, 'Vector3d', for the current and
// computed (pre adjacency search) positions, and
//-an integer to store the particles id
typedef MemberTypes<Vector3d, Vector3d, int> Particle;
typedef SellCSigma<Particle> SCS;

void setPosition(SCS* scs,
    o::Vector<3> orig, o::Vector<3> dest) {
  auto x_scs_d = scs->get<0>();
  auto xtgt_scs_d = scs->get<1>();
  auto setPos = SCS_LAMBDA(const int& eid, const int& pid, const bool& mask) {
    if (mask) {
      for(int i=0; i<3; i++) {
        x_scs_d(pid,i) = orig[i];
        xtgt_scs_d(pid,i) = dest[i];
      }
      printf("elm %d ptcl %d position %f %f %f target position %f %f %f\n",
             eid, pid,
             x_scs_d(pid,0), x_scs_d(pid,1), x_scs_d(pid,2),
             xtgt_scs_d(pid,0), xtgt_scs_d(pid,1), xtgt_scs_d(pid,2));
    }
  };
  scs->parallel_for(setPos);
}

int main(int argc, char** argv) {
  pumipic::Library pic_lib(&argc, &argv);
  Omega_h::Library& lib = pic_lib.omega_h_lib();
  const auto world = lib.world();
  auto mesh = o::gmsh::read(argv[1], world);

  o::Int nelems = mesh.nelems();

  const o::LO numPtcls = 1; 

  //Particle data
  o::Write<o::Real> xpoints(3*numPtcls, -1.0);

  //TODO set points here X={0, 0.416667, 0.25}
  //Unexpected results : all on surface. origin: (0 0 0); dest: (-1 1 1); 0.1,0.1,0  11,1,-1
  o::Write<o::Real> res({1.99091,1.0,0.29697});

  //TODO create SCS structure with one particle in element 159
  const int initialElement = 159;
  SCS::kkLidView ptcls_per_elem("ptcls_per_elem", nelems);
  //Element gids is left empty since there is no partitioning of the mesh yet
  SCS::kkGidView element_gids;
  Omega_h::parallel_for(nelems, OMEGA_H_LAMBDA(const int& i) {
    ptcls_per_elem(i) = 0;
    if (i == initialElement)
      ptcls_per_elem(i) = numPtcls;
  });
  //'sigma', 'V', and the 'policy' control the layout of the SCS structure 
  //in memory and can be ignored until performance is being evaluated.  These
  //are reasonable initial settings for OpenMP.
  const int sigma = INT_MAX; // full sorting
  const int V = 1024;
  Kokkos::TeamPolicy<Kokkos::DefaultExecutionSpace> policy(10000, 4);
  //Create the particle structure
  SCS* scs = new SCS(policy, sigma, V, nelems, numPtcls,
						       ptcls_per_elem, element_gids);

  o::Vector<3> orig{2, 0.5,0.3};
  o::Vector<3> dest{1.1,50,0};
  setPosition(scs,orig,dest);

  o::LO maxLoops = 100;
  const auto scsCapacity = scs->capacity();
  o::Write<o::LO> elem_ids(scsCapacity,-1);
  auto x = scs->get<0>();
  auto xtgt = scs->get<1>();
  auto pid = scs->get<2>();
  o::Write<o::Real> xpoints_d(3 * scsCapacity, "intersection points");
  o::Write<o::LO> xface_id(scsCapacity, "intersection faces");
  bool isFound = p::search_mesh<Particle>(mesh, scs, x, xtgt, pid, elem_ids,
                                          xpoints_d, xface_id, maxLoops);

  //coordinates
  const auto mesh2verts = mesh.ask_elem_verts();
  const auto coords = mesh.coords();
  //for collision cross-check test non-containment in any element
  o::Write<o::LO> found_in_dw(nelems,-1);
  o::parallel_for(nelems, OMEGA_H_LAMBDA(o::LO ielem) {
    const auto tetv2v = o::gather_verts<4>(mesh2verts, ielem);
    const auto M = o::gather_vectors<4, 3>(coords, tetv2v);
    o::Vector<4> bcc;
    g::find_barycentric_tet(M, dest, bcc);
    found_in_dw[ielem] = g::all_positive(bcc);
  });

  o::Read<o::LO> found_in_dr(found_in_dw);
  const auto found_in = o::get_max<o::LO>(found_in_dr);

  int status = 1;
  if(found_in == -1)
   status = 0;
    
  std::cout << "Collision test: origin: " << orig[0] << "," << orig[1] << "," << orig[2]
            << " Dest: " << dest[0] << "," << dest[1] << "," << dest[2]
            << " Element_id: " << found_in << " status " << status << "\n";

  return status;
}

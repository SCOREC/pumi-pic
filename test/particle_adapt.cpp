#include "particle_structs.hpp"
#include "Omega_h_build.hpp" // build_box
#include "Omega_h_library.hpp" // world
#include "Omega_h_mesh.hpp"
#include "Omega_h_file.hpp"
#include <Omega_h_metric.hpp>
#include <Omega_h_array_ops.hpp>
#include <Omega_h_adapt.hpp>
#include "team_policy.hpp"

using particle_structs::SellCSigma;
using particle_structs::MemberTypes;


typedef MemberTypes<double[3]> Type;
typedef Kokkos::DefaultExecutionSpace ExeSpace;
typedef SellCSigma<Type,ExeSpace> SCS;

int main(int argc, char* argv[]) {
  auto lib = Omega_h::Library(&argc, &argv);
  auto world = lib.world();
  auto mesh = Omega_h::build_box(world, OMEGA_H_SIMPLEX, 1, 1, 1, 2, 2, 0, false);

  // Initalize Particles

  int nElems = mesh.nelems();
  int nppe = 3;
  int nPtcls = mesh.nelems() * nppe;

  SCS::kkLidView ptclsPerElem("ptcls_per_elem", nElems);
  SCS::kkGidView elemGIDs("gids", nElems);
  Kokkos::parallel_for(nElems, KOKKOS_LAMBDA(const int i) {
    ptclsPerElem(i) = nppe;
    elemGIDs(i) = nElems;
  });

  Kokkos::TeamPolicy<ExeSpace> policy = pumipic::TeamPolicyAuto(nElems,32);
  SCS* ptcls = new SCS(policy, 5, 2, nElems, nPtcls, ptclsPerElem, elemGIDs);

  // Set Particle Positions

  auto cells2nodes = mesh.get_adj(Omega_h::FACE, Omega_h::VERT).ab2b;
  auto nodes2coords = mesh.coords();
  auto ptclPos = ptcls->get<0>();

  auto lambda = PS_LAMBDA(const int& e, const int& pid, const int& mask) {
    if(mask > 0) {
      auto elmVerts = Omega_h::gather_verts<3>(cells2nodes, Omega_h::LO(e));
      auto vtxCoords = Omega_h::gather_vectors<3,2>(nodes2coords, elmVerts);
      auto center = average(vtxCoords);
      int v = (pid / nElems) % 3; //nearest vertex
      auto pos = vtxCoords[v] + ((center - vtxCoords[v]) * .1); // point near vertex
      for (int i=0; i<2; i++)
        ptclPos(pid, i) = pos[i];
    }
  };
  ps::parallel_for(ptcls, lambda);

  // Adaptation

  auto metrics = Omega_h::get_implied_isos(&mesh);
  auto scalar = Omega_h::metric_eigenvalue_from_length(0.5);
  metrics = Omega_h::multiply_each_by(metrics, scalar);
  mesh.add_tag(Omega_h::VERT, "metric", 1, metrics);
  auto opts = Omega_h::AdaptOpts(&mesh);
  adapt(&mesh, opts);
  mesh.remove_tag(Omega_h::VERT, "metric");

  Omega_h::vtk::write_vtu("particleCube.vtu", &mesh);

  delete ptcls;
  return 0;
}
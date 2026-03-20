#include "particle_structs.hpp"
#include "Omega_h_build.hpp" // build_box
#include "Omega_h_library.hpp" // world
#include "Omega_h_mesh.hpp"
#include "Omega_h_file.hpp"
#include <Omega_h_metric.hpp>
#include <Omega_h_array_ops.hpp>
#include <Omega_h_adapt.hpp>
#include <Omega_h_for.hpp>
#include "team_policy.hpp"
#include <pcms/point_search.h>
#include <MemberTypeLibraries.h>
#include "pumipic_adaptation.hpp"

using particle_structs::SellCSigma;
using particle_structs::MemberTypes;

typedef MemberTypes<double[3], int> Type;
typedef Kokkos::DefaultExecutionSpace ExeSpace;
typedef SellCSigma<Type,ExeSpace> SCS;
typedef ps::ParticleStructure<Type,ExeSpace> PS;

void resize(PS*& ptcls, int newNElems) {
  int nPtcls = ptcls->nPtcls();
  PS::kkLidView ptclsPerElem("new_ptcls_per_elem", newNElems);
  PS::kkGidView elemGIDs("new_gids", newNElems);

  auto copyPtclsPerElem = PS_LAMBDA(const int& e, const int& pid, const int& mask) {
    if(mask > 0 && e < newNElems)
      Kokkos::atomic_add(&(ptclsPerElem(e)), 1);
  };
  ps::parallel_for(ptcls, copyPtclsPerElem);

  Kokkos::parallel_for(newNElems, KOKKOS_LAMBDA(const int i) {
    elemGIDs(i) = i;
  });

  Kokkos::TeamPolicy<ExeSpace> policy = pumipic::TeamPolicyAuto(newNElems,32);
  PS* newPtcls = new SCS(policy, 1, 32, newNElems, nPtcls, ptclsPerElem, elemGIDs);
  newPtcls->copyParticleData(ptcls);

  delete ptcls;
  ptcls = newPtcls;
}

int main(int argc, char* argv[]) {
  auto lib = Omega_h::Library(&argc, &argv);
  auto world = lib.world();
  auto mesh = Omega_h::build_box(world, OMEGA_H_SIMPLEX, 1, 1, 1, 2, 2, 0, false);
  const int dim = 2;

  // Initalize Particles

  int nElems = mesh.nelems();
  int nppe = 3;
  int nPtcls = mesh.nelems() * nppe;

  PS::kkLidView ptclsPerElem("ptcls_per_elem", nElems);
  PS::kkGidView elemGIDs("gids", nElems);
  Kokkos::parallel_for(nElems, KOKKOS_LAMBDA(const int i) {
    ptclsPerElem(i) = nppe;
    elemGIDs(i) = i;
  });

  Kokkos::TeamPolicy<ExeSpace> policy = pumipic::TeamPolicyAuto(nElems,32);
  PS* ptcls = new SCS(policy, 1, 32, nElems, nPtcls, ptclsPerElem, elemGIDs);

  // Set Particle Positions

  auto cells2nodes = mesh.get_adj(dim, Omega_h::VERT).ab2b;
  auto nodes2coords = mesh.coords();
  auto ptclPos = ptcls->get<0>();
  PS::kkLidView vtxPerElm("vtx_per_elm", nElems);

  auto setPositions = PS_LAMBDA(const int& e, const int& pid, const int& mask) {
    if(mask > 0) {
      auto elmVerts = Omega_h::gather_verts<dim+1>(cells2nodes, Omega_h::LO(e));
      auto vtxCoords = Omega_h::gather_vectors<dim+1,dim>(nodes2coords, elmVerts);
      auto center = average(vtxCoords);
      int v = Kokkos::atomic_fetch_inc(&vtxPerElm[e]); //cycle through vertices
      auto pos = vtxCoords[v] + ((center - vtxCoords[v]) * .5); // point near vertex
      printf("%f, %f\n", pos[0], pos[1]);
      for (int i=0; i<dim; i++)
        ptclPos(pid, i) = pos[i];
    }
  };
  ps::parallel_for(ptcls, setPositions);

  // Adaptation

  Omega_h::vtk::write_vtu("particleCubeBefore.vtu", &mesh);
  Omega_h::ParticleAdapt<dim> particleAdapt(ptcls);
  // double factors[]{1.8, 1.7, 0.6, 0.3};
  for (int i=0; i<1; i++) {
    auto metrics = Omega_h::get_implied_isos(&mesh);
    auto scalar = Omega_h::metric_eigenvalue_from_length(1.25);
    metrics = Omega_h::multiply_each_by(metrics, scalar);
    mesh.add_tag(Omega_h::VERT, "metric", 1, metrics);
    auto opts = Omega_h::AdaptOpts(&mesh);
    opts.xfer_opts.user_xfer = std::make_shared<Omega_h::ParticleAdapt<dim>>(particleAdapt);

    adapt(&mesh, opts);
    mesh.remove_tag(Omega_h::VERT, "metric");
  }
  Omega_h::vtk::write_vtu("particleCubeAfter.vtu", &mesh);

  // Paricle Search

  pcms::GridPointSearch search{mesh, 50, 50};
  Kokkos::View<pcms::Real*[dim]> points("test_points", nPtcls);
  auto copyPoints = PS_LAMBDA(const int& e, const int& pid, const int& mask) {
    if(mask > 0)
      for (int i=0; i<dim; i++)
        points(pid, i) = ptclPos(pid, i);
  };
  ps::parallel_for(ptcls, copyPoints);
  auto searchResults = search(points);

  printf("==COMPARE RESULTS==\n");
  auto printResults = PS_LAMBDA(const int& e, const int& pid, const int& mask) {
    if(mask > 0) {
      auto [dim, idx, coords] = searchResults(pid);
      printf("ptcl %d old %d search %d adapt %d\n", pid, e, idx, particleAdapt.ptclElems(pid));
    }
  };
  ps::parallel_for(ptcls, printResults);

  // Move Particle Elements

  resize(ptcls, mesh.nelems());
  PS::kkLidView newElement("new_element", ptcls->capacity());
  auto ptclID = ptcls->get<1>();
  auto getNewElement = PS_LAMBDA(const int& e, const int& pid, const int& mask) {
    ptclID(pid) = pid;
    if(mask > 0) {
      auto [dim, idx, coords] = searchResults(pid);
      newElement(pid) = idx;
    }
    else
      newElement(pid) = -1;
  };
  ps::parallel_for(ptcls, getNewElement);
  ptcls->rebuild(newElement);

  // Assert New Elements

  ptclID = ptcls->get<1>();
  PS::kkLidView failed = PS::kkLidView("failed", 1);
  auto assertElement = PS_LAMBDA(const int& e, const int& pid, const int& mask) {
    if(mask > 0) {
      const int id = ptclID(pid);
      const int destElem = newElement(id);
      if (destElem != e) {
        printf("[ERROR] Particle %d was moved to incorrect element %d (should be in element %d)\n", id, e, destElem);
        failed(0) = 1;
      }
    }
  };
  ps::parallel_for(ptcls, assertElement);
  int fails = ps::getLastValue(failed);

  delete ptcls;
  return fails;
}
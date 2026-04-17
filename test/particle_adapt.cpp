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
#include "pumipic_adaptation.hpp"

typedef ps::SellCSigma<Type,ExeSpace> SCS;
typedef ps::DPS<Type,ExeSpace> DPS;

PS* createPtclStructure(Omega_h::Mesh& mesh, int nelems, int ppe) {
  PS::kkLidView ptclsPerElem("ptcls_per_elem", nelems);
  PS::kkGidView elemGIDs("gids", nelems);
  Kokkos::parallel_for(nelems, KOKKOS_LAMBDA(const int i) {
    ptclsPerElem(i) = ppe;
    elemGIDs(i) = i;
  });

  Kokkos::TeamPolicy<ExeSpace> policy = pumipic::TeamPolicyAuto(nelems,32);
  // return new SCS(policy, 1, 32, nelems, nelems*ppe, ptclsPerElem, elemGIDs);
  return new DPS(policy, nelems, nelems*ppe, ptclsPerElem, elemGIDs);
}

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
  // PS* newPtcls = new SCS(policy, 1, 32, newNElems, nPtcls, ptclsPerElem, elemGIDs);
  DPS* newPtcls = new DPS(policy, newNElems, nPtcls, ptclsPerElem, elemGIDs);
  newPtcls->copyParticleData(static_cast<DPS*>(ptcls));

  delete ptcls;
  ptcls = newPtcls;
}

template<int dim>
void adaptMesh(Omega_h::Mesh& mesh, PS*& ptcls, Omega_h::ParticleAdapt<dim>& ptclAdapt, const std::vector<double>& length) {
  // double factors[]{1.8, 1.7, 0.6, 0.3};
  for (int i=0; i<length.size(); i++) {
    auto metrics = Omega_h::get_implied_isos(&mesh);
    auto scalar = Omega_h::metric_eigenvalue_from_length(length[i]);
    metrics = Omega_h::multiply_each_by(metrics, scalar);
    mesh.add_tag(Omega_h::VERT, "metric", 1, metrics);
    auto opts = Omega_h::AdaptOpts(&mesh);
    opts.xfer_opts.user_xfer = std::make_shared<Omega_h::ParticleAdapt<dim>>(ptclAdapt);

    adapt(&mesh, opts);
    mesh.remove_tag(Omega_h::VERT, "metric");
  }
  Omega_h::vtk::write_vtu("box_after_adapt.vtu", &mesh);
  // Omega_h::vtk::write_vtu("box_edges_after_adapt.vtu", &mesh, 1);
}

int migratePtclsAfterAdapt(Omega_h::Mesh& mesh, PS*& ptcls) {
  resize(ptcls, mesh.nelems());
  //Move ptcl elements
  PS::kkLidView newElement("new_element", ptcls->capacity());
  auto ptclPos = ptcls->get<POS>();
  auto ptclElem = ptcls->get<PARENT>();
  auto ptclDim = ptcls->get<DIM>();
  auto ptclID = ptcls->get<PID>();
  printf("\n== Particle Positions ==\nx, y, elem, dim\n");
  auto getNewElement = PS_LAMBDA(const int& e, const int& pid, const int& mask) {
    ptclID(pid) = pid;
    if(mask > 0) {
      newElement(pid) = ptclElem(pid);
      printf("%f, %f, %d, %d\n", ptclPos(pid, 0), ptclPos(pid, 1), ptclElem(pid), ptclDim(pid));
    }
    else
      newElement(pid) = -1;
  };
  ps::parallel_for(ptcls, getNewElement);
  ptcls->rebuild(newElement);

  //Assert ptcls moved
  ptclID = ptcls->get<PID>();
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
  return ps::getLastValue(failed);
}

template<int dim>
int compareWithSearch(Omega_h::Mesh& mesh, PS*& ptcls) {
  auto ptclPos = ptcls->get<POS>();
  pcms::GridPointSearch search{mesh, 50, 50};
  Kokkos::View<pcms::Real*[dim]> points("test_points", ptcls->capacity()*dim);
  auto copyPoints = PS_LAMBDA(const int& e, const int& pid, const int& mask) {
    if(mask > 0)
      for (int i=0; i<dim; i++)
        points(pid, i) = ptclPos(pid, i);
  };
  ps::parallel_for(ptcls, copyPoints);
  auto searchResults = search(points);

  auto ptclElem = ptcls->get<PARENT>();
  auto ptclID = ptcls->get<PID>();
  PS::kkLidView failed = PS::kkLidView("failed", 1);
  auto printResults = PS_LAMBDA(const int& e, const int& pid, const int& mask) {
    if(mask > 0) {
      auto [eDim, idx, coords] = searchResults(pid);
      if (idx != ptclElem(pid)) {
        printf("[ERROR] Particle %-5d : search elem %-5d != migration elem %-5d \n", ptclID(pid), idx, ptclElem(pid));
        failed(0) = 1;
      }

    }
  };
  ps::parallel_for(ptcls, printResults);
  return ps::getLastValue(failed);
}

template<int dim>
int isParticleInLowest(Omega_h::Mesh& mesh, PS*& ptcls, Omega_h::ParticleAdapt<dim>& ptclAdapt) {
  auto ptclElem = ptcls->get<PARENT>();
  auto ptclChild = ptcls->get<CHILD>();
  auto ptclDim = ptcls->get<DIM>();
  auto ptclID = ptcls->get<PID>();
  PS::kkLidView failed = PS::kkLidView("failed", 1);
  Omega_h::Adj upward[dim];
  Omega_h::Adj downward[dim];
  for (int i=0; i<dim; i++) {
    upward[i] = mesh.ask_up(i, dim);
    downward[i] = mesh.ask_down(dim, i);
  }

  auto printResults = PS_LAMBDA(const int& e, const int& pid, const int& mask) {
    if (mask <= 0) return;
    auto d = ptclDim(pid);
    auto child = ptclAdapt.getChildElem(downward, d, ptclElem(pid), ptclChild(pid));
    auto lowestElemIdx = upward[d].a2ab[child];
    auto lowestElem = upward[d].ab2b[lowestElemIdx];
    if (ptclElem(pid) == lowestElem) return;
    printf("Ptcl %-2d: Not on lowest elem. Is %-2d should be %-2d\n", ptclID(pid), ptclElem(pid), lowestElem);
    failed(0) = 1;
  };
  ps::parallel_for(ptcls, printResults);
  return ps::getLastValue(failed);
}

template<int dim>
int testVerts(Omega_h::Mesh& mesh)
{
  printf("== Test: Migrate ptcl from vertex to vertex ==\n");
  PS* ptcls = createPtclStructure(mesh, mesh.nverts(), 1);
  Omega_h::ParticleAdapt<dim> ptclAdapt(ptcls, mesh);
  auto vert2elem = mesh.ask_up(Omega_h::VERT, dim);
  auto nodes2coords = mesh.coords();
  auto ptclPos = ptcls->get<POS>();
  auto ptclElem = ptcls->get<PARENT>();
  auto ptclDim = ptcls->get<DIM>();

  auto setPtclInfo = PS_LAMBDA(const int& e, const int& pid, const int& mask) {
    if(mask > 0) {
      auto elem_begin = vert2elem.a2ab[e];
      auto elem = vert2elem.ab2b[elem_begin];
      auto pos = get_vector<dim>(nodes2coords, Omega_h::LO(e));
      for (int i=0; i<dim; i++)
        ptclPos(pid, i) = pos[i];
      ptclElem(pid) = elem;
      ptclDim(pid) = 0;
    }
  };
  ps::parallel_for(ptcls, setPtclInfo);
  adaptMesh<dim>(mesh, ptcls, ptclAdapt, {.75});
  int fails = migratePtclsAfterAdapt(mesh, ptcls);
  fails += compareWithSearch<dim>(mesh, ptcls);
  fails += isParticleInLowest<dim>(mesh, ptcls, ptclAdapt);
  delete ptcls;
  return fails;
}

template<int dim>
int testEdges(Omega_h::Mesh& mesh)
{
  printf("== Test: Migrate ptcl from edges ==\n");
  PS* ptcls = createPtclStructure(mesh, mesh.nedges(), 1);
  Omega_h::ParticleAdapt<dim> ptclAdapt(ptcls, mesh);

  Omega_h::Adj downward[dim];
  for (int i=0; i<dim; i++) downward[i] = mesh.ask_down(dim, i);
  auto edge2elem = mesh.ask_up(Omega_h::EDGE, dim);
  auto edge2verts = mesh.get_adj(Omega_h::EDGE, Omega_h::VERT).ab2b;
  auto nodes2coords = mesh.coords();
  auto ptclPos = ptcls->get<POS>();
  auto ptclElem = ptcls->get<PARENT>();
  auto ptclChild = ptcls->get<CHILD>();
  auto ptclDim = ptcls->get<DIM>();

  auto setPtclInfo = PS_LAMBDA(const int& e, const int& pid, const int& mask) {
    if(mask > 0) {
      auto elem_begin = edge2elem.a2ab[e];
      auto parent = edge2elem.ab2b[elem_begin];
      auto edgeVerts = Omega_h::gather_verts<2>(edge2verts, Omega_h::LO(e));
      auto vtxCoords = Omega_h::gather_vectors<2,2>(nodes2coords, edgeVerts);
      auto pos = (e % 2 == 0) ? average(vtxCoords) : vtxCoords[0] + ((vtxCoords[1] - vtxCoords[0]) / 4);
      for (int i=0; i<dim; i++)
        ptclPos(pid, i) = pos[i];
      ptclElem(pid) = parent;
      ptclChild(pid) = ptclAdapt.getChildIndex(downward, 1, parent, e);
      ptclDim(pid) = 1;
    }
  };
  ps::parallel_for(ptcls, setPtclInfo);
  adaptMesh<dim>(mesh, ptcls, ptclAdapt, {.75});
  int fails = migratePtclsAfterAdapt(mesh, ptcls);
  fails += compareWithSearch<dim>(mesh, ptcls);
  fails += isParticleInLowest<dim>(mesh, ptcls, ptclAdapt);
  delete ptcls;
  return fails;
}

template<int dim>
int testAll(Omega_h::Mesh& mesh)
{
  PS* ptcls = createPtclStructure(mesh, mesh.nelems(), 3);
  Omega_h::ParticleAdapt<dim> ptclAdapt(ptcls, mesh);
  auto cells2nodes = mesh.get_adj(dim, Omega_h::VERT).ab2b;
  auto nodes2coords = mesh.coords();
  auto ptclPos = ptcls->get<POS>();
  auto ptclElem = ptcls->get<PARENT>();
  auto ptclDim = ptcls->get<DIM>();
  PS::kkLidView vtxPerElm("vtx_per_elm", mesh.nelems());

  auto setPtclInfo = PS_LAMBDA(const int& e, const int& pid, const int& mask) {
    if(mask > 0) {
      auto elmVerts = Omega_h::gather_verts<dim+1>(cells2nodes, Omega_h::LO(e));
      auto vtxCoords = Omega_h::gather_vectors<dim+1,dim>(nodes2coords, elmVerts);
      auto center = average(vtxCoords);
      int v = Kokkos::atomic_fetch_inc(&vtxPerElm[e]); //cycle through vertices
      auto pos = vtxCoords[v] + ((center - vtxCoords[v]) * .5); // point near vertex
      for (int i=0; i<dim; i++)
        ptclPos(pid, i) = pos[i];
      ptclElem(pid) = e;
      ptclDim(pid) = dim;
    }
  };
  ps::parallel_for(ptcls, setPtclInfo);

  // Adaptation
  adaptMesh<dim>(mesh, ptcls, ptclAdapt, {.75});
  int fails = migratePtclsAfterAdapt(mesh, ptcls);
  fails += compareWithSearch<dim>(mesh, ptcls);
  delete ptcls;
  return fails;
}

int main(int argc, char* argv[]) {
  auto lib = Omega_h::Library(&argc, &argv);
  auto world = lib.world();
  auto mesh = Omega_h::build_box(world, OMEGA_H_SIMPLEX, 1, 1, 1, 2, 2, 0, false);
  Omega_h::vtk::write_vtu("box_before_adapt.vtu", &mesh);
  const int dim = 2;
  // int fails = testVerts<dim>(mesh);
  // int fails = testEdges<dim>(mesh);
  int fails = testAll<dim>(mesh);
  return fails;
}
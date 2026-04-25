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
namespace OH = Omega_h;

PS* createPtclStructure(OH::Mesh& mesh, int nelems, int ppe) {
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
void adaptMesh(OH::Mesh& mesh, PS*& ptcls, OH::ParticleAdapt<dim>& ptclAdapt, const std::vector<double>& length) {
  // double factors[]{1.8, 1.7, 0.6, 0.3};
  for (int i=0; i<length.size(); i++) {
    auto metrics = OH::get_implied_isos(&mesh);
    auto scalar = OH::metric_eigenvalue_from_length(length[i]);
    metrics = OH::multiply_each_by(metrics, scalar);
    mesh.add_tag(OH::VERT, "metric", 1, metrics);
    auto opts = OH::AdaptOpts(&mesh);
    opts.xfer_opts.user_xfer = std::make_shared<OH::ParticleAdapt<dim>>(ptclAdapt);

    adapt(&mesh, opts);
    mesh.remove_tag(OH::VERT, "metric");
  }
  OH::vtk::write_vtu("box_after_adapt.vtu", &mesh);
  OH::vtk::write_vtu("box_edges_after_adapt.vtu", &mesh, 1);
}

template<int dim>
int migratePtclsAfterAdapt(OH::ParticleAdapt<dim>& ptclAdapt) {
  OH::Mesh& mesh = ptclAdapt.mesh;
  PS*& ptcls = ptclAdapt.ptcls;
  resize(ptcls, mesh.nelems());
  //Move ptcl elements
  PS::kkLidView newElement("new_element", ptcls->capacity());
  ptclAdapt.update(mesh);
  auto ptclID = ptcls->get<PID>();
  printf("\n== Particle Positions ==\nx, y, \"(pid, parent, child, dim)\"\n");
  auto getNewElement = PS_LAMBDA(const int& e, const int& pid, const int& mask) {
    ptclID(pid) = pid;
    if(mask > 0) {
      newElement(pid) = ptclAdapt.pParent(pid);
      printf("%f, %f, \"(%d, %d, %d, %d)\"\n", ptclAdapt.pPos(pid, 0), ptclAdapt.pPos(pid, 1), pid, newElement(pid), ptclAdapt.getChildElem(pid), ptclAdapt.pDim(pid));
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
int compareWithPosition(OH::ParticleAdapt<dim>& ptclAdapt) {
  ptclAdapt.update(ptclAdapt.mesh);
  auto vert2coords = ptclAdapt.mesh.coords();
  auto edge2verts = ptclAdapt.mesh.ask_verts_of(OH::EDGE);
  PS::kkLidView failed = PS::kkLidView("failed", 1);
  auto getNewElement = PS_LAMBDA(const int& e, const int& pid, const int& mask) {
    if (mask <= 0) return;
    auto pPos = ptclAdapt.getPos(pid);
    if (ptclAdapt.pDim(pid) == 0) {
      auto verts = gather_verts<dim+1>(ptclAdapt.new_downward[0].ab2b, OH::LO(ptclAdapt.pParent(pid)));
      auto coords = gather_vectors<dim+1,dim>(vert2coords, verts);
      if (!OH::are_close(coords[ptclAdapt.pChild(pid)], pPos)) {
        printf("[ERROR] Particle %d not at correct vertex\n", pid);
        failed(0) = 1;
      }
    }
    else if (ptclAdapt.pDim(pid) == 1) {
      auto child = ptclAdapt.getChildElem(pid);
      auto eVerts = gather_verts<2>(edge2verts, child);
      auto eCoords = gather_vectors<2, dim>(vert2coords, eVerts);
      if (OH::are_close(OH::distance(eCoords[0], pPos) + OH::distance(eCoords[1], pPos), OH::distance(eCoords[0], eCoords[1]))) return;
      printf("[ERROR] Particle %d is on edge %d which is not correct\n", pid, child);
      failed(0) = 1;
    }
  };
  ps::parallel_for(ptclAdapt.ptcls, getNewElement);
  return ps::getLastValue(failed);
}

template<int dim>
int compareWithSearch(OH::Mesh& mesh, PS*& ptcls) {
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
int isParticleInLowest(OH::Mesh& mesh, PS*& ptcls, OH::ParticleAdapt<dim>& ptclAdapt) {
  ptclAdapt.update(mesh);
  auto ptclID = ptcls->get<PID>();
  PS::kkLidView failed = PS::kkLidView("failed", 1);

  auto printResults = PS_LAMBDA(const int& e, const int& pid, const int& mask) {
    if (mask <= 0 || ptclAdapt.pDim(pid) == dim) return;
    auto d = ptclAdapt.pDim(pid);
    auto child = ptclAdapt.getChildElem(pid);
    auto lowestElemIdx = ptclAdapt.new_upward[d].a2ab[child];
    auto lowestElem = ptclAdapt.new_upward[d].ab2b[lowestElemIdx];
    if (ptclAdapt.pParent(pid) == lowestElem) return;
    printf("Ptcl %-2d: Not on lowest parent. Is (%-2d) should be (%-2d)\n", ptclID(pid), ptclAdapt.pParent(pid), lowestElem);
    failed(0) = 1;
  };
  ps::parallel_for(ptcls, printResults);
  return ps::getLastValue(failed);
}

template<int dim>
int testVerts(OH::Mesh mesh)
{
  printf("\n== Test: Migrate ptcl from vertices ==\n\n");
  PS* ptcls = createPtclStructure(mesh, mesh.nverts(), 1);
  OH::ParticleAdapt<dim> ptclAdapt(ptcls, mesh);
  auto nodes2coords = mesh.coords();

  auto setPtclInfo = PS_LAMBDA(const int& e, const int& pid, const int& mask) {
    if(mask > 0) {
      auto elem_begin = ptclAdapt.new_upward[0].a2ab[e];
      auto parent = ptclAdapt.new_upward[0].ab2b[elem_begin];
      auto pos = get_vector<dim>(nodes2coords, OH::LO(e));
      for (int i=0; i<dim; i++)
        ptclAdapt.pPos(pid, i) = pos[i];
      ptclAdapt.setPtcl(pid, 0, parent, e);
    }
  };
  ps::parallel_for(ptcls, setPtclInfo);
  adaptMesh<dim>(mesh, ptcls, ptclAdapt, {.75});
  int fails = migratePtclsAfterAdapt<dim>(ptclAdapt);
  fails += compareWithSearch<dim>(mesh, ptcls);
  fails += isParticleInLowest<dim>(mesh, ptcls, ptclAdapt);
  fails += compareWithPosition<dim>(ptclAdapt);
  delete ptcls;
  return fails;
}

template<int dim>
int testEdges(OH::Mesh mesh)
{
  printf("\n== Test: Migrate ptcl from edges ==\n\n");
  PS* ptcls = createPtclStructure(mesh, mesh.nedges(), 3);
  OH::ParticleAdapt<dim> ptclAdapt(ptcls, mesh);
  auto edge2verts = mesh.get_adj(OH::EDGE, OH::VERT).ab2b;
  auto nodes2coords = mesh.coords();
  PS::kkLidView vtxPerElm("vtx_per_elm", mesh.nedges());

  auto setPtclInfo = PS_LAMBDA(const int& e, const int& pid, const int& mask) {
    if(mask > 0) {
      auto elem_begin = ptclAdapt.new_upward[1].a2ab[e];
      auto parent = ptclAdapt.new_upward[1].ab2b[elem_begin];
      auto edgeVerts = OH::gather_verts<2>(edge2verts, OH::LO(e));
      auto vtxCoords = OH::gather_vectors<2,2>(nodes2coords, edgeVerts);
      int v = Kokkos::atomic_fetch_inc(&vtxPerElm[e]); //cycle through vertices
      auto center = average(vtxCoords);
      const double interval[3] = {.5, 1, 1.5};
      auto pos = vtxCoords[0] + ((center - vtxCoords[0]) * interval[v]); // point near vertex

      for (int i=0; i<dim; i++)
        ptclAdapt.pPos(pid, i) = pos[i];
      ptclAdapt.setPtcl(pid, 1, parent, e);
    }
  };
  ps::parallel_for(ptcls, setPtclInfo);
  adaptMesh<dim>(mesh, ptcls, ptclAdapt, {.75});
  int fails = migratePtclsAfterAdapt<dim>(ptclAdapt);
  fails += compareWithSearch<dim>(mesh, ptcls);
  fails += isParticleInLowest<dim>(mesh, ptcls, ptclAdapt);
  fails += compareWithPosition<dim>(ptclAdapt);
  delete ptcls;
  return fails;
}

template<int dim>
int testFaces(OH::Mesh mesh)
{
  printf("\n== Test: Migrate ptcl from faces ==\n\n");
  PS* ptcls = createPtclStructure(mesh, mesh.nelems(), 3);
  OH::ParticleAdapt<dim> ptclAdapt(ptcls, mesh);
  PS::kkLidView vtxPerElm("vtx_per_elm", mesh.nelems());
  auto nodes2coords = mesh.coords();

  auto setPtclInfo = PS_LAMBDA(const int& e, const int& pid, const int& mask) {
    if(mask > 0) {
      auto elmVerts = OH::gather_verts<dim+1>(ptclAdapt.new_downward->ab2b, OH::LO(e));
      auto vtxCoords = OH::gather_vectors<dim+1,dim>(nodes2coords, elmVerts);
      auto center = average(vtxCoords);
      int v = Kokkos::atomic_fetch_inc(&vtxPerElm[e]); //cycle through vertices
      auto pos = vtxCoords[v] + ((center - vtxCoords[v]) * .5); // point near vertex
      for (int i=0; i<dim; i++)
        ptclAdapt.pPos(pid, i) = pos[i];
      ptclAdapt.pParent(pid) = e;
      ptclAdapt.pDim(pid) = dim;
    }
  };
  ps::parallel_for(ptcls, setPtclInfo);

  // Adaptation
  adaptMesh<dim>(mesh, ptcls, ptclAdapt, {.75});
  int fails = migratePtclsAfterAdapt<dim>(ptclAdapt);
  fails += compareWithSearch<dim>(mesh, ptcls);
  fails += isParticleInLowest<dim>(mesh, ptcls, ptclAdapt);
  fails += compareWithPosition<dim>(ptclAdapt);
  delete ptcls;
  return fails;
}

int main(int argc, char* argv[]) {
  auto lib = OH::Library(&argc, &argv);
  auto world = lib.world();

  auto mesh2D = [&]() { return OH::build_box(world, OMEGA_H_SIMPLEX, 1, 1, 1, 2, 2, 0, false);};
  auto mesh = mesh2D();
  OH::vtk::write_vtu("box_before_adapt.vtu", &mesh);
  const int dim = 2;
  int fails = 0;
  fails += testVerts<dim>(mesh2D());
  fails += testEdges<dim>(mesh2D());
  fails += testFaces<dim>(mesh2D());
  // OH::printOrderInfo<dim>(mesh);
  return fails;
}
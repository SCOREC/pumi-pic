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
  Omega_h::vtk::write_vtu("box_edges_after_adapt.vtu", &mesh, 1);
}

template<int dim>
int migratePtclsAfterAdapt(Omega_h::ParticleAdapt<dim>& ptclAdapt) {
  Omega_h::Mesh& mesh = ptclAdapt.mesh;
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
int compareWithPosition(Omega_h::ParticleAdapt<dim>& ptclAdapt) {
  ptclAdapt.update(ptclAdapt.mesh);
  auto vert2coords = ptclAdapt.mesh.coords();
  PS::kkLidView failed = PS::kkLidView("failed", 1);
  auto getNewElement = PS_LAMBDA(const int& e, const int& pid, const int& mask) {
    if (mask <= 0) return;
    if (ptclAdapt.pDim(pid) == 0) {
      auto verts = gather_verts<dim+1>(ptclAdapt.new_downward[0].ab2b, Omega_h::LO(ptclAdapt.pParent(pid)));
      auto coords = gather_vectors<dim+1,dim>(vert2coords, verts);
      if (!Omega_h::are_close(coords[ptclAdapt.pChild(pid)][0], ptclAdapt.pPos(pid, 0)) || !Omega_h::are_close(coords[ptclAdapt.pChild(pid)][1], ptclAdapt.pPos(pid, 1))) {
        printf("[ERROR] Particle %d not at correct vertex\n", pid);
        failed(0) = 1;
      }
    }
  };
  ps::parallel_for(ptclAdapt.ptcls, getNewElement);
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
  ptclAdapt.update(mesh);
  auto ptclID = ptcls->get<PID>();
  PS::kkLidView failed = PS::kkLidView("failed", 1);

  auto printResults = PS_LAMBDA(const int& e, const int& pid, const int& mask) {
    if (mask <= 0) return;
    auto d = ptclAdapt.pDim(pid);
    auto child = ptclAdapt.getChildElem(pid);
    auto lowestElemIdx = ptclAdapt.new_upward[d].a2ab[child];
    auto lowestElem = ptclAdapt.new_upward[d].ab2b[lowestElemIdx];
    if (ptclAdapt.pParent(pid) == lowestElem) return;
    printf("Ptcl %-2d: Not on lowest elem. Is %-2d should be %-2d\n", ptclID(pid), ptclAdapt.pParent(pid), lowestElem);
    failed(0) = 1;
  };
  ps::parallel_for(ptcls, printResults);
  return ps::getLastValue(failed);
}

template<int dim>
int testVerts(Omega_h::Mesh mesh)
{
  printf("\n== Test: Migrate ptcl from vertices ==\n\n");
  PS* ptcls = createPtclStructure(mesh, mesh.nverts(), 1);
  Omega_h::ParticleAdapt<dim> ptclAdapt(ptcls, mesh);
  auto nodes2coords = mesh.coords();

  auto setPtclInfo = PS_LAMBDA(const int& e, const int& pid, const int& mask) {
    if(mask > 0) {
      auto elem_begin = ptclAdapt.new_upward[0].a2ab[e];
      auto parent = ptclAdapt.new_upward[0].ab2b[elem_begin];
      auto pos = get_vector<dim>(nodes2coords, Omega_h::LO(e));
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
int testEdges(Omega_h::Mesh mesh)
{
  printf("\n== Test: Migrate ptcl from edges ==\n\n");
  PS* ptcls = createPtclStructure(mesh, mesh.nedges(), 1);
  Omega_h::ParticleAdapt<dim> ptclAdapt(ptcls, mesh);
  auto edge2verts = mesh.get_adj(Omega_h::EDGE, Omega_h::VERT).ab2b;
  auto nodes2coords = mesh.coords();

  auto setPtclInfo = PS_LAMBDA(const int& e, const int& pid, const int& mask) {
    if(mask > 0) {
      auto elem_begin = ptclAdapt.new_upward[1].a2ab[e];
      auto parent = ptclAdapt.new_upward[1].ab2b[elem_begin];
      auto edgeVerts = Omega_h::gather_verts<2>(edge2verts, Omega_h::LO(e));
      auto vtxCoords = Omega_h::gather_vectors<2,2>(nodes2coords, edgeVerts);
      auto pos = (e % 2 == 0) ? average(vtxCoords) : vtxCoords[0] + ((vtxCoords[1] - vtxCoords[0]) / 4);
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
  delete ptcls;
  return fails;
}

template<int dim>
int testAll(Omega_h::Mesh mesh)
{
  PS* ptcls = createPtclStructure(mesh, mesh.nelems(), 3);
  Omega_h::ParticleAdapt<dim> ptclAdapt(ptcls, mesh);
  PS::kkLidView vtxPerElm("vtx_per_elm", mesh.nelems());
  auto nodes2coords = mesh.coords();

  auto setPtclInfo = PS_LAMBDA(const int& e, const int& pid, const int& mask) {
    if(mask > 0) {
      auto elmVerts = Omega_h::gather_verts<dim+1>(ptclAdapt.new_downward->ab2b, Omega_h::LO(e));
      auto vtxCoords = Omega_h::gather_vectors<dim+1,dim>(nodes2coords, elmVerts);
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
  delete ptcls;
  return fails;
}

namespace Omega_h {
  template <int dim>
  void printOrderInfo(Mesh& mesh) {
    printf("\n== ORDER INFO ==\n");
    auto elem2vert = mesh.get_adj(dim, Omega_h::VERT);
    auto elem2edge = mesh.get_adj(dim, Omega_h::EDGE);
    parallel_for(mesh.nelems(), OMEGA_H_LAMBDA(LO elem) {
      // auto i = elem2vert.a2ab[elem];
      auto verts = Omega_h::gather_verts<dim+1>(elem2vert.ab2b, Omega_h::LO(elem));
      auto edges = Omega_h::gather_down<dim+1>(elem2edge.ab2b, Omega_h::LO(elem));
      printf("Elem %d : Verts (%d, %d, %d) Edges (%d, %d, %d)\n", elem, verts[0], verts[1], verts[2], edges[0], edges[1], edges[2]);
    });
  }
}

int main(int argc, char* argv[]) {
  auto lib = Omega_h::Library(&argc, &argv);
  auto world = lib.world();

  auto mesh2D = [&]() { return Omega_h::build_box(world, OMEGA_H_SIMPLEX, 1, 1, 1, 2, 2, 0, false);};
  auto mesh = mesh2D();
  Omega_h::vtk::write_vtu("box_before_adapt.vtu", &mesh);
  const int dim = 2;
  int fails = 0;
  fails += testVerts<dim>(mesh2D());
  fails += testEdges<dim>(mesh2D());
  fails += testAll<dim>(mesh2D());
  // Omega_h::printOrderInfo<dim>(mesh);
  return fails;
}
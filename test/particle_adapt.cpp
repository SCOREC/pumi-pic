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
#include "pumipic_adaptation.hpp"

#ifdef PP_ENABLE_PCMS
#include <pcms/point_search.h>
#endif

#ifdef OMEGA_H_USE_EGADS
#include <Omega_h_egads.hpp>
#endif

namespace OH = Omega_h;
using particle_structs::MemberTypes;
enum MemberIndex{POS, PARENT, CHILD, PDIM, PID};
typedef MemberTypes<double[3], Omega_h::LO, Omega_h::Int, Omega_h::Int, int> Type;
typedef Kokkos::DefaultExecutionSpace ExeSpace;
typedef ps::ParticleStructure<Type,ExeSpace> PS;
typedef ps::SellCSigma<Type,ExeSpace> SCS;
typedef ps::DPS<Type,ExeSpace> DPS;

template<int dim>
using PADAPT = OH::ParticleAdapt<dim, PS, POS, PARENT, CHILD, PDIM>;

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
  PS* newPtcls = new DPS(policy, newNElems, nPtcls, ptclsPerElem, elemGIDs);
  copyParticleData(newPtcls, ptcls);
  delete ptcls;
  ptcls = newPtcls;
}

template<int dim>
void printParticleData(std::string filename, PADAPT<dim>& pAdapt) {
  #ifndef PP_USE_GPU
  std::ofstream outFile(filename);
  outFile << "x, y, z, dim, \"(pid, parent, child)\"\n";
  for (int pid=0; pid<pAdapt.ptcls->nPtcls(); pid++) {
    OH::Vector<3> pos = OH::zero_vector<3>();
    for (int i=0; i<dim; i++) pos[i] = pAdapt.pPos(pid, i);
    outFile << pos[0] << ", " << pos[1] << ", " << pos[2] << ", " << pAdapt.pDim(pid) << ", ";
    outFile << "\"(" << pid << ", " << pAdapt.pParent(pid) << ", " << pAdapt.getChildElem(pid) << ")\"\n";
  }
  #endif
}

template<int dim, int size>
void adaptMesh(PADAPT<dim>& pAdapt, OH::Few<double, size> length) {
  OH::vtk::write_vtu("box_before_adapt.vtu", &pAdapt.mesh);
  for (int i=0; i<length.size(); i++) {
    auto metrics = OH::get_implied_isos(&pAdapt.mesh);
    auto scalar = OH::metric_eigenvalue_from_length(length[i]);
    metrics = OH::multiply_each_by(metrics, scalar);
    pAdapt.mesh.add_tag(OH::VERT, "metric", 1, metrics);
    auto opts = OH::AdaptOpts(&pAdapt.mesh);
    opts.xfer_opts.user_xfer = std::make_shared<PADAPT<dim>>(pAdapt);
    adapt(&pAdapt.mesh, opts);
    pAdapt.mesh.remove_tag(OH::VERT, "metric");
  }
  OH::vtk::write_vtu("box_after_adapt.vtu", &pAdapt.mesh);
  OH::vtk::write_vtu("box_edges_after_adapt.vtu", &pAdapt.mesh, 1);
}

template<int dim>
int migratePtclsAfterAdapt(PADAPT<dim>& pAdapt) {
  OH::Mesh& mesh = pAdapt.mesh;
  PS*& ptcls = pAdapt.ptcls;
  resize(ptcls, mesh.nelems());
  //Move ptcl elements
  PS::kkLidView newElement("new_element", ptcls->capacity());
  pAdapt.update(mesh);
  auto ptclID = ptcls->get<PID>();
  printf("\n== Particle Positions ==\nx, y, z, dim, \"(pid, parent, child)\"\n");
  auto getNewElement = PS_LAMBDA(const int& e, const int& pid, const int& mask) {
    ptclID(pid) = pid;
    if(mask > 0) {
      newElement(pid) = pAdapt.pParent(pid);
      OH::Vector<3> pos = OH::zero_vector<3>();
      for (int i=0; i<dim; i++) pos[i] = pAdapt.pPos(pid, i);
      printf("%f, %f, %f, %d, \"(%d, %d, %d)\"\n", pos[0], pos[1], pos[2], pAdapt.pDim(pid), pid, newElement(pid), pAdapt.getChildElem(pid));
    }
    else newElement(pid) = -1;
  };
  ps::parallel_for(ptcls, getNewElement);

  printParticleData("particle_data_after.csv", pAdapt);
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
OMEGA_H_DEVICE double dist2Plane(OH::Matrix<dim, 3> tri, OH::Vector<dim> pos) {
  if constexpr (dim == 3) {
    auto normal = OH::get_triangle_normal(tri[0], tri[1], tri[2]);
    auto dist = (pos - tri[0]) * (normal) / OH::norm(normal);
    return dist;
  } else return 0;
}

template<int dim>
int compareWithPosition(PADAPT<dim>& pAdapt) {
  pAdapt.update(pAdapt.mesh);
  auto vert2coords = pAdapt.mesh.coords();
  auto edge2verts = pAdapt.mesh.ask_verts_of(OH::EDGE);
  auto face2verts = pAdapt.mesh.ask_verts_of(OH::FACE);
  PS::kkLidView failed = PS::kkLidView("failed", 1);
  auto getNewElement = PS_LAMBDA(const int& e, const int& pid, const int& mask) {
    if (mask <= 0) return;
    auto pPos = pAdapt.getPos(pid);
    if (pAdapt.pDim(pid) == OH::VERT) {
      auto verts = OH::gather_verts<dim+1>(pAdapt.downward[OH::VERT].ab2b, OH::LO(pAdapt.pParent(pid)));
      auto coords = OH::gather_vectors<dim+1,dim>(vert2coords, verts);
      if (!OH::are_close(coords[pAdapt.pChild(pid)], pPos)) {
        printf("[ERROR] Particle %d not at correct vertex\n", pid);
        failed(0) = 1;
      }
    }
    else if (pAdapt.pDim(pid) == OH::EDGE) {
      auto child = pAdapt.getChildElem(pid);
      auto eVerts = OH::gather_verts<2>(edge2verts, child);
      auto eCoords = OH::gather_vectors<2, dim>(vert2coords, eVerts);
      auto baryCoords = OH::barycentric_from_global<dim,1>(pPos, eCoords);
      if (!OH::is_barycentric_inside(baryCoords, OH::EPSILON)){
        printf("[ERROR] Particle %d is on edge %d which is not correct\n", pid, child);
        failed(0) = 1;
      }
    }
    else if (pAdapt.pDim(pid) == OH::FACE) {
      auto child = pAdapt.getChildElem(pid);
      auto eVerts = OH::gather_verts<3>(face2verts, child);
      auto eCoords = OH::gather_vectors<3, dim>(vert2coords, eVerts);
      auto baryCoords = OH::barycentric_from_global<dim,2>(pPos, eCoords);
      auto dist = dist2Plane(eCoords, pPos);
      if (!OH::are_close(dist, 0) || !OH::is_barycentric_inside(baryCoords, OH::EPSILON)){
        printf("[ERROR] Particle %d is on face %d which is not correct\n", pid, child);
        failed(0) = 1;
      }
    }
    auto parentVerts = OH::gather_verts<dim+1>(pAdapt.downward->ab2b, OH::LO(pAdapt.pParent(pid)));
    auto parentCoords = OH::gather_vectors<dim+1,dim>(vert2coords, parentVerts);
    auto baryCoords = OH::barycentric_from_global<dim,dim>(pPos, parentCoords);
    if (OH::is_barycentric_inside(baryCoords, OH::EPSILON)) return;
    printf("[ERROR] Particle %d is not in parent %d\n", pid, pAdapt.pParent(pid));
    failed(0) = 1;
  };
  ps::parallel_for(pAdapt.ptcls, getNewElement);
  return ps::getLastValue(failed);
}

#ifdef PP_ENABLE_PCMS
template<int dim>
int compareWithSearch(OH::ParticleAdapt<dim, PS>& pAdapt) {
  if (dim == 3) return 0;
  PS*& ptcls = pAdapt.ptcls;
  auto ptclPos = ptcls->get<POS>();
  pcms::GridPointSearch search{pAdapt.mesh, 50, 50};
  Kokkos::View<pcms::Real*[2]> points("test_points", ptcls->capacity()*dim);
  auto copyPoints = PS_LAMBDA(const int& e, const int& pid, const int& mask) {
    if(mask > 0)
      for (int i=0; i<dim; i++)
        points(pid, i) = ptclPos(pid, i);
  };
  ps::parallel_for(ptcls, copyPoints);
  auto searchResults = search(points);

  auto ptclElem = ptcls->get<PARENT>();
  auto ptclID = ptcls->get<PID>();
  auto printResults = PS_LAMBDA(const int& e, const int& pid, const int& mask) {
    if(mask > 0) {
      auto [eDim, idx, coords] = searchResults(pid);
      if (idx != ptclElem(pid))
        printf("[WARNING] Particle %-5d : search elem %-5d != migration elem %-5d \n", ptclID(pid), idx, ptclElem(pid));
    }
  };
  ps::parallel_for(ptcls, printResults);
  return 0;
}
#endif

template<int dim>
int isParticleInLowest(PADAPT<dim>& pAdapt) {
  pAdapt.update(pAdapt.mesh);
  PS*& ptcls = pAdapt.ptcls;
  auto ptclID = ptcls->get<PID>();
  PS::kkLidView failed = PS::kkLidView("failed", 1);
  auto printResults = PS_LAMBDA(const int& e, const int& pid, const int& mask) {
    if (mask <= 0 || pAdapt.pDim(pid) == dim) return;
    auto child = pAdapt.getChildElem(pid);
    auto lowestElem = pAdapt.getLowestParent(child, pAdapt.pDim(pid));
    if (pAdapt.pParent(pid) == lowestElem) return;
    printf("Ptcl %-2d: Not on lowest parent. Is (%-2d) should be (%-2d)\n", ptclID(pid), pAdapt.pParent(pid), lowestElem);
    failed(0) = 1;
  };
  ps::parallel_for(pAdapt.ptcls, printResults);
  return ps::getLastValue(failed);
}

template<int dim>
int runAdaptTests(PADAPT<dim>& pAdapt) {
  int fails = migratePtclsAfterAdapt<dim>(pAdapt);
  fails += isParticleInLowest<dim>(pAdapt);
  fails += compareWithPosition<dim>(pAdapt);
  #ifdef PP_ENABLE_PCMS
  fails += compareWithSearch<dim>(pAdapt);
  #endif
  return fails;
}

template <int test_dim, int mesh_dim, int size>
void initParticles(PADAPT<mesh_dim>& pAdapt, OH::Few<double, size> lengthCenter) {
  PS::kkLidView vtxPerElm("vtx_per_elm", pAdapt.mesh.nents(test_dim));
  auto test_ent2verts = pAdapt.mesh.get_adj(test_dim, OH::VERT).ab2b;
  auto nodes2coords = pAdapt.mesh.coords();
  auto setPtclInfo = PS_LAMBDA(const int& e, const int& pid, const int& mask) {
    if(mask > 0) {
      auto parent = pAdapt.getLowestParent(e, test_dim);
      auto elmVerts = OH::gather_verts<test_dim+1>(test_ent2verts, OH::LO(e));
      auto vtxCoords = OH::gather_vectors<test_dim+1, mesh_dim>(nodes2coords, elmVerts);
      auto center = average(vtxCoords);
      int v = Kokkos::atomic_fetch_inc(&vtxPerElm[e]); //cycle through vertices
      auto pos = vtxCoords[0] + ((center - vtxCoords[0]) * lengthCenter[v]); // point near vertex
      for (int i=0; i<mesh_dim; i++) pAdapt.pPos(pid, i) = pos[i];
      pAdapt.setPtcl(pid, test_dim, parent, e);
    }
  };
  ps::parallel_for(pAdapt.ptcls, setPtclInfo);
}

#if defined(OMEGA_H_USE_EGADS)
void compute_implied_metric(Omega_h::Mesh* mesh) {
  auto metrics = Omega_h::get_implied_metrics(mesh);
  metrics = Omega_h::limit_metric_gradation(mesh, metrics, 1.0);
  mesh->add_tag(Omega_h::VERT, "metric", Omega_h::symm_ncomps(mesh->dim()), metrics);
}

void compute_target_metric(Omega_h::Mesh* mesh, double length) {
  auto metric = Omega_h::diagonal(Omega_h::metric_eigenvalues_from_lengths( Omega_h::vector_3(length, length, length)));
  auto metrics = Omega_h::repeat_symm(mesh->nverts(), metric);
  mesh->add_tag(Omega_h::VERT, "target_metric", Omega_h::symm_ncomps(mesh->dim()), metrics);
}

template<int dim, int size>
void adaptSnapMesh(PADAPT<dim>& pAdapt, OH::AdaptOpts& opts, OH::Few<double, size> length) {
  OH::vtk::write_vtu("box_before_adapt.vtu", &pAdapt.mesh);
  printParticleData("particle_data_before.csv", pAdapt);
  pAdapt.setOpts(&opts);
  for (int i=0; i<length.size(); i++) {
    opts.xfer_opts.user_xfer = std::make_shared<PADAPT<dim>>(pAdapt);
    compute_implied_metric(&pAdapt.mesh);
    compute_target_metric(&pAdapt.mesh, length[i]);
    while (OH::approach_metric(&pAdapt.mesh, opts))
      OH::adapt(&pAdapt.mesh, opts);
    pAdapt.mesh.remove_tag(OH::VERT, "metric");
  }
  OH::vtk::write_vtu("box_after_adapt.vtu", &pAdapt.mesh);
  OH::vtk::write_vtu("box_edges_after_adapt.vtu", &pAdapt.mesh, 1);
}

template <int test_dim, int mesh_dim, int size>
int testSnap(OH::Mesh mesh, OH::AdaptOpts opts, OH::Few<double, size> lengthCenter)
{
  printf("\n== Test: Migrate ptcl from dimension %d ==\n\n", test_dim);
  PS* ptcls = createPtclStructure(mesh, mesh.nents(test_dim), lengthCenter.size());
  PADAPT<mesh_dim> pAdapt(ptcls, mesh, true);
  initParticles<test_dim, mesh_dim>(pAdapt, lengthCenter);

  // Adaptation
  adaptSnapMesh<mesh_dim>(pAdapt, opts, {.25});
  int fails = runAdaptTests(pAdapt);
  delete ptcls;
  return fails;
}
#endif

template<int dim, int size>
int testVerts(OH::Mesh mesh, OH::Few<double, size> averageLength)
{
  printf("\n== Test: Migrate ptcl from vertices ==\n\n");
  PS* ptcls = createPtclStructure(mesh, mesh.nverts(), 1);
  PADAPT<dim> pAdapt(ptcls, mesh);
  auto nodes2coords = mesh.coords();
  auto setPtclInfo = PS_LAMBDA(const int& e, const int& pid, const int& mask) {
    if(mask > 0) {
      auto parent = pAdapt.getLowestParent(e, OH::VERT);
      auto pos = OH::get_vector<dim>(nodes2coords, OH::LO(e));
      for (int i=0; i<dim; i++)
        pAdapt.pPos(pid, i) = pos[i];
      pAdapt.pParent(pid) = parent;
      // pAdapt.setPtcl(pid, OH::VERT, parent, e);
    }
  };
  ps::parallel_for(ptcls, setPtclInfo);
  pAdapt.populateFields();
  adaptMesh<dim>(pAdapt, averageLength);
  int fails = runAdaptTests(pAdapt);
  delete ptcls;
  return fails;
}

template <int test_dim, int mesh_dim, int size>
int testDimension(OH::Mesh mesh, OH::Few<double, size> lengthCenter)
{
  printf("\n== Test: Migrate ptcl from dimension %d ==\n\n", test_dim);
  PS* ptcls = createPtclStructure(mesh, mesh.nents(test_dim), lengthCenter.size());
  PADAPT<mesh_dim> pAdapt(ptcls, mesh);
  initParticles<test_dim, mesh_dim>(pAdapt, lengthCenter);

  // Adaptation
  adaptMesh<mesh_dim>(pAdapt, OH::Few<double, 1>{.5});
  int fails = runAdaptTests(pAdapt);
  delete ptcls;
  return fails;
}

int main(int argc, char* argv[]) {
  auto lib = OH::Library(&argc, &argv);
  auto world = lib.world();

  int fails = 0;

  // Refinement Tests:
  auto create2DMesh = [&]() { return OH::build_box(world, OMEGA_H_SIMPLEX, 1, 1, 1, 2, 2, 0, false);};
  auto create3DMesh = [&]() { return OH::build_box(world, OMEGA_H_SIMPLEX, 1, 1, 1, 2, 2, 2, false);};
  fails += testVerts<2>(create2DMesh(), OH::Few<double, 1>{.5});
  fails += testDimension<1,2>(create2DMesh(), OH::Few<double, 3>{.25, .5, 1});
  fails += testDimension<2,2>(create2DMesh(), OH::Few<double, 3>{.25, .5, 1});
  fails += testVerts<3>(create3DMesh(), OH::Few<double, 1>{.5});
  fails += testDimension<1,3>(create3DMesh(), OH::Few<double, 3>{.25, .5, 1});
  fails += testDimension<2,3>(create3DMesh(), OH::Few<double, 3>{.25, .5, 1});
  fails += testDimension<3,3>(create3DMesh(), OH::Few<double, 4>{.1, .25, .5, 1});

  // Coarsen Tests:
  auto large2DMesh = [&]() { return OH::build_box(world, OMEGA_H_SIMPLEX, 1, 1, 1, 4, 4, 0, false);};
  auto large3DMesh = [&]() { return OH::build_box(world, OMEGA_H_SIMPLEX, 1, 1, 1, 4, 4, 4, false);};
  fails += testVerts<2>(large2DMesh(), OH::Few<double, 1>{2});
  fails += testVerts<3>(large3DMesh(), OH::Few<double, 1>{2});

  // Coarsen, Refinement and Swap Tests:
  fails += testVerts<2>(large2DMesh(), OH::Few<double, 2>{2, .4});
  fails += testVerts<3>(large3DMesh(), OH::Few<double, 2>{2, .4});

  #if defined(OMEGA_H_USE_EGADS) && defined(OMEGA_H_USE_LIBMESHB)
  // OH::AdaptOpts opts3D(3);
  // Omega_h::Mesh mesh(&lib);
  // OH::meshb::read(&mesh, argv[2]);
  // opts3D.egads_model = OH::egads_load(argv[1]);
  // Omega_h::egads_reclassify(&mesh, opts3D.egads_model);
  // fails += testSnap<1,3>(mesh, opts3D, {.25, .5, 1});
  // if (opts3D.egads_model) OH::egads_free(opts3D.egads_model);
  #endif

  return fails;
}
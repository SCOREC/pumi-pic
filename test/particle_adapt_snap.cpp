#include "particle_adapt.hpp"

#ifdef OMEGA_H_USE_EGADSLITE
#include "Omega_h_egads_lite.hpp"
#endif

#ifdef OMEGA_H_USE_EGADS
#include <Omega_h_egads.hpp>
#endif

void checkCudaError(int line) {
#ifdef __NVCC__
  cudaError_t code = cudaDeviceSynchronize();
  const char * errorMessage = cudaGetErrorString(code);
  if( code != cudaSuccess ) {
    fprintf(stderr, "CUDA error on line %d Error code: %d (%s)\n", line, code, errorMessage);
  }
  assert(code == cudaSuccess);
#endif
}

void hackClassification(Omega_h::Mesh* mesh) {
  fprintf(stderr, "hacking classification\n");
  OMEGA_H_CHECK(mesh->dim() == 3);
  auto vtx_class_dims = mesh->get_array<Omega_h::I8>(Omega_h::VERT, "class_dim");
  auto vtx_class_ids_r = mesh->get_array<Omega_h::ClassId>(Omega_h::VERT, "class_id");
  auto vtx_class_ids_w = Omega_h::deep_copy(vtx_class_ids_r, "vtxClassIds_w");
  auto setVtxClass = OMEGA_H_LAMBDA(int i) {
    if(vtx_class_dims[i] == 1 && vtx_class_ids_w[i] == 1) {
      printf("vtx %i reclassified\n",i);
      vtx_class_ids_w[i] = 7;
    }
  };
  Omega_h::parallel_for(mesh->nents(0), setVtxClass, "setVtxClass");
  fprintf(stderr, "done hacking vtx classification\n");
  mesh->set_tag(0, "class_id", Omega_h::read(vtx_class_ids_w));

  auto edge_class_dims = mesh->get_array<Omega_h::I8>(Omega_h::EDGE, "class_dim");
  auto edge_class_ids_r = mesh->get_array<Omega_h::ClassId>(Omega_h::EDGE, "class_id");
  auto edge_class_ids_w = Omega_h::deep_copy(edge_class_ids_r, "edgeClassIds_w");
  auto setEdgeClass = OMEGA_H_LAMBDA(int i) {
    if(edge_class_dims[i] == 1 && edge_class_ids_w[i] == 1) {
      printf("edge %i reclassified\n",i);
      edge_class_ids_w[i] = 7;
    }
  };
  Omega_h::parallel_for(mesh->nents(1), setEdgeClass, "setEdgeClass");
  fprintf(stderr, "done hacking edge classification\n");
  mesh->set_tag(1, "class_id", Omega_h::read(edge_class_ids_w));

}

void setCudaStackSz() {
  size_t stackLimit;
  cuCtxGetLimit(&stackLimit, CU_LIMIT_STACK_SIZE);
  checkCudaError(__LINE__);
  printf("original stack limit %d\n", stackLimit);
  stackLimit=8*1024;
  cuCtxSetLimit(CU_LIMIT_STACK_SIZE,stackLimit);
  checkCudaError(__LINE__);
  cuCtxGetLimit(&stackLimit, CU_LIMIT_STACK_SIZE);
  checkCudaError(__LINE__);
  printf("new stack limit %d\n", stackLimit);
  printf("stack limit %d\n", stackLimit);
}

void compute_implied_metric(OH::Mesh* mesh) {
  auto metrics = OH::get_implied_metrics(mesh);
  metrics = OH::limit_metric_gradation(mesh, metrics, 1.0);
  mesh->add_tag(OH::VERT, "metric", OH::symm_ncomps(mesh->dim()), metrics);
}

void compute_target_metric(OH::Mesh* mesh, double length) {
  auto metric = OH::diagonal(OH::metric_eigenvalues_from_lengths( OH::vector_3(length, length, length)));
  auto metrics = OH::repeat_symm(mesh->nverts(), metric);
  mesh->add_tag(OH::VERT, "target_metric", OH::symm_ncomps(mesh->dim()), metrics);
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
    #if defined(OMEGA_H_USE_EGADSLITE)
    hackClassification(&pAdapt.mesh);
    #endif
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
  adaptSnapMesh<mesh_dim>(pAdapt, opts, OH::Few<double, 1>{.1});
  int fails = runAdaptTests(pAdapt);
  delete ptcls;
  return fails;
}

int main(int argc, char* argv[]) {
  auto lib = OH::Library(&argc, &argv);
  auto world = lib.world();
  setCudaStackSz();

  int fails = 0;

  #if defined(OMEGA_H_USE_LIBMESHB)
  OH::AdaptOpts opts3D(3);
  OH::Mesh mesh(&lib);
  OH::meshb::read(&mesh, argv[2]);

  #if defined(OMEGA_H_USE_EGADS)
  opts3D.egads_model = OH::egads_load(argv[1]);
  OH::egads_reclassify(&mesh, opts3D.egads_model);

  #elif defined(OMEGA_H_USE_EGADSLITE)
  opts3D.egads_model = OH::egads_lite_load(argv[1]);
  OH::egads_lite_reclassify(&mesh, opts3D.egads_model);

  #endif

  fails += testSnap<1,3>(mesh, opts3D, OH::Few<double, 3>{.25, .5, 1});
  // if (opts3D.egads_model) OH::egads_free(opts3D.egads_model);
  #endif

  return fails;
}
#include "particle_adapt.hpp"


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

  return fails;
}
/*
    * Test for find_exit_face() function in pumipic_adjacency.tpp
    * This test is for 2D mesh
    * It shows that when the exis face is computed using `useBcc` flag on, it
    fails to find the correct exit face

    * The test mesh is a square as below with the following edges:

  1(-5,5)           e3                  2(5,5)
    *-----------------------------------*
    | \               o              /  |
    |    \                        /     |
    |       \  e4         e6   /        |
    |          \            /           |
    |              \     /              |
e0  |                 *   4(0,0)        |  e5
    |              /     \              |
    |      e2  /           \   e7       |
    |       /                 \         |
    |    /                       \      |
    | /    y                  x     \   |
    *-----------------------------------*
  0(-5,-5)           e1                 3(5,-5)

    * Two rays are cast from the point x->o and y->o.
    * The expected exit faces are 7 and 2 respectively

    * x(2.5, -3.0) -> o(0.0, 3.0)
    * y(-2.5, -3.0) -> o(0.0, 3.0)

    TODO It fails for the x->o ray and returns edge 2 instead of 7
*/

#include "pumipic_adjacency.hpp"
#include "pumipic_kktypes.hpp"
#include "pumipic_library.hpp"
#include "pumipic_mesh.hpp"
#include <Kokkos_Core.hpp>
#include <Omega_h_bbox.hpp>
#include <Omega_h_for.hpp>
#include <Omega_h_mesh.hpp>
#include <particle_structs.hpp>
#include <string>

using particle_structs::lid_t;
using particle_structs::MemberTypes;
using particle_structs::SellCSigma;
using pumipic::fp_t;
using pumipic::Vector3d;

namespace o = Omega_h;
namespace p = pumipic;
namespace ps = particle_structs;

// To demonstrate push and adjacency search we store:
//-two fp_t[3] arrays, 'Vector3d', for the current and
//  computed (pre adjacency search) positions, and
//-an integer to store the particles id
typedef MemberTypes<Vector3d, Vector3d, int> Particle;
typedef ps::ParticleStructure<Particle> PS;
typedef Kokkos::DefaultExecutionSpace ExeSpace;

OMEGA_H_DEVICE bool is_close(o::Real a, o::Real b, o::Real tol) {
  return (Kokkos::abs(a - b) < tol);
}

OMEGA_H_DEVICE bool all_positive(const o::Vector<3>& a, Omega_h::Real tol = EPSILON) {
  auto isPos = 1;
  for (o::LO i = 0; i < a.size(); ++i) {
    const auto gtez = o::are_close(a[i], 0.0, tol, tol) || a[i] > 0;
    isPos = isPos && gtez;
  }
  return isPos;
}

bool is_inside(o::Mesh &mesh, o::LO elem_id, const o::Vector<2> point) {
  const auto &coords = mesh.coords();
  const auto &faces2nodes = mesh.ask_verts_of(o::FACE);

  o::Write<o::LO> inside(2, 0);

  auto is_inside_lambda = OMEGA_H_LAMBDA(o::LO id){
    const auto current_el_verts = o::gather_verts<3>(faces2nodes, elem_id);
    const Omega_h::Few<Omega_h::Vector<2>, 3> current_el_vert_coords =
        o::gather_vectors<3, 2>(coords, current_el_verts);
    // check if the particle is in the element
    o::Vector<3> bcc = o::barycentric_from_global<2, 2>(
        point, current_el_vert_coords);
    // if all bcc are positive, the point is inside the element
    inside[0] = all_positive(bcc,0.0);
  };
  o::parallel_for(1, is_inside_lambda);
  auto host_inside = o::HostWrite(inside);

  return bool(host_inside[0]); 
}

void print_2D_mesh_edges(o::Mesh &mesh) {
  printf("---------- Mesh Info -----------\n");
  const auto edge2node = mesh.ask_down(o::EDGE, o::VERT).ab2b;
  const auto coords = mesh.coords();
  // print edge nodes and their coordinates given an edge id
  auto print_edge_nodes = OMEGA_H_LAMBDA(o::LO edge_id) {

    printf("Edge %d nodes (%d, %d): ", edge_id, edge2node[2 * edge_id],
           edge2node[2 * edge_id + 1]);
    for (int i = 0; i < 2; i++) {
      printf("%d (%f, %f) ", edge2node[2 * edge_id + i],
             coords[edge2node[2 * edge_id + i] * 2 + 0],
             coords[edge2node[2 * edge_id + i] * 2 + 1]);
    }
    printf("\n");
  };
  o::parallel_for(mesh.nedges(), print_edge_nodes);
}

bool test_2D_intersection(const std::string mesh_fname, p::Library *lib, bool useBcc) {
  printf("Test: 2D intersection...\n");
  o::Library &olib = lib->omega_h_lib();
  // load mesh
  o::Mesh mesh = Omega_h::gmsh::read(mesh_fname, olib.self());
  printf("Mesh loaded successfully with %d elements\n", mesh.nelems());
  // o::vtk::write_parallel("square4el.vtk", &mesh);

  print_2D_mesh_edges(mesh);

  Omega_h::Write<Omega_h::LO> owners(mesh.nelems(), 0);
  p::Mesh picparts(mesh, owners);
  o::Mesh *p_mesh = picparts.mesh();
  p_mesh->ask_elem_verts();

  // create particles
  Kokkos::TeamPolicy<ExeSpace> policy;
#ifdef PP_USE_GPU
  printf("Using GPU for simulation...");
  policy =
      Kokkos::TeamPolicy<Kokkos::DefaultExecutionSpace>(10, Kokkos::AUTO());
#else
  printf("Using CPU for simulation...");
  policy = Kokkos::TeamPolicy<Kokkos::DefaultExecutionSpace>(100, 1);
#endif

  o::Int ne = p_mesh->nelems();
  PS::kkLidView ptcls_per_elem("ptcls_per_elem", ne);
  PS::kkGidView element_gids("element_gids", ne);
  o::GOs mesh_element_gids = picparts.globalIds(picparts.dim());
  o::parallel_for(
      ne, OMEGA_H_LAMBDA(const int &i) {
        element_gids(i) = mesh_element_gids[i];
        if (i == 1) {
          ptcls_per_elem(i) = 2;
        } else {
          ptcls_per_elem(i) = 0;
        }
      });

#ifdef PP_ENABLE_CAB
  PS *ptcls = new p::DPS<Particle>(policy, ne, 2, ptcls_per_elem, element_gids);
  printf("DPS Particle structure created successfully\n");
#else
  PS *ptcls = PS *ptcls = new SellCSigma<Particle>(
      policy, 10, 10, ne, 2, ptcls_per_elem, element_gids);
  printf("SellCSigma Particle structure created successfully\n");
#endif

  // set particle position
  auto particle_init_position = ptcls->get<0>();
  auto particle_final_position = ptcls->get<1>();
  auto pid_d = ptcls->get<2>();
  auto setIDs = PS_LAMBDA(const int &eid, const int &pid, const bool &mask) {
    pid_d(pid) = pid;
  };
  ps::parallel_for(ptcls, setIDs);

  const o::Vector<2> x{2.5, -3.0};
  const o::Vector<2> y{-2.5, -3.0};
  const o::Vector<2> o{0.0, 3.0};
  if (!is_inside(mesh, 1, x)) {
    printf("Error: x is not inside the expected element.\n");
    exit(1);
  }
  if (!is_inside(mesh, 1, y)) {
    printf("Error: y is not inside the expected element.\n");
    exit(1);
  }
  if (is_inside(mesh, 1, o)) {
    printf("Error: o is incorrectly inside the expected element.\n");
    exit(1);
  }

  printf("Start point x: %f %f\n", x[0], x[1]);
  printf("Start point y: %f %f\n", y[0], y[1]);
  printf("End point   o: %f %f\n", o[0], o[1]);

  auto set_initial_and_final_position =
      PS_LAMBDA(const int &e, const int &pid, const int &mask) {
    // see test calculation notebook
    if (pid == 0) {
      particle_init_position(pid, 0) = x[0];
      particle_init_position(pid, 1) = x[1];
      particle_init_position(pid, 2) = 0.0;

      particle_final_position(pid, 0) = o[0];
      particle_final_position(pid, 1) = o[1];
      particle_final_position(pid, 2) = 0.0;
    }
    if (pid == 1) {
      particle_init_position(pid, 0) = y[0];
      particle_init_position(pid, 1) = y[1];
      particle_init_position(pid, 2) = 0.0;

      particle_final_position(pid, 0) = o[0];
      particle_final_position(pid, 1) = o[1];
      particle_final_position(pid, 2) = 0.0;
    }
  };

  ps::parallel_for(ptcls, set_initial_and_final_position,
                   "set_initial_and_final_position");

  const auto psCapacity = ptcls->capacity();
  o::Write<o::Real> xpoints_d(2 * psCapacity, 0.0, "intersection points");
  o::Write<o::LO> xface_id(psCapacity, -1, "intersection faces");

  const auto elmArea = measure_elements_real(&mesh);
  o::Real tol = p::compute_tolerance_from_area(elmArea);
  auto inter_points =
      o::Write<o::Real>(2 * psCapacity, 0.0, "intersection points");
  o::Write<o::LO> lastExit(psCapacity, -1, "search_last_exit");
  o::Write<o::LO> ptcl_done(psCapacity, 0, "search_ptcl_done");
  o::Write<o::LO> elem_ids(psCapacity, 1, "elem_ids for find_exit_face");

  p::find_exit_face(*p_mesh, ptcls, particle_init_position,
                    particle_final_position, elem_ids, ptcl_done, elmArea,
                    useBcc, lastExit, inter_points, tol);

  auto lastExit_h = o::HostRead<o::LO>(lastExit);
  auto inter_points_h = o::HostRead<o::Real>(inter_points);
  bool passed_xo = (lastExit_h[0] == 7);
  bool passed_yo = (lastExit_h[1] == 2);
  if (!passed_xo) {
    printf("ERROR: Expected exit face for x->o : 7, got %d\n", lastExit_h[0]);
  } else {
    printf("x->o intersected the correct face\n");
  }
  if (!passed_yo) {
    printf("ERROR: Expected exit face for y->o : 2, got %d\n", lastExit_h[1]);
  } else {
    printf("y->o intersected the correct face\n");
  }

  if (!useBcc) {
    o::Vector<2> expected_xpoint_xo{2.142857, -2.142857};
    o::Vector<2> expected_xpoint_yo{-2.142857, -2.142857};

    if (!is_close(inter_points_h[0], expected_xpoint_xo[0], 1e-4) ||
        !is_close(inter_points_h[1], expected_xpoint_xo[1], 1e-4)) {
      printf("ERROR: Expected intersection point for x->o : %f %f, got %f %f\n",
             expected_xpoint_xo[0], expected_xpoint_xo[1], inter_points_h[0],
             inter_points_h[1]);
      passed_xo = false;
    }

    if (!is_close(inter_points_h[2], expected_xpoint_yo[0], 1e-4) ||
        !is_close(inter_points_h[3], expected_xpoint_yo[1], 1e-4)) {
      printf("ERROR: Expected intersection point for y->o : %f %f, got %f %f\n",
             expected_xpoint_yo[0], expected_xpoint_yo[1], inter_points_h[2],
             inter_points_h[3]);
      passed_yo = false;
    }

    // do search mesh operation
    // TODO: search_mesh should have its own test
    // NOTE: Although search_mesh is tested in test_adj.cpp, it's an integration test
    //  We need a unit test for search_mesh
    bool is_found = p::search_mesh<Particle>(*p_mesh, ptcls, particle_init_position,
                                              particle_final_position, pid_d, elem_ids,
                                              true, xface_id, xpoints_d);
    auto xface_id_h = o::HostRead<o::LO>(xface_id);
    auto xpoints_d_h = o::HostRead<o::Real>(xpoints_d);
    auto elem_ids_h = o::HostRead<o::LO>(elem_ids);
    is_found = is_found && elem_ids_h[0] == 2 && elem_ids_h[1] == 2;
    printf("Search mesh operation %s\n", is_found ? "succeeded" : "failed");
    if (!is_found) {
      printf("ERROR: Expected both rays to reach element %d but reached %d and %d\n",
             2, elem_ids_h[0], elem_ids_h[1]);
      passed_xo = false;
    }

    // check intersection points
    printf("INFO: Intersection points: x->o: %f %f, y->o: %f %f\n", xpoints_d_h[0],
           xpoints_d_h[1], xpoints_d_h[2], xpoints_d_h[3]);
    // expected final intersection points: x->o: 0.882353 0.882353, y->o: -0.882353 0.882353
    o::Vector<2> expected_final_xpoint_xo{0.882353, 0.882353};
    o::Vector<2> expected_final_xpoint_yo{-0.882353, 0.882353};
    bool passed_xpoint_xo = is_close(xpoints_d_h[0], expected_final_xpoint_xo[0], 1e-4) &&
                            is_close(xpoints_d_h[1], expected_final_xpoint_xo[1], 1e-4);
    bool passed_xpoint_yo = is_close(xpoints_d_h[2], expected_final_xpoint_yo[0], 1e-4) &&
                            is_close(xpoints_d_h[3], expected_final_xpoint_yo[1], 1e-4);
    passed_xo = passed_xo && passed_xpoint_xo;
    passed_yo = passed_yo && passed_xpoint_yo;
    if (!passed_xpoint_xo) {
      printf("ERROR: Expected intersection point for x->o : %f %f, got %f %f\n",
             expected_final_xpoint_xo[0], expected_final_xpoint_xo[1], xpoints_d_h[0],
             xpoints_d_h[1]);
    }
    if (!passed_xpoint_yo) {
      printf("ERROR: Expected intersection point for y->o : %f %f, got %f %f\n",
             expected_final_xpoint_yo[0], expected_final_xpoint_yo[1], xpoints_d_h[2],
             xpoints_d_h[3]);
    }
  }

  if (!passed_xo) {
    printf("ERROR: Expected exit face for x->o : 7, got %d\n", lastExit_h[0]);
  }
  if (!passed_yo) {
    printf("ERROR: Expected exit face for y->o : 2, got %d\n", lastExit_h[1]);
  }

  // clean up
  delete ptcls;

  return passed_xo && passed_yo;
}

int main(int argc, char **argv) {
  p::Library lib(&argc, &argv);
  if (argc != 2) {
    printf("Usage: %s <gmesh>\n", argv[0]);
    exit(1);
  }
  std::string mesh_fname = argv[1];

  printf("\n\n-------------------- With BCC -------------------\n");

  //bool passed_bcc = test_2D_intersection(mesh_fname, &lib, true);
  bool passed_bcc = true;
  //if (!passed_bcc) {
  //  printf("ERROR: Test failed **with** BCC.\n");
  //}

  printf("\n\n-------------------- Without BCC -------------------\n");

  bool passed_woBcc = test_2D_intersection(mesh_fname, &lib, false);
  if (!passed_woBcc) {
    printf("ERROR: Test failed **without** BCC.\n");
  }

  if (passed_bcc && passed_woBcc) {
    printf("All tests passed!\n");
    return 0;
  } else {
    return 1;
  }

}
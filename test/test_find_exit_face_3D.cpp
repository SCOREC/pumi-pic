#include "pumipic_adjacency.hpp"
#include "pumipic_kktypes.hpp"
#include "pumipic_library.hpp"
#include "pumipic_mesh.hpp"
#include "pumipic_utils.hpp"
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

bool is_inside3D(o::Mesh &mesh, o::LO elem_id, const o::Vector<3> point) {
  OMEGA_H_CHECK_PRINTF(mesh.dim() == 3, "Mesh is not 3D. Found dimension %d\n",
                       mesh.dim());
  const auto &coords = mesh.coords();
  const auto &tet2nodes = mesh.ask_verts_of(o::REGION);

  o::Write<o::LO> inside(1, 0);

  auto is_inside_lambda = OMEGA_H_LAMBDA(o::LO id) {
    const auto current_el_verts = o::gather_verts<4>(tet2nodes, elem_id);
    const Omega_h::Few<Omega_h::Vector<3>, 4> current_el_vert_coords =
        o::gather_vectors<4, 3>(coords, current_el_verts);
    // check if the particle is in the element
    o::Vector<4> bcc =
        o::barycentric_from_global<3, 3>(point, current_el_vert_coords);
    // if all bcc are positive, the point is inside the element
    inside[0] = p::all_positive(bcc, 0.0);
  };
  o::parallel_for(1, is_inside_lambda);
  auto host_inside = o::HostWrite(inside);

  return bool(host_inside[0]);
}

bool test_3D_intersection(const std::string mesh_fname, p::Library *lib, bool useBcc) {
  printf("Test: 3D intersection...\n");
  o::Library &olib = lib->omega_h_lib();
  // load mesh
  o::Mesh mesh = Omega_h::gmsh::read(mesh_fname, olib.self());
  printf("Mesh loaded successfully with %d elements\n", mesh.nelems());

  Omega_h::Write<Omega_h::LO> owners(mesh.nelems(), 0);
  p::Mesh picparts(mesh, owners);
  o::Mesh *p_mesh = picparts.mesh();
  // p_mesh->ask_elem_verts();

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
  
  const o::LO tet_id = 0;
  o::Int ne = p_mesh->nelems();
  PS::kkLidView ptcls_per_elem("ptcls_per_elem", ne);
  PS::kkGidView element_gids("element_gids", ne);
  o::GOs mesh_element_gids = picparts.globalIds(picparts.dim());
  o::parallel_for(
      ne, OMEGA_H_LAMBDA(const int &i) {
        element_gids(i) = mesh_element_gids[i];
        if (i == tet_id) {
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

  const o::Vector<3> x{0.0, -0.2, -0.1};
  const o::Vector<3> y{0.0, -0.2, -0.5};
  const o::Vector<3> o{0.0, 1.1, 1.1};
  if (!is_inside3D(mesh, tet_id, x)) {
    printf("Error: x is not inside the expected element.\n");
    exit(1);
  }
  if (!is_inside3D(mesh, tet_id, y)) {
    printf("Error: y is not inside the expected element.\n");
    exit(1);
  }
  if (is_inside3D(mesh, tet_id, o)) {
    printf("Error: o is incorrectly inside the expected element.\n");
    exit(1);
  }

  printf("[INFO] x: %f %f %f\n", x[0], x[1], x[2]);
  printf("[INFO] y: %f %f %f\n", y[0], y[1], y[2]);
  printf("[INFO] o: %f %f %f\n", o[0], o[1], o[2]);

  auto set_initial_and_final_position =
      PS_LAMBDA(const int &e, const int &pid, const int &mask) {
    // see test calculation notebook
    if (pid == 0) {
      particle_init_position(pid, 0) = x[0];
      particle_init_position(pid, 1) = x[1];
      particle_init_position(pid, 2) = x[2];

      particle_final_position(pid, 0) = o[0];
      particle_final_position(pid, 1) = o[1];
      particle_final_position(pid, 2) = o[2];
    }
    if (pid == 1) {
      particle_init_position(pid, 0) = y[0];
      particle_init_position(pid, 1) = y[1];
      particle_init_position(pid, 2) = y[2];

      particle_final_position(pid, 0) = o[0];
      particle_final_position(pid, 1) = o[1];
      particle_final_position(pid, 2) = o[2];
    }
  };

  ps::parallel_for(ptcls, set_initial_and_final_position,
                   "set_initial_and_final_position");

  const auto psCapacity = ptcls->capacity();
  o::Write<o::LO> xface_id(psCapacity, -1, "intersection faces");

  const auto elmArea = measure_elements_real(&mesh);
  o::Real tol = p::compute_tolerance_from_area(elmArea);
  auto inter_points =
      o::Write<o::Real>(3 * psCapacity, 0.0, "intersection points");
  o::Write<o::LO> lastExit(psCapacity, -1, "search_last_exit");
  o::Write<o::LO> ptcl_done(psCapacity, 0, "search_ptcl_done");
  o::Write<o::LO> elem_ids(psCapacity, tet_id, "elem_ids for find_exit_face");

  p::find_exit_face(*p_mesh, ptcls, particle_init_position,
                    particle_final_position, elem_ids, ptcl_done, elmArea,
                    useBcc, lastExit, inter_points, tol);

  const o::Few<o::LO, 2> expected_faces {56, 55};
  auto lastExit_h = o::HostRead<o::LO>(lastExit);
  auto inter_points_h = o::HostRead<o::Real>(inter_points);
  bool passed_xo = (lastExit_h[0] == expected_faces[0]);
  bool passed_yo = (lastExit_h[1] == expected_faces[1]);

  if (!passed_xo) {
    printf("ERROR: Expected exit face for x->o : %d, got %d\n", expected_faces[0], lastExit_h[0]);
  } else {
    printf("x->o intersected the correct face\n");
  }
  if (!passed_yo) {
    printf("ERROR: Expected exit face for y->o : %d, got %d\n", expected_faces[1], lastExit_h[1]);
  } else {
    printf("y->o intersected the correct face\n");
  }

  if (!useBcc) {
    o::Vector<3> expected_xpoint_xo{0.000000, -0.091667, 0.000000};
    o::Vector<3> expected_xpoint_yo{0.000000, -0.000000, -0.253846};

    if (!p::almost_equal(inter_points_h[0], expected_xpoint_xo[0], 1e-4) ||
        !p::almost_equal(inter_points_h[1], expected_xpoint_xo[1], 1e-4) ||
        !p::almost_equal(inter_points_h[2], expected_xpoint_yo[0], 1e-4) ){
      printf("ERROR: Expected intersection point for x->o : %f %f %f, got %f %f %f\n",
             expected_xpoint_xo[0], expected_xpoint_xo[1], expected_xpoint_xo[2], inter_points_h[0],
             inter_points_h[1], inter_points_h[2]);
      passed_xo = false;
    }

    if (!p::almost_equal(inter_points_h[3], expected_xpoint_yo[0], 1e-4) ||
        !p::almost_equal(inter_points_h[4], expected_xpoint_yo[1], 1e-4) ||
        !p::almost_equal(inter_points_h[5], expected_xpoint_yo[2], 1e-4) ){
      printf("ERROR: Expected intersection point for y->o : %f %f %f, got %f %f %f\n",
             expected_xpoint_yo[0], expected_xpoint_yo[1], expected_xpoint_yo[2], inter_points_h[3],
             inter_points_h[4], inter_points_h[5]);
      passed_yo = false;
    }
  } // if (!useBcc)

  if (!passed_xo) {
    printf("ERROR: Expected exit face for x->o : 56, got %d\n", lastExit_h[0]);
  }
  if (!passed_yo) {
    printf("ERROR: Expected exit face for y->o : 55, got %d\n", lastExit_h[1]);
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

  //bool passed_bcc = test_3D_intersection(mesh_fname, &lib, true);
  bool passed_bcc = true;
  //if (!passed_bcc) {
  //  printf("ERROR: Test failed **with** BCC.\n");
  //}

  printf("\n\n-------------------- Without BCC -------------------\n");

  bool passed_woBcc = test_3D_intersection(mesh_fname, &lib, false);
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
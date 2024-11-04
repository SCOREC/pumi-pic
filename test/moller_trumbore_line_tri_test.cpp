/*
    * Test for moller_trumbore_line_triangle() function in pumipic_adjacency.tpp
    * The mesh is a cube with 24 tetrahedra
*/
#include <Omega_h_bbox.hpp>
#include <Omega_h_defines.hpp>
#include <Omega_h_fail.hpp>
#include <Omega_h_for.hpp>
#include <Omega_h_macros.h>
#include <Omega_h_mesh.hpp>
#include <pumipic_adjacency.hpp>
#include <string>

namespace o = Omega_h;

OMEGA_H_DEVICE bool is_close(o::Real a, o::Real b, o::Real tol) {
  return (Kokkos::abs(a - b) < tol);
}

OMEGA_H_DEVICE bool all_positive(const o::Vector<4> &a,
                                 Omega_h::Real tol = EPSILON) {
  auto isPos = 1;
  for (o::LO i = 0; i < a.size(); ++i) {
    const auto gtez = o::are_close(a[i], 0.0, tol, tol) || a[i] > 0;
    isPos = isPos && gtez;
  }
  return isPos;
}

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
    inside[0] = all_positive(bcc, 0.0);
  };
  o::parallel_for(1, is_inside_lambda);
  auto host_inside = o::HostWrite(inside);

  return bool(host_inside[0]);
}

void print_2D_mesh_edges(o::Mesh &mesh) {
  printf("------------------ Mesh Info ------------------\n");
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
  // add a barrier to make sure all the prints are done
  printf("------------------ Mesh Info ------------------\n");
}

bool moller_trumbore_line_tri_test(const std::string mesh_fname, o::Library *lib) {
  printf("Test: moller_trumbore_line_tri from pumipic_adjacency.tpp ...\n");
  o::Mesh mesh = Omega_h::gmsh::read(mesh_fname, lib->self());
  printf("Mesh loaded successfully with %d elements\n", mesh.nelems());

  const o::LO tet_id = 0;
  const o::Vector<3> x{0.0, -0.2, -0.1};
  const o::Vector<3> y{0.0, -0.2, -0.5};
  const o::Vector<3> o{0.0, 1.1, 1.1};

  printf("[INFO] x: %f %f %f\n", x[0], x[1], x[2]);
  printf("[INFO] y: %f %f %f\n", y[0], y[1], y[2]);
  printf("[INFO] o: %f %f %f\n", o[0], o[1], o[2]);

  if (!is_inside3D(mesh, tet_id, x)) {
    printf("Error: x is not inside element 1.\n");
    exit(1);
  }
  if (!is_inside3D(mesh, tet_id, y)) {
    printf("Error: y is not inside element 1.\n");
    exit(1);
  }
  if (is_inside3D(mesh, tet_id, o)) {
    printf("Error: o should not be inside element 1.\n");
    exit(1);
  }

  // read adjacency data
  const auto& coords = mesh.coords();
  const auto& tet2tri = mesh.ask_down(o::REGION, o::FACE).ab2b;
  const auto& tet2nodes = mesh.ask_verts_of(o::REGION);
  const auto& face2nodes = mesh.ask_verts_of(o::FACE);
  o::Write<o::Real> xpoints(2*3, 0.0, "xpoints xo, yo");
  o::Write<o::LO> face_intersected(2, -1, "face_intersected");

  auto get_intersections = OMEGA_H_LAMBDA(o::LO ray_id) {
    const o::Few<o::LO, 4> tet_faces {tet2tri[4*tet_id], tet2tri[4*tet_id+1], tet2tri[4*tet_id+2], tet2tri[4*tet_id+3]};
    const auto tet_nodes = o::gather_verts<4>(tet2nodes, tet_id);
    const auto origin = (ray_id == 0) ? x : y;

    for (o::LO face_id : tet_faces) {
      auto face_nodes = o::gather_verts<3>(face2nodes, face_id);
      auto face_coords = o::gather_vectors<3, 3>(coords, face_nodes);
      o::Vector<3> xpoint{0.0, 0.0, 0.0};
      o::Real dprodj = 0.0;
      o::Real closeness = 0.0;
      o::LO flip = pumipic::isFaceFlipped(face_id, face_nodes, tet_nodes);
      bool success = success = pumipic::moller_trumbore_line_triangle(
          face_coords, origin, o, xpoint, 1e-6, flip, dprodj, closeness);
      if (success) {
        xpoints[ray_id * 3] = xpoint[0];
        xpoints[ray_id * 3 + 1] = xpoint[1];
        xpoints[ray_id * 3 + 2] = xpoint[2];
        face_intersected[ray_id] = face_id;
        printf("INFO: %s ray intersected face %d(%d,%d,%d) at point (%f, %f, %f) with dprodj %f and closeness %f\n",
               (ray_id == 0) ? "x->o" : "y->o", face_id, face_nodes[0], face_nodes[1], face_nodes[2],
               xpoint[0], xpoint[1], xpoint[2], dprodj, closeness);
      } // if intersects
    } // for faces
  };
  o::parallel_for(2, get_intersections, "get_intersections_run");

  o::Few<o::LO, 2> expected_face_intersected {56, 55};
  o::Few<o::Real, 2*3> expected_xpoints {0.000000, -0.091667, 0.000000, 0.000000, -0.000000, -0.253846};
  
  o::HostRead<o::LO> face_intersected_host(face_intersected);
  o::HostRead<o::Real> xpoints_host(xpoints);

  bool face_intersected_success =
      (face_intersected_host[0] == expected_face_intersected[0]) &&
      (face_intersected_host[1] == expected_face_intersected[1]);
  if (!face_intersected_success) {
    printf("Error: face_intersected failed\n");
    printf("Expected: %d %d, Found: %d %d\n", expected_face_intersected[0],
           expected_face_intersected[1], face_intersected_host[0],
           face_intersected_host[1]);
  }
  bool xpoints_success = is_close(xpoints_host[0], expected_xpoints[0], 1e-4) &&
                         is_close(xpoints_host[1], expected_xpoints[1], 1e-4) &&
                         is_close(xpoints_host[2], expected_xpoints[2], 1e-4) &&
                         is_close(xpoints_host[3], expected_xpoints[3], 1e-4) &&
                         is_close(xpoints_host[4], expected_xpoints[4], 1e-4) &&
                         is_close(xpoints_host[5], expected_xpoints[5], 1e-4);
  if (!xpoints_success) {
    printf("Error: xpoints failed\n");
    printf("Expected: %f %f %f %f %f %f, Found: %f %f %f %f %f %f\n",
           expected_xpoints[0], expected_xpoints[1], expected_xpoints[2],
           expected_xpoints[3], expected_xpoints[4], expected_xpoints[5],
           xpoints_host[0], xpoints_host[1], xpoints_host[2], xpoints_host[3],
           xpoints_host[4], xpoints_host[5]);
  }

  return face_intersected_success && xpoints_success;
}

int main(int argc, char **argv) {
  o::Library lib(&argc, &argv);
  if (argc != 2) {
    printf("Usage: %s <a 3D gmsh>\n", argv[0]);
    exit(1);
  }
  std::string mesh_fname = argv[1];

  if (!moller_trumbore_line_tri_test(mesh_fname, &lib)) {
    printf("Test failed\n");
    return 1;
  }
  printf("Test passed.\n");
  return 0;
}
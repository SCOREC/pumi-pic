/*
    * Test for line_edge_2d() function in pumipic_adjacency.tpp
    * Tests if it can find the intersection point between x->o and y->o rays and
    * edge 7 and 2 respectively

  1(-5,5)           e3                  2(5,5)
    *-----------------------------------*
    | \               o       p      /  |
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
    * p(2.5, 3.0) -> x(2.5, -3.0)

    * Expected intersection points:
      * x->o: (2.142857, -2.142857)
      * y->o: (-2.142857, -2.142857)

*/
#include <Omega_h_bbox.hpp>
#include <Omega_h_for.hpp>
#include <Omega_h_macros.h>
#include <Omega_h_mesh.hpp>
#include <pumipic_adjacency.hpp>
#include <string>

namespace o = Omega_h;

OMEGA_H_DEVICE bool is_close(o::Real a, o::Real b, o::Real tol) {
  return (Kokkos::abs(a - b) < tol);
}

OMEGA_H_DEVICE bool all_positive(const o::Vector<3> &a,
                                 Omega_h::Real tol = EPSILON) {
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

  auto is_inside_lambda = OMEGA_H_LAMBDA(o::LO id) {
    const auto current_el_verts = o::gather_verts<3>(faces2nodes, elem_id);
    const Omega_h::Few<Omega_h::Vector<2>, 3> current_el_vert_coords =
        o::gather_vectors<3, 2>(coords, current_el_verts);
    // check if the particle is in the element
    o::Vector<3> bcc =
        o::barycentric_from_global<2, 2>(point, current_el_vert_coords);
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

bool test_line_edge_2d(const std::string mesh_fname, o::Library *lib) {
  printf("Test: line_edge_2d from pumipic_adjacency.tpp ...\n");
  o::Mesh mesh = Omega_h::gmsh::read(mesh_fname, lib->self());
  printf("Mesh loaded successfully with %d elements\n", mesh.nelems());

  // const o::Vector<3> x {0.0, 2.0, -3.0};
  // const o::Vector<3> o {0.0, -4.0, -3.0};
  const o::Vector<2> x{2.5, -3.0};
  const o::Vector<2> y{-2.5, -3.0};
  const o::Vector<2> o{0.0, 3.0};
  const o::Vector<2> p{2.5, 3.0};
  if (!is_inside(mesh, 1, x)) {
    printf("Error: x is not inside element 1.\n");
    exit(1);
  }
  if (!is_inside(mesh, 1, y)) {
    printf("Error: y is not inside element 1.\n");
    exit(1);
  }
  if (is_inside(mesh, 1, o)) {
    printf("Error: o should not be inside element 1.\n");
    exit(1);
  }
  if (!is_inside(mesh, 2, p)) {
    printf("Error: p is not inside element 2.\n");
    exit(1);
  }

  printf("[INFO] x: %f %f\n", x[0], x[1]);
  printf("[INFO] y: %f %f\n", y[0], y[1]);
  printf("[INFO] o: %f %f\n", o[0], o[1]);
  printf("[INFO] p: %f %f\n", p[0], p[1]);

  const auto &coords = mesh.coords();
  const auto &edge2node = mesh.ask_down(o::EDGE, o::VERT).ab2b;
  const auto &face2node = mesh.ask_down(o::FACE, o::VERT).ab2b;
  o::Write<o::Real> xpoints(3 * 2, 0.0, "xpoints xo, yo");
  o::Write<o::LO> edge_intersected(3, -1, "edge_intersected");

  auto test_rays = OMEGA_H_LAMBDA(o::LO ray_id) {
    // x->o ray goes through edge 7
    // y->o ray goes through edge 2
    o::Vector<2> xpoint{0.0, 0.0};
    bool success = false;
    o::LO edge_id;
    o::LO face_id;
    if (ray_id == 0) {
      edge_id = 7;
      face_id = 1;
      printf("Testing x->o ray\n");
    } else if (ray_id == 1) {
      edge_id = 2;
      face_id = 1;
      printf("Testing y->o ray\n");
    } else if (ray_id == 3) {
      edge_id = 6;
      face_id = 2;
      printf("Testing p->x ray\n");
    }
    const auto edge_nodes = o::gather_verts<2>(edge2node, edge_id);
    const auto face_nodes = o::gather_verts<3>(face2node, face_id);
    // TODO: Since gmsh reads the mesh in cw orientation, the edge is flipped
    // remove the ! later after fixing the readgmsh in Omega_h
    bool flip = !pumipic::isFaceFlipped(-1/*not used*/, edge_nodes, face_nodes);
    // TODO: This flip check should go to another test
    if (ray_id == 0 && flip != 1) { // edge 7
      printf("Error: Edge 7 is **flipped**\n");
    }
    if (ray_id == 1 && flip != 0) { // edge 2
      printf("Error: Edge 2 is **not flipped**\n");
    }
    printf("[INFO] Edge %d (%d %d) flip: %d\n", edge_id, edge_nodes[0], edge_nodes[1], flip);
    printf("[INFO] Face %d nodes: %d %d %d\n", face_id, face_nodes[0], face_nodes[1], face_nodes[2]);
    const auto edge_coords = o::gather_vectors<2, 2>(coords, edge_nodes);
    printf("Edge %d coords: %d(%f %f) %d(%f %f)\n", edge_id, edge_nodes[0], edge_coords[0][0],
           edge_coords[0][1], edge_nodes[1], edge_coords[1][0], edge_coords[1][1]);
    if (ray_id == 0) {
      success =
          pumipic::line_edge_2d(edge_coords, x, o, xpoint, 1e-6, flip);
      if (!success) {
        printf("Error: x->o ray did not intersect edge 7\n");
      }
      printf("Intersection point x->o: %f %f\n", xpoint[0], xpoint[1]);
      xpoints[0] = xpoint[0];
      xpoints[1] = xpoint[1];
    } else if (ray_id == 1) {
      success =
          pumipic::line_edge_2d(edge_coords, y, o, xpoint, 1e-8, flip);
      if (!success) {
        printf("Error: y->o ray did not intersect edge 2\n");
      }
      printf("Intersection point y->o: %f %f\n", xpoint[0], xpoint[1]);
      xpoints[2] = xpoint[0];
      xpoints[3] = xpoint[1];
    } else if (ray_id == 2) {
      success =
          pumipic::line_edge_2d(edge_coords, p, x, xpoint, 1e-8, flip);
      if (!success) {
        printf("Error: p->x ray did not intersect edge 6\n");
      }
      printf("Intersection point p->x: %f %f\n", xpoint[0], xpoint[1]);
      xpoints[4] = xpoint[0];
      xpoints[5] = xpoint[1];
    }
    edge_intersected[ray_id] = (success) ? edge_id : -1;
  };
  o::parallel_for(3, test_rays);

  o::Vector<2> expected_xpoint_xo{2.142857, -2.142857};
  o::Vector<2> expected_xpoint_yo{-2.142857, -2.142857};

  auto host_xpoints = o::HostRead<o::Real>(xpoints);
  bool xo_success = is_close(host_xpoints[0], expected_xpoint_xo[0], 1e-4) &&
                    is_close(host_xpoints[1], expected_xpoint_xo[1], 1e-4);
  bool yo_success = is_close(host_xpoints[2], expected_xpoint_yo[0], 1e-4) &&
                    is_close(host_xpoints[3], expected_xpoint_yo[1], 1e-4);
  bool px_success = is_close(host_xpoints[4], 2.5, 1e-4);

  if (!xo_success) {
    printf("Error: x->o ray did not intersect edge 7 at the expected point\n");
    printf("Expected: (%f %f), Found: (%f,%f)\n", expected_xpoint_xo[0],
           expected_xpoint_xo[1], host_xpoints[0], host_xpoints[1]);
  }

  if (!yo_success) {
    printf("Error: y->o ray did not intersect edge 2 at the expected point\n");
    printf("Expected: (%f %f), Found: (%f,%f)\n", expected_xpoint_yo[0],
           expected_xpoint_yo[1], host_xpoints[2], host_xpoints[3]);
  }

  if (!px_success) {
    printf("Error: p->x ray did not intersect edge 6 at the expected point\n");
    printf("Expected x: %f, Found: %f\n", 2.5, host_xpoints[4]);
  }

  auto host_edge_intersected = o::HostRead<o::LO>(edge_intersected);
  bool found_edges = (host_edge_intersected[0] == 7) && (host_edge_intersected[1] == 2) && (host_edge_intersected[2] == 6);

  return xo_success && yo_success && found_edges;
}

int main(int argc, char **argv) {
  o::Library lib(&argc, &argv);
  if (argc != 2) {
    printf("Usage: %s <gmesh>\n", argv[0]);
    exit(1);
  }
  std::string mesh_fname = argv[1];

  if (!test_line_edge_2d(mesh_fname, &lib)) {
    printf("Test failed\n");
    return 1;
  }
  printf("Test passed\n");
  return 0;
}
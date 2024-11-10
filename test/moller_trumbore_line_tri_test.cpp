/*
* Properties of the function
*   - It intersects a ray (not a line segment) with a triagle
*   - Does not intersect the face that it entered through
*   - Paper link is given with the function declaration
*   - It should detect when the ray intersects an edge of the triangle but not doing here
*      //TODO Investigate this later.
*/
#include <Kokkos_Core.hpp>
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


/*
* Testing that the function intersects as ray i.e. gives intersection even if
* the line segment ends. Also tests what happens if it intersects and edge.
*
*
* Test Description: The line starts at element 0 and ends inside element 12.
* It enters element 12 interseting the edge between face 33 and 39.
* But it moves on and intersects face 28.
* //TODO check why it's not showing that it intersects face 33 and 39 since
* //TODO it goes though their common edge.
* //TODO It is seen when the element is 3. The ray enters from edge 56 and
* //TODO exits through the edge of face 58 and 54.
*/
bool moller_trumbore_extrapolated_intersection_test(const std::string mesh_fname, o::Library *lib, o::LO tet_id)
{
  printf("Test: moller_trumbore_line_tri test for element %d line extrapolation ...\n", tet_id);
  o::Mesh mesh = Omega_h::gmsh::read(mesh_fname, lib->self());
  printf("Mesh loaded successfully with %d elements\n", mesh.nelems());
  const auto elmArea = measure_elements_real(&mesh);
  o::Real tol = pumipic::compute_tolerance_from_area(elmArea);
  printf("[INFO] Planned tol is %.16f\n", tol);

  const o::Vector<3> o{0.0, -0.2, -0.5};
  const o::Vector<3> z{0.0, -0.2, 0.9};

  printf("[INFO] o: %f %f %f\n", o[0], o[1], o[2]);
  printf("[INFO] z: %f %f %f\n", z[0], z[1], z[2]);

  if (!is_inside3D(mesh, 12, z)) {
    printf("Error: z is not inside element 12.\n");
    Kokkos::finalize();
    exit(1);
  }
  if (!is_inside3D(mesh, 0, o)) {
    printf("Error: o is not inside element 0.\n");
    Kokkos::finalize();
    exit(1);
  }

  // read adjacency data
  const auto& coords = mesh.coords();
  const auto& tet2tri = mesh.ask_down(o::REGION, o::FACE).ab2b;
  const auto& tet2nodes = mesh.ask_verts_of(o::REGION);
  const auto& face2nodes = mesh.ask_verts_of(o::FACE);
  // save the intersections with all the tet faces
  o::Write<o::Real> xpoints(4*3, 0.0, "xpoints xo, yo");
  o::Write<o::LO> face_intersected(4, -1, "face_intersected");

  auto get_intersections = OMEGA_H_LAMBDA(o::LO ray_id) {
    const o::Few<o::LO, 4> tet_faces {tet2tri[4*tet_id], tet2tri[4*tet_id+1], tet2tri[4*tet_id+2], tet2tri[4*tet_id+3]};
    printf("[INFO] Tet Faces are %d %d %d %d\n", tet_faces[0], tet_faces[1], tet_faces[2], tet_faces[3]);
    const auto tet_nodes = o::gather_verts<4>(tet2nodes, tet_id);

    for (int i = 0; i < 4; i++) {
      o::LO face_id = tet_faces[i];
      auto face_nodes = o::gather_verts<3>(face2nodes, face_id);
      auto face_coords = o::gather_vectors<3, 3>(coords, face_nodes);

      o::Vector<3> xpoint;
      o::Real dprodj;
      o::Real closeness;
      o::LO flip = pumipic::isFaceFlipped(i, face_nodes, tet_nodes);

      bool success = success = pumipic::moller_trumbore_line_triangle(
          face_coords, o, z, xpoint, tol, flip, dprodj, closeness);

      xpoints[i * 3 + 0] = xpoint[0];
      xpoints[i * 3 + 1] = xpoint[1];
      xpoints[i * 3 + 2] = xpoint[2];
      face_intersected[i] = success;

      printf("INFO: ray o->z %s face %d(%d,%d,%d) at point (%f, %f, "
             "%f) with dprodj %f and closeness %f\n", (success) ? "intersected" : "did not intersect",
             face_id, face_nodes[0], face_nodes[1], face_nodes[2], xpoint[0],
             xpoint[1], xpoint[2], dprodj, closeness);

    } // for faces
  };
  o::parallel_for(1, get_intersections, "get_intersections_run");

  // * For element 3 : the faces are 58 53 54 56 and 58 and 54 intersects
  // * For element 20: the faces are 39 37 38 54 and only 39   intersects
  // * For element 12: the faces are 28 29 33 39 and only 28   intersects
  // ! It shows that even though the ray intersects face 39 when inside
  // ! element 20, i.e. it enters element 12 through 39 but it doesn't 
  // ! seem to intersect it when it's inside element 12.
  o::Few<o::LO, 4> expected_intersects;
  if (tet_id==3) {
    expected_intersects = {1, 0, 1, 0};
  } else if (tet_id==20){
    expected_intersects = {1, 0, 0, 0};
  } else if (tet_id==12) {
    expected_intersects = {1, 0, 0, 0};
  } else {
    printf("[ERROR] Case not handled yet.\n");
    Kokkos::finalize();
    exit(1);
  }

  // * For face 58 or 54, z of intersection point = 0.8
  // * For face 28,       z of intersection point = 1.0
  o::Real expected_xpoint_z;
  if (tet_id==3){
    expected_xpoint_z = 0.8;
  } else if (tet_id==20){
    expected_xpoint_z = 0.8;
  } else if (tet_id==12){
    expected_xpoint_z = 1.0;
  } else {
    printf("[ERROR] Case not handled yet.\n");
    Kokkos::finalize();
    exit(1);
  }


  auto face_intersected_host = o::HostRead<o::LO>(face_intersected);
  auto xpoints_host = o::HostRead<o::Real>(xpoints);

  bool passed_intersections = (face_intersected_host[0]==expected_intersects[0] && face_intersected_host[1]==expected_intersects[1] &&
                               face_intersected_host[2]==expected_intersects[2] && face_intersected_host[3]==expected_intersects[3]);
  bool passed_xpoints       = (Kokkos::abs(xpoints_host[2]-expected_xpoint_z) < tol);

  if (!passed_intersections){
    printf("[ERROR] It should intersect as %d %d %d %d but found %d %d %d %d\n",
    expected_intersects[0], expected_intersects[1], expected_intersects[2], expected_intersects[3],
    face_intersected_host[0], face_intersected_host[1], face_intersected_host[2], face_intersected_host[3]);
  }

  if (!passed_xpoints){
    printf("[ERROR] It should intersect face at z = %f but intersected at %f\n", expected_xpoint_z, xpoints_host[2]);
  }

  return passed_intersections && passed_xpoints;
}

int main(int argc, char **argv) {
  o::Library lib(&argc, &argv);
  if (argc != 2) {
    printf("Usage: %s <a 3D gmsh>\n", argv[0]);
    exit(1);
  }
  std::string mesh_fname = argv[1];

  bool extrapolated_intersection_test_case12 =
      moller_trumbore_extrapolated_intersection_test(mesh_fname, &lib, 12);
  if (!extrapolated_intersection_test_case12) {
    printf("Extrapolated line intersection test failed for element 12. \n");
  }

  bool extrapolated_intersection_test_case20 =
      moller_trumbore_extrapolated_intersection_test(mesh_fname, &lib, 20);
  if (!extrapolated_intersection_test_case20) {
    printf("Extrapolated line intersection test failed for element 20. \n");
  }

  bool extrapolated_intersection_test_case3 =
      moller_trumbore_extrapolated_intersection_test(mesh_fname, &lib, 3);
  if (!extrapolated_intersection_test_case3) {
    printf("Extrapolated line intersection test failed for element 3. \n");
  }

  bool two_ray_test = moller_trumbore_line_tri_test(mesh_fname, &lib);
  if (!two_ray_test) {
    printf("Test failed\n");
  }

  int all_passed = (extrapolated_intersection_test_case12 &&
                    extrapolated_intersection_test_case3 &&
                    extrapolated_intersection_test_case20 && two_ray_test)
                       ? 0
                       : 1;

  return all_passed;
}
#include <Kokkos_Core.hpp>
#include <Omega_h_bbox.hpp>
#include <Omega_h_defines.hpp>
#include <Omega_h_fail.hpp>
#include <Omega_h_for.hpp>
#include <Omega_h_macros.h>
#include <Omega_h_mesh.hpp>
#include <pumipic_adjacency.hpp>
#include <pumipic_utils.hpp>
#include <string>

namespace o = Omega_h;
namespace p = pumipic;

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
    o::Vector<4> bcc =
        o::barycentric_from_global<3, 3>(point, current_el_vert_coords);
    inside[0] = p::all_positive(bcc, 0.0);
  };
  o::parallel_for(1, is_inside_lambda);
  auto host_inside = o::HostWrite(inside);

  return bool(host_inside[0]);
}

/*
* A new flag is added to the moller_trumbore_line_triangle method to indicate if the line is a segment or a ray.
* The line segment doesn't intersect face 28 (face nodes 5, 11, 4) of element 12, but the ray does.
*/
bool moller_trumbore_line_segment_intersection_test(
    const std::string mesh_fname, o::Library *lib, o::LO tet_id, bool is_line_seg) {
  printf("\n\n--------------------------------------------------------------------\n");
  printf("Test: moller_trumbore_line_tri test for element %d %s ...\n", tet_id, (is_line_seg) ? "line segment" : "ray");
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
          face_coords, o, z, xpoint, tol, flip, dprodj, closeness, is_line_seg);

      xpoints[i * 3 + 0] = xpoint[0];
      xpoints[i * 3 + 1] = xpoint[1];
      xpoints[i * 3 + 2] = xpoint[2];
      face_intersected[i] = success;

      printf("INFO: ray o->z %s face %d(%d (%f %f %f),%d (%f %f %f),%d (%f %f %f)) at point (%f, %f, "
             "%f) with dprodj %f and closeness %f\n", (success) ? "intersected" : "did not intersect",
             face_id, face_nodes[0], face_coords[0][0], face_coords[0][1], face_coords[0][2],
                face_nodes[1], face_coords[1][0], face_coords[1][1], face_coords[1][2],
                face_nodes[2], face_coords[2][0], face_coords[2][1], face_coords[2][2],
                xpoint[0], xpoint[1], xpoint[2], dprodj, closeness);
    } // for faces
  };
  o::parallel_for(1, get_intersections, "get_intersections_run");

  // * For element 12: the faces are 28 29 33 39 and **does not** intersect any face if line segment.
  // * If ray mode is on, it intersects face 28.

  o::Few<o::LO, 4> expected_intersects;
  if (tet_id == 12) {
    if (is_line_seg) {
      expected_intersects = {0, 0, 0, 0};
    } else {
        expected_intersects = {1, 0, 0, 0};
    }
  } else {
    printf("[ERROR] Case not handled yet.\n");
    Kokkos::finalize();
    exit(1);
  }

  // * For face 28(although it doesn't intersect), z of intersection point = 1.0
  o::Real expected_xpoint_z;
  if (tet_id==12){
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

  bool line_segment_test_case = moller_trumbore_line_segment_intersection_test(mesh_fname, &lib, 12, true);
  if (!line_segment_test_case) {
    printf("[ERROR] Line segment test failed.\n");
  }

  bool ray_test_case = moller_trumbore_line_segment_intersection_test(mesh_fname, &lib, 12, false);
  if (!ray_test_case) {
    printf("[ERROR] Ray test failed.\n");
  }

  int all_passed = (line_segment_test_case && ray_test_case) ? 0 : 1;
  return all_passed;
}
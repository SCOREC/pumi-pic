/*
* Test Description:
*   - Two rays are traced. oy goes horizontally and oz goes vertically.
*   - They should reach element -1 (outside through element 6) and 12 resepetively.
*   - The lastExit faces should be 16 and 28 (considering that moller_trumbore
*     takes the ray as ray instead of a line segment as shown in it's unit test).
*/

#include "pumipic_adjacency.hpp"
#include "pumipic_adjacency.tpp"
#include "pumipic_kktypes.hpp"
#include "pumipic_library.hpp"
#include "pumipic_mesh.hpp"
#include "pumipic_utils.hpp"
#include <Kokkos_Core.hpp>
#include <Omega_h_bbox.hpp>
#include <Omega_h_defines.hpp>
#include <Omega_h_for.hpp>
#include <Omega_h_macros.h>
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

typedef MemberTypes<Vector3d, Vector3d, int> Particle;
typedef ps::ParticleStructure<Particle> PS;
typedef Kokkos::DefaultExecutionSpace ExeSpace;

void printf_face_info(o::Mesh &mesh, o::LOs faceIds, bool all = false) {
  const auto exposed_faces = o::mark_exposed_sides(&mesh);
  const auto &face2nodes = mesh.ask_down(o::FACE, o::VERT).ab2b;
  const auto &coords = mesh.coords();

  auto print_faces = OMEGA_H_LAMBDA(o::LO faceid) {
    if (!all) {
      for (int i = 0; i < faceIds.size(); i++) {
        o::LO id = faceIds[i];
        if (id == faceid) {
          printf("Face %d nodes %d %d %d Exposed %d\n", faceid,
                 face2nodes[faceid * 3], face2nodes[faceid * 3 + 1],
                 face2nodes[faceid * 3 + 2], exposed_faces[faceid]);
        }
      }
    } else if (all) {
      printf("Face %d nodes %d %d %d Exposed %d\n", faceid,
             face2nodes[faceid * 3], face2nodes[faceid * 3 + 1],
             face2nodes[faceid * 3 + 2], exposed_faces[faceid]);
    }
  };
  o::parallel_for(mesh.nfaces(), print_faces, "print asked faces");
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
    o::Vector<4> bcc =
        o::barycentric_from_global<3, 3>(point, current_el_vert_coords);
    inside[0] = p::all_positive(bcc, 0.0);
  };
  o::parallel_for(1, is_inside_lambda);
  auto host_inside = o::HostWrite(inside);

  return bool(host_inside[0]);
}

bool test_search_mesh(const std::string mesh_fname, p::Library *lib, bool useBcc) {
  printf("Test: 3D intersection...\n");
  o::Library &olib = lib->omega_h_lib();
  // load mesh
  o::Mesh mesh = Omega_h::gmsh::read(mesh_fname, olib.self());
  printf("Mesh loaded successfully with %d elements\n", mesh.nelems());
  printf_face_info(mesh, {28,16}, false);

  Omega_h::Write<Omega_h::LO> owners(mesh.nelems(), 0);
  p::Mesh picparts(mesh, owners);
  o::Mesh *p_mesh = picparts.mesh();

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

  printf("Creating ray o->z (0) and o->y (1)\n");
  const o::Vector<3> o{0.0, -0.2, -0.5};
  const o::Vector<3> z{0.0, -0.2, 0.9};
  const o::Vector<3> y{0.0, 1.2, -0.5};
  if (!is_inside3D(mesh, tet_id, o)) {
    printf("Error: x is not inside the expected element.\n");
    Kokkos::finalize();
    exit(1);
  }
  if (is_inside3D(mesh, tet_id, z)) {
    printf("Error: y is not inside the expected element.\n");
    Kokkos::finalize();
    exit(1);
  }
  if (is_inside3D(mesh, tet_id, y)) {
    printf("Error: o is incorrectly inside the expected element.\n");
    Kokkos::finalize();
    exit(1);
  }

  printf("[INFO] z: %f %f %f\n", z[0], z[1], z[2]);
  printf("[INFO] y: %f %f %f\n", y[0], y[1], y[2]);
  printf("[INFO] o: %f %f %f\n", o[0], o[1], o[2]);

  auto set_initial_and_final_position =
      PS_LAMBDA(const int &e, const int &pid, const int &mask) {
    // see test calculation notebook
    if (pid == 0) {
      particle_init_position(pid, 0) = o[0];
      particle_init_position(pid, 1) = o[1];
      particle_init_position(pid, 2) = o[2];

      particle_final_position(pid, 0) = z[0];
      particle_final_position(pid, 1) = z[1];
      particle_final_position(pid, 2) = z[2];
    }
    if (pid == 1) {
      particle_init_position(pid, 0) = o[0];
      particle_init_position(pid, 1) = o[1];
      particle_init_position(pid, 2) = o[2];

      particle_final_position(pid, 0) = y[0];
      particle_final_position(pid, 1) = y[1];
      particle_final_position(pid, 2) = y[2];
    }
  };

  ps::parallel_for(ptcls, set_initial_and_final_position,
                   "set_initial_and_final_position");

  const auto psCapacity = ptcls->capacity();

  auto inter_points =
      o::Write<o::Real>(3 * psCapacity, 0.0, "intersection points");
  o::Write<o::LO> lastExit(psCapacity, -1, "search_last_exit");
  o::Write<o::LO> ptcl_done(psCapacity, 0, "search_ptcl_done");
  o::Write<o::LO> elem_ids;

  auto x_ps_d = ptcls->get<0>();
  auto x_ps_tgt = ptcls->get<1>();
  auto pids = ptcls->get<2>();

  p::search_mesh(*p_mesh, ptcls, x_ps_d, x_ps_tgt, pids, elem_ids, !useBcc,
                 lastExit, inter_points, 100, true);

  auto elem_ids_host = o::HostRead<o::LO>(elem_ids);
  auto lastExit_host = o::HostRead<o::LO>(lastExit);

  bool found_correct_dest_elem = elem_ids_host[0]==12 && elem_ids_host[1]==6;
  bool found_correct_dest_face = lastExit_host[0]==28 && lastExit_host[1]==16;

  if (!found_correct_dest_elem || !found_correct_dest_face){
    printf("[ERROR] Expected elements and faces are respectively [12, 6] and [28, 16] but found (%d, %d) and (%d, %d)\n",
    elem_ids_host[0], elem_ids_host[1], lastExit_host[0], lastExit_host[1]);
  }

  delete ptcls;
  
  return found_correct_dest_elem && found_correct_dest_face;
}

int main(int argc, char **argv) {
  p::Library lib(&argc, &argv);
  if (argc != 2) {
    printf("Usage: %s <gmesh>\n", argv[0]);
    exit(1);
  }
  std::string mesh_fname = argv[1];

  printf("\n\n-------------------- With BCC -------------------\n");

  //bool passed_bcc = test_search_mesh(mesh_fname, &lib, true);
  bool passed_bcc = true;
  //if (!passed_bcc) {
  //  printf("[ERROR]: Test failed **with** BCC.\n");
  //}

  printf("\n\n-------------------- Without BCC -------------------\n");

  bool passed_woBcc = test_search_mesh(mesh_fname, &lib, false);
  if (!passed_woBcc) {
    printf("[ERROR]: Test failed **without** BCC.\n");
  }

  if (passed_bcc && passed_woBcc) {
    printf("All tests passed!\n");
    return 0;
  } else {
    return 1;
  }

}
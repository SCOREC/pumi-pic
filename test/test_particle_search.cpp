/*
 * Fuad's two-trajectory Gmsh regression: one particle stops inside element 12,
 * and one leaves the domain through face 16. Under the line-segment API, a
 * particle reaching its destination has no exit face (-1).
 */

#include "particle_structure.hpp"
#include "ppMacros.h"
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
#include <Omega_h_mark.hpp>
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

void apply_vacuum_boundary_condition(o::Mesh &mesh, PS *ptcls,
                                  o::Write<o::LO>& elem_ids,
                                  o::Write<o::LO>& ptcl_done,
                                  o::Write<o::LO>& lastExit,
                                  o::Write<o::LO>& xFace) {
  const auto& side_is_exposed = o::mark_exposed_sides(&mesh);

  auto checkExposedEdges = PS_LAMBDA(const int e, const int pid, const int mask){
    if (mask > 0 && !ptcl_done[pid]){
        printf("lastExit[%d] = %d\n", pid, lastExit[pid]);
        if (lastExit[pid] == -1) {
            ptcl_done[pid] = 1;
        } else {
            assert(lastExit[pid] != -1);
            const o::LO bridge = lastExit[pid];
            const bool exposed = side_is_exposed[bridge];
            ptcl_done[pid] = exposed;
            xFace[pid] = lastExit[pid];
            elem_ids[pid] = exposed ? -1 : elem_ids[pid];
        }
    }
  };
  p::parallel_for(ptcls, checkExposedEdges, "apply vacumm boundary condition");
}

void move_to_new_element(o::Mesh &mesh, PS *ptcls, o::Write<o::LO>& elem_ids,
                         o::Write<o::LO>& ptcl_done, o::Write<o::LO>& lastExit) {
  const int dim = mesh.dim();
  const auto& face2elems = mesh.ask_up(dim-1, dim);
  const auto& face2elemElem = face2elems.ab2b;
  const auto& face2elemOffset = face2elems.a2ab;

  auto set_next_element = PS_LAMBDA(const int &e, const int &pid, const int &mask) {
    if (mask > 0 && !ptcl_done[pid]) {
      auto searchElm = elem_ids[pid];
      auto bridge = lastExit[pid];
      auto e2f_first = face2elemOffset[bridge];
      auto e2f_last = face2elemOffset[bridge + 1];
      auto upFaces = e2f_last - e2f_first;
      assert(upFaces == 2);
      auto faceA = face2elemElem[e2f_first];
      auto faceB = face2elemElem[e2f_first + 1];
      assert(faceA != faceB);
      assert(faceA == searchElm || faceB == searchElm);
      auto nextElm = (faceA == searchElm) ? faceB : faceA;
      elem_ids[pid] = nextElm;
    }
  };
  parallel_for(ptcls, set_next_element, "pumipic_set_next_element");
}

template <typename Segment>
void handle_particle_at_elem_boundary(o::Mesh &mesh, PS *ptcls, o::Write<o::LO> &elem_ids, o::Write<o::LO> &next_elements,
               o::Write<o::LO> &inter_faces, o::Write<o::LO> &lastExit, o::Write<o::Real> &inter_points,
               o::Write<o::LO> &ptcl_done, Segment origin, Segment destination) {
  apply_vacuum_boundary_condition(mesh, ptcls, elem_ids, ptcl_done, lastExit, inter_faces);
  move_to_new_element(mesh, ptcls, elem_ids, ptcl_done, lastExit);
}

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

// Namespace-scope callable types can enclose CUDA extended lambdas.
struct VacuumBoundaryHandler {
  template <typename Segment>
  void operator()(o::Mesh& mesh, PS* particles, o::Write<o::LO>& elements,
                  o::Write<o::LO>& next, o::Write<o::LO>& faces,
                  o::Write<o::LO>& exits, o::Write<o::Real>& points,
                  o::Write<o::LO>& done, Segment origin, Segment destination) const {
    handle_particle_at_elem_boundary(mesh, particles, elements, next, faces,
                                    exits, points, done, origin, destination);
  }
};

bool test_particle_search(const std::string mesh_fname, p::Library *lib, bool useBcc) {
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

  o::Write<o::LO> next_elements;
  o::Write<o::LO> inter_faces;
  VacuumBoundaryHandler handler;
  auto elm_area = measure_elements_real(p_mesh);
  bool found = p::trace_particle_through_mesh(*p_mesh, ptcls, x_ps_d, x_ps_tgt, pids,
                 elem_ids, next_elements, true, inter_faces, inter_points, lastExit,
                 100, true, handler, elm_area, ptcl_done);

  auto elem_ids_host = o::HostRead<o::LO>(elem_ids);
  auto lastExit_host = o::HostRead<o::LO>(lastExit);

  bool found_correct_dest_elem = elem_ids_host[0]==12 && elem_ids_host[1]==-1;
  bool found_correct_dest_face = lastExit_host[0]==-1 && lastExit_host[1]==16;

  if (!found_correct_dest_elem || !found_correct_dest_face){
    printf("[ERROR] Expected elements and faces are respectively [12, -1(leak out)] and [-1, 16] but found (%d, %d) and (%d, %d)\n",
    elem_ids_host[0], elem_ids_host[1], lastExit_host[0], lastExit_host[1]);
  } else {
    printf("[PASSSED] Found Correct destination elements (%d %d) and faces (%d %d) for particle 0 and 1 respectively.\n",
    elem_ids_host[0], elem_ids_host[1], lastExit_host[0], lastExit_host[1]);
  }

  delete ptcls;

  return found && found_correct_dest_elem && found_correct_dest_face;
}

int main(int argc, char **argv) {
  p::Library lib(&argc, &argv);
  if (argc != 2) {
    printf("Usage: %s <gmesh>\n", argv[0]);
    return 1;
  }
  return test_particle_search(argv[1], &lib, false) ? 0 : 1;
}

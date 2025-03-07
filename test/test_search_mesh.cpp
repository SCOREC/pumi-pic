//
// Created by Fuad Hasan on 2/25/25.
//

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

void print_test_info();
void printf_face_info(o::Mesh &mesh, o::LOs faceIds, bool all);
bool is_inside3D(o::Mesh &mesh, o::LO elem_id, const o::Vector<3> point);

bool is_close(const double a, const double b, double tol = 1e-8) {
    return std::abs(a - b) < tol;
}

OMEGA_H_INLINE bool is_close_d(const double a, const double b, double tol = 1e-8) {
    return Kokkos::abs(a - b) < tol;
}


int main(int argc, char* argv[]){
    // ----------------------------------------------------------------------------------//
    // ----------------------------------- Set up ---------------------------------------//
    // ----------------------------------------------------------------------------------//

    print_test_info();

    printf("\n============================ Setting up Mesh and Particles ======================\n");
    p::Library lib(&argc, &argv);
    auto &olib = lib.omega_h_lib();
    auto world = olib.world();
    // simplest 3D mesh
    auto mesh =
            Omega_h::build_box(world, OMEGA_H_SIMPLEX, 1, 1, 1, 1, 1, 1, false);
    printf("[INFO] Mesh created with %d vertices and %d faces\n",
           mesh.nverts(), mesh.nfaces());

    printf_face_info(mesh, {}, true);

    Omega_h::Write<Omega_h::LO> owners(mesh.nelems(), 0);
    p::Mesh picparts(mesh, owners);
    o::Mesh *p_mesh = picparts.mesh();
    // create particle structure with 5 particles
    int num_ptcls = 5;

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
            ne, OMEGA_H_LAMBDA(
                    const int &i) {
                element_gids(i) = mesh_element_gids[i];
                if (i == tet_id) {
                    ptcls_per_elem(i) = num_ptcls;
                } else {
                    ptcls_per_elem(i) = 0;
                }
            });

#ifdef PP_ENABLE_CAB
    PS *ptcls = new p::DPS<Particle>(policy, ne, num_ptcls, ptcls_per_elem, element_gids);
    printf("DPS Particle structure created successfully\n");
#else
    PS *ptcls = PS *ptcls = new SellCSigma<Particle>(
      policy, 10, 10, ne, 2, ptcls_per_elem, element_gids);
  printf("SellCSigma Particle structure created successfully\n");
#endif

  printf("\n====================== Mesh and Particles are set up ============================\n");

    // ----------------------------------------------------------------------------------//
    // ------------------------------- Particle Movement ---------------------------------//
    // ----------------------------------------------------------------------------------//

    printf("\n\n\n============================ Particle Movement Setup =================================\n");

    // The following are verified using: https://gist.github.com/Fuad-HH/5e0aed99f271617e283e9108091fb1cb
    o::Vector<3> particle_initial_loc {0.5, 0.75, 0.25};
    o::Real tiny_movement = 1e-9;
    o::Vector<3> p0_dest = particle_initial_loc + o::Vector<3>({tiny_movement, tiny_movement, tiny_movement}); // moved a little bit
    o::Vector<3> p1_dest = {0.8, 0.9, 0.1}; // inside the same element
    o::Vector<3> p2_dest = {0.8, 0.1, 0.95}; // moved to element 3
    // p2 intersections :   1. [0.57894737 0.57894737 0.43421053]
    //                      2. [0.61111111 0.50925926 0.50925926]
    //                      3. [0.6875     0.34375    0.6875    ]
    o::Vector<3> p3_dest = {-0.5, 0.75, 0.25}; // moved out of the mesh through element 1
    // p3 intersections :   1. [0.25 0.75 0.25]
    //                      2. [0.   0.75 0.25]
    o::Vector<3> p4_dest = {0.5, 0.75, -1.5}; // moved out of the mesh through face 0 of element 0
    // p4 intersections :   1. [0.5 0.75 0]


    // Set particle positions
    auto particle_orig = ptcls->get<0>();
    auto particle_dest = ptcls->get<1>();
    auto particle_id   = ptcls->get<2>();

    auto set_particle_path = PS_LAMBDA(const int& eid, const int& pid, const int& mask) {
        if (mask > 0) {
            particle_id(pid) = pid;

            particle_orig(pid, 0) = particle_initial_loc[0];
            particle_orig(pid, 1) = particle_initial_loc[1];
            particle_orig(pid, 2) = particle_initial_loc[2];

            if (pid == 0) {
                particle_dest(pid, 0) = p0_dest[0];
                particle_dest(pid, 1) = p0_dest[1];
                particle_dest(pid, 2) = p0_dest[2];
            } else if (pid == 1) {
                particle_dest(pid, 0) = p1_dest[0];
                particle_dest(pid, 1) = p1_dest[1];
                particle_dest(pid, 2) = p1_dest[2];
            } else if (pid == 2) {
                particle_dest(pid, 0) = p2_dest[0];
                particle_dest(pid, 1) = p2_dest[1];
                particle_dest(pid, 2) = p2_dest[2];
            } else if (pid == 3) {
                particle_dest(pid, 0) = p3_dest[0];
                particle_dest(pid, 1) = p3_dest[1];
                particle_dest(pid, 2) = p3_dest[2];
            } else if (pid == 4) {
                particle_dest(pid, 0) = p4_dest[0];
                particle_dest(pid, 1) = p4_dest[1];
                particle_dest(pid, 2) = p4_dest[2];
            } else {
                printf("[ERROR] Particle id %d is not supported\n", pid);
            }

            printf("Particle id %d starts at (%f, %f, %f) and moves to (%f, %f, %f)\n",
                   pid, particle_orig(pid, 0), particle_orig(pid, 1), particle_orig(pid, 2),
                   particle_dest(pid, 0), particle_dest(pid, 1), particle_dest(pid, 2));
        }
    };
    pumipic::parallel_for(ptcls, set_particle_path, "set_particle_path");

    // -------------------------------------- Particle Movement Set up Done --------------------------------------//
    auto elem_ids = Omega_h::Write<Omega_h::LO> (0, "elem ids");
    bool requireIntersection = true;
    auto interFaces = Omega_h::Write<Omega_h::LO> (0, "inter faces");
    auto interPoints = Omega_h::Write<Omega_h::Real> (0, "inter points");

    bool success = pumipic::search_mesh(mesh, ptcls, particle_orig, particle_dest, particle_id, elem_ids, requireIntersection, interFaces, interPoints, 100, 1);
    if (!success) {
        printf("[ERROR] search_mesh failed\n");
        delete ptcls;
        throw std::runtime_error("search_mesh failed");
    }


    // ----------------------------------- Check Results -----------------------------------//
    // check sizes
    OMEGA_H_CHECK_PRINTF(elem_ids.size() == ptcls->capacity(), "elem_ids size %d != ptcls capacity %d\n",
                         elem_ids.size(), ptcls->capacity());
    OMEGA_H_CHECK_PRINTF(interFaces.size() == ptcls->capacity(), "interFaces size %d != ptcls capacity %d\n",
                         interFaces.size(), ptcls->capacity());
    OMEGA_H_CHECK_PRINTF(interPoints.size() == 3*ptcls->capacity(), "interPoints size %d != 3 x ptcls capacity %d\n",
                         interPoints.size(), 3*ptcls->capacity());

    Omega_h::Few<Omega_h::LO, 5> expected_elem_ids = {0, 0, 3, -1, -1};
    Omega_h::Few<Omega_h::LO, 5> expected_interFaces = {-1, -1, -1, 1, 0};
    // p3 and p4 reaches the boundary
    auto p3_at_boundary = p3_dest; p3_at_boundary[0] = 0.0;
    auto p4_at_boundary = p4_dest; p4_at_boundary[2] = 0.0;
    Omega_h::Few<Omega_h::Vector<3>, 5> expected_interPoints = {p0_dest, p1_dest, p2_dest, p3_at_boundary, p4_at_boundary};

    printf("\n\n\n============================ Checking Search Results =================================\n");
    auto check_search_results = PS_LAMBDA(const auto e, const auto pid, const auto mask) {
        if (mask > 0) {
        printf("\tPid %d: elem_id (%2d, %2d), interFace (%2d, %2d), interPoint ([% .10f, % .10f, % .10f],[% .10f, % .10f, % .10f])\n",
                   pid, elem_ids[pid], expected_elem_ids[pid], interFaces[pid], expected_interFaces[pid],
                   interPoints[3*pid], interPoints[3*pid+1], interPoints[3*pid+2],
                   expected_interPoints[pid][0], expected_interPoints[pid][1], expected_interPoints[pid][2]);

        OMEGA_H_CHECK_PRINTF(elem_ids[pid] == expected_elem_ids[pid], "Particle %d: Expected elem_id %d != found elem_id %d\n",
                             pid, expected_elem_ids[pid], elem_ids[pid]);
        OMEGA_H_CHECK_PRINTF(interFaces[pid] == expected_interFaces[pid], "Particle %d: Expected interFace %d != found interFace %d\n",
                             pid, expected_interFaces[pid], interFaces[pid]);
        OMEGA_H_CHECK_PRINTF(is_close_d(interPoints[3*pid], expected_interPoints[pid][0], 1e-10),
                             "Particle %d: Expected interPoint x %f != found interPoint x %f\n",
                             pid, expected_interPoints[pid][0], interPoints[3*pid]);
        OMEGA_H_CHECK_PRINTF(is_close_d(interPoints[3*pid+1], expected_interPoints[pid][1], 1e-10),
                             "Particle %d: Expected interPoint y %f != found interPoint y %f\n",
                             pid, expected_interPoints[pid][1], interPoints[3*pid+1]);
        OMEGA_H_CHECK_PRINTF(is_close_d(interPoints[3*pid+2], expected_interPoints[pid][2], 1e-10),
                             "Particle %d: Expected interPoint z %f != found interPoint z %f\n",
                             pid, expected_interPoints[pid][2], interPoints[3*pid+2]);
        }
    };
    pumipic::parallel_for(ptcls, check_search_results, "check_search_results");
    printf("\n============================ Search Results are Correct =================================\n");



    // ------------------------------------------- Clean up -------------------------------------------//
    delete ptcls;

    return 0;
}

void print_test_info() {
    printf("\n\n!!!!!!!!!!!!!!!!!!!!!!!!!!!! Important !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n"
           "!!!!!!!!!!!!!!! This test only works when compiled in DEBUG mode !!!!!!!!!!!!!!!!\n\n");

    std::string message = "\n========================== Test Search Mesh Function ======================\n";
    message += "Testing 'search_mesh' function form pumipic_adjacency.tpp ...\n";
    message += "A 3D internal mesh is created and particles will be moved to different locations\n";
    message += "or will be kept in the same element or same position.\n";
    message += "Particle movements are: \n";
    message += "\t1. Inside the same element at same position\n";
    message += "\t2. Inside the same element but at different position\n";
    message += "\t3. Different element's interior\n";
    message += "\t4. Moves out of the mesh traversing multiple elements\n";
    message += "\t5. Moves to the boundary from the same element it started\n";
    message += "\n===============================================================\n";
    printf("%s", message.c_str());
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

void printf_face_info(o::Mesh &mesh, o::LOs faceIds, bool all) {
    const auto exposed_faces = o::mark_exposed_sides(&mesh);
    const auto &face2nodes = mesh.ask_down(o::FACE, o::VERT).ab2b;
    const auto &face2cellcell = mesh.ask_up(o::FACE, o::REGION).ab2b;
    const auto &face2celloffset = mesh.ask_up(o::FACE, o::REGION).a2ab;
    const auto &coords = mesh.coords();

    auto print_faces = OMEGA_H_LAMBDA(o::LO
                                      faceid) {
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
            int n_adj_cells = face2celloffset[faceid+1] - face2celloffset[faceid];
            if(n_adj_cells == 1){
                printf("Face %d is exposed and with %d cell\n", faceid, face2cellcell[face2celloffset[faceid]]);
            } else if (n_adj_cells ==2) {
                printf("Face %d internal with %d %d\n", faceid, face2cellcell[face2celloffset[faceid]],
                       face2cellcell[face2celloffset[faceid] + 1]);
            } else {
                printf("[ERROR] Face cannot be adjacent to more than 2 cells\n found %d\n", n_adj_cells);
            }
        }
    };
    o::parallel_for(mesh.nfaces(), print_faces, "print asked faces");
}

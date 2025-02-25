//
// Created by Fuad Hasan on 2/12/25.
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

void printf_face_info(o::Mesh &mesh, o::LOs faceIds, bool all = false);

void print_test_info();

bool is_close(const double a, const double b, double tol = 1e-8) {
    return std::abs(a - b) < tol;
}

OMEGA_H_INLINE bool is_close_d(const double a, const double b, double tol = 1e-8) {
    return Kokkos::abs(a - b) < tol;
}

template<typename ParticleType, typename Segment3d>
struct empty_function {
    void operator()(
            o::Mesh &mesh, ps::ParticleStructure<ParticleType> *ptcls,
            o::Write<o::LO> &elem_ids, o::Write<o::LO>& next_elements, o::Write<o::LO> &inter_faces,
            o::Write<o::LO> &lastExit, o::Write<o::Real> &inter_points,
            o::Write<o::LO> &ptcl_done,
            Segment3d x_ps_orig,
            Segment3d x_ps_tgt) const {
        printf("Empty Function Called\n");
    }
};


int main(int argc, char **argv) {
    // **************************************** Setup ***************************************************************//
    p::Library lib(&argc, &argv);
    print_test_info();
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

    printf("\n\n\n==============================================================================================\n");
    printf(">============ Checking when particles move from cell 0 to 5 =================================<\n");
    printf("==============================================================================================\n");

    // set particle position
    Omega_h::Vector<3> cell0_centroid{0.5, 0.75, 0.25};
    Omega_h::Vector<3> cell5_centroid{0.75, 0.5, 0.25};
    auto particle_init_position = ptcls->get<0>();
    auto particle_final_position = ptcls->get<1>();
    auto pid_d = ptcls->get<2>();
    auto setIDs = PS_LAMBDA(
            const int &eid,
            const int &pid,
            const bool &mask) {
        if (mask > 0) {
            particle_init_position(pid, 0) = cell0_centroid[0];
            particle_init_position(pid, 1) = cell0_centroid[1];
            particle_init_position(pid, 2) = cell0_centroid[2];

            particle_final_position(pid, 0) = cell5_centroid[0];
            particle_final_position(pid, 1) = cell5_centroid[1];
            particle_final_position(pid, 2) = cell5_centroid[2];

            pid_d(pid) = pid;


            printf("Initialized particle %d origin (%f, %f, %f) and destination (%f, %f, %f)\n",
                   pid,
                   particle_init_position(pid, 0), particle_init_position(pid, 1), particle_init_position(pid, 2),
                   particle_final_position(pid, 0), particle_final_position(pid, 1), particle_final_position(pid, 2));
        }
    };
    ps::parallel_for(ptcls, setIDs);
    // ********************************************** Particle Setup Done ******************************************//

    // **************************************** Set up Auxiliary Search Arrays *************************************//
    const bool requireIntersection = true;

    Omega_h::Write<Omega_h::LO> elem_ids(0, "element ids");
    Omega_h::Write<Omega_h::LO> inter_faces(0, "inter faces");
    Omega_h::Write<Omega_h::LO> last_exit(0, "last exit");
    Omega_h::Write<Omega_h::Real> inter_points(0, "inter points");
    Omega_h::Write<Omega_h::LO> next_elements(0, "next elements");

    o::Reals elmArea = measure_elements_real(&mesh);
    o::Real tol = pumipic::compute_tolerance_from_area(elmArea);

    // *********************************************** Run The Search *********************************************//
    empty_function<Particle,typeof(particle_final_position)> emptyFunction;

    printf("*** Searching ... ***\n");
    // After a single search operation, auxiliary arrays will be filled and they will be tested
    bool success = pumipic::trace_particle_through_mesh(mesh, ptcls, particle_init_position, particle_final_position,
                    pid_d, elem_ids, next_elements, requireIntersection, inter_faces, inter_points, last_exit, 1, true, emptyFunction, elmArea, tol);
    printf("*** Search Done ***\n");
    if (success){
        printf("[ERROR] Search Shouldn't pass...\n");
    }
    else {
        printf("Particle search failed as expected...\n");
    }

    // ******************************************* Checks *********************************************************//
    OMEGA_H_CHECK_PRINTF(inter_faces.size() == ptcls->capacity(), "inter faces and ptcls capacity mismatch(%d,%d)",
                         inter_faces.size(), ptcls->capacity());
    OMEGA_H_CHECK_PRINTF(elem_ids.size() == ptcls->capacity(), "elem ids and ptcls capacity mismatch(%d,%d)",
                         elem_ids.size(), ptcls->capacity());
    OMEGA_H_CHECK_PRINTF(elem_ids.size() == next_elements.size(), "elem ids and next elements size mismatch(%d,%d)",
                         elem_ids.size(), next_elements.size());

    Omega_h::Vector<3> expected_intersection {0.75, 0.5, 0.25};
    auto check_arrays = PS_LAMBDA(const int& e, const int& pid, const int& mask){
        if (mask>0) {
            printf("Pid %d Intersection Face %d\n", pid, inter_faces[pid]);
            printf("Pid %d Last Exit %d\n", pid, last_exit[pid]);
            OMEGA_H_CHECK_PRINTF(last_exit[pid] == 10, "Expected face 10 between cell 0 and 5 but found %d\n", last_exit[pid]);

            printf("Pid %d intersects at (%f, %f, %f)\n", pid, inter_points[pid*3], inter_points[pid*3+1], inter_points[pid*3+2]);
            OMEGA_H_CHECK_PRINTF(is_close_d(inter_points[pid*3], expected_intersection[0]), "Expected %f, found %f\n", expected_intersection[0], inter_points[pid*3]);
            OMEGA_H_CHECK_PRINTF(is_close_d(inter_points[pid*3+1], expected_intersection[1]), "Expected %f, found %f\n", expected_intersection[1], inter_points[pid*3+1]);
            OMEGA_H_CHECK_PRINTF(is_close_d(inter_points[pid*3+2], expected_intersection[2]), "Expected %f, found %f\n", expected_intersection[2], inter_points[pid*3+2]);

            printf("Pid %d Elem Id %d\n", pid, elem_ids[pid]);
            OMEGA_H_CHECK_PRINTF(elem_ids[pid] == 0, "Expected element 0 but found %d\n", elem_ids[pid]);

            printf("Pid %d Next Element %d\n", pid, next_elements[pid]);
            OMEGA_H_CHECK_PRINTF(next_elements[pid] == 5, "Expected next element 5 but found %d\n", next_elements[pid]);
        }
    };
    pumipic::parallel_for(ptcls, check_arrays, "check arrays");

    printf("============================ Particle Moving for 0 to 5 Passed ===============================\n");
    printf("==============================================================================================\n");


    // *************************************************************************************************************//


    // *************************************************** Check When Particles remain in the same element *********//
    printf("\n\n\n==============================================================================================\n");
    printf(">============ Checking when particles remain in the same element ============================<\n");
    printf("==============================================================================================\n");

    // set particle destination to the same element
    Omega_h::Vector<3> another_location_in_cell0{0.5, 0.75, 0.2};

    auto set_new_dest_in_cell0 = PS_LAMBDA (const int& e, const int& pid, const int& mask){
        if (mask > 0) {
            particle_final_position(pid, 0) = another_location_in_cell0[0];
            particle_final_position(pid, 1) = another_location_in_cell0[1];
            particle_final_position(pid, 2) = another_location_in_cell0[2];

            printf("New particle %d destination (%f, %f, %f)\n",
            pid, particle_final_position(pid, 0),
            particle_final_position(pid, 1),
            particle_final_position(pid, 2));
        }
    };
    pumipic::parallel_for(ptcls, set_new_dest_in_cell0, "set new destination in cell 0");


    // reset auxiliary arrays
    elem_ids = Omega_h::Write<Omega_h::LO>(0, "element ids");
    inter_faces = Omega_h::Write<Omega_h::LO>(0, "inter faces");
    last_exit = Omega_h::Write<Omega_h::LO>(0, "last exit");
    inter_points = Omega_h::Write<Omega_h::Real>(0, "inter points");
    next_elements = Omega_h::Write<Omega_h::LO>(0, "next elements");

    printf("*** Searching ... ***\n");
    // run the search again
    success = pumipic::trace_particle_through_mesh(mesh, ptcls, particle_init_position, particle_final_position,
                    pid_d, elem_ids, next_elements, requireIntersection, inter_faces, inter_points, last_exit, 1, true, emptyFunction, elmArea, tol);
    printf("*** Search Done ***\n");

    if (success){
        printf("[ERROR] Search Shouldn't return success...\n");
    }
    else {
        printf("Particle search failed as expected...\n");
    }

    // ******************************************* Checks *********************************************************//
    OMEGA_H_CHECK_PRINTF(inter_faces.size() == ptcls->capacity(), "inter faces and ptcls capacity mismatch(%d,%d)",
                         inter_faces.size(), ptcls->capacity());
    OMEGA_H_CHECK_PRINTF(elem_ids.size() == ptcls->capacity(), "elem ids and ptcls capacity mismatch(%d,%d)",
                         elem_ids.size(), ptcls->capacity());
    OMEGA_H_CHECK_PRINTF(elem_ids.size() == next_elements.size(), "elem ids and next elements size mismatch(%d,%d)",
                         elem_ids.size(), next_elements.size());

    // expected intersection point is the same as the particle destination
    auto check_move_in_same_element = PS_LAMBDA(const int& e, const int& pid, const int& mask){
        if (mask>0) {
            printf("Pid %d Intersection Face %d\n", pid, inter_faces[pid]);
            printf("Pid %d Last Exit %d\n", pid, last_exit[pid]);
            OMEGA_H_CHECK_PRINTF(last_exit[pid] == -1, "Expected no intersection but found %d\n", last_exit[pid]);

            printf("Pid %d Elem Id %d\n", pid, elem_ids[pid]);
            OMEGA_H_CHECK_PRINTF(elem_ids[pid] == 0, "Expected element 0 but found %d\n", elem_ids[pid]);

            printf("Pid %d intersects (reaches) at (%f, %f, %f) and expected (%f, %f, %f)\n",
                   pid,
                   inter_points[pid*3], inter_points[pid*3+1], inter_points[pid*3+2],
                   another_location_in_cell0[0], another_location_in_cell0[1], another_location_in_cell0[2]);

            OMEGA_H_CHECK_PRINTF(is_close_d(inter_points[pid*3], another_location_in_cell0[0]), "Expected %f, found %f\n", another_location_in_cell0[0], inter_points[pid * 3]);
            OMEGA_H_CHECK_PRINTF(is_close_d(inter_points[pid*3+1], another_location_in_cell0[1]), "Expected %f, found %f\n", another_location_in_cell0[1], inter_points[pid * 3 + 1]);
            OMEGA_H_CHECK_PRINTF(is_close_d(inter_points[pid*3+2], another_location_in_cell0[2]), "Expected %f, found %f\n", another_location_in_cell0[2], inter_points[pid * 3 + 2]);

            printf("Pid %d Next Element %d\n", pid, next_elements[pid]);
            OMEGA_H_CHECK_PRINTF(next_elements[pid] == 0, "Expected next element 0 but found %d\n", next_elements[pid]);
        }
    };
    pumipic::parallel_for(ptcls, check_move_in_same_element, "check when particles move inside the same element as the origin(0 here)");

    printf("============================ Particle Moving from 0 to 0 Passed ==============================\n");
    printf("==============================================================================================\n");



    // ******************************************* Clean Up *********************************************************//
    delete ptcls;
    return 0;
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

void print_test_info() {
    std::string message = "\n========================== Test Trace Particle ======================\n";
    message += "Testing 'trace_particle_through_mesh' function form pumipic_adjacency.tpp ...\n";
    message += "A 3D internal mesh is created and particles will be moved to a new element\n";
    message += "or will be kept in the same element.\n";
    message += "The boundary handler won't be implemeted here but the main function will do minimum\n";
    message += "to keep the particles moving\n";
    message += "\n===============================================================\n";
    printf("%s", message.c_str());
}